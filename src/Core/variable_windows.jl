# this contains the meat of the optimization tooling 

"""
Populate a WindowSpec struct (see types.jl)
Args:
    - t_start: the start time of the window
    - window_T: the length of the window
    - n_obs: the number of 'observations' from the window via which the loss will be computed
    - stage: which stage in the optimization are we at? eg. in an iteration schedule [100,200,300], stage 3 would have 300 iterations.
    - policy: 'beginning' or 'random'; where do we initialize our window throughout the trajectory? 
"""
function make_window_spec(t_start, window_T, n_obs::Integer; stage::Integer, iteration::Integer, policy::AbstractString)::WindowSpec
    t_end = t_start + window_T # we've already checked to make sure this fits
    WindowSpec(stage, iteration, String(policy), Float64(t_start), Float64(t_end), Float64(window_T), n_obs,
               collect(LinRange(t_start + window_T / n_obs, t_end, n_obs))) # note that here we avoid comparing against the initial condition, since these are forced to match anyway
end

"""
Populate a TrainingWindow struct. This is similar to WindowSpec, but it contains both the hefty reference snapshot data to be compared against, and any projected data if needed for a ROM
Args:
- u_ref: the reference soln initial condition
- spec: a WindowSpec struct (see above, and types.jl)
- project: if we're doing a rom, this projects an initial condition down to the reduced basis. If not, it's just the reference initial condition.
"""
function materialize_window(u_ref, spec::WindowSpec, project=nothing)::TrainingWindow
    u0 = copy(u_ref(spec.t_start))
    TrainingWindow(spec, u0, isnothing(project) ? copy(u0) : project(u0), [copy(u_ref(ti)) for ti in spec.t_obs])
end

"""
Builds a vector of WindowSpecs for a given stage of the optimization process
Args: 
- tspan: 2-tuple of floats, (t0,tfinal), the span of time our reference has simulated
- stage: which stage in the optimization are we at? eg. in an iteration schedule [100,200,300], stage 3 would have 300 iterations.
- n_iter: the number of iterations in this stage
- window_T: the time length of windows for this stage
- n_obs: the number of observations via which we compare to true soln in this stage
- policy: 'beginning' or 'random'; where do we initialize our window throughout the trajectory? 
- rng: a randomizer that builds a (structured, based on seed) random trajectory of initial window times, if policy == 'random'
"""
function build_stage_window_specs(tspan::Tuple{Float64,Float64}, stage::Integer, n_iter::Integer, window_T, n_obs::Integer, policy::AbstractString, rng)::Vector{WindowSpec}
    t0, tfinal = tspan
    window_T <= tfinal - t0 || error("window_T entries cannot exceed the full trajectory length")
    latest_start = tfinal - window_T
    specs = WindowSpec[] # create an empty vector whose element type is WindowSpec
    for iteration in 1:n_iter
        if policy == "beginning"
            t_start = t0
        else
            t_start = t0 + rand(rng) * (latest_start - t0)
        end
        push!(specs, make_window_spec(t_start, window_T, n_obs; stage, iteration, policy))
    end
    specs
end

"""
Build the schedule of all windows for the whole optimization, across stages, returned as a vector of vectors of WindowSpecs
Args:
- tspan: 2-tuple of floats, (t0,tfinal), the span of time our reference has simulated
- N_iter_schedule: the iteration schedule, eg. [100,500,1000]
- window_T_schedule: the schedule of window time lengths, eg [0.05,10.0,32.0] 
- policy schedule: the schedule of window start policies, eg. ['beginning','beginning','random']
- window_seed: the seed for rng that randomizes window start times if the above is 'random'
"""
function build_window_schedule(tspan::Tuple{Float64,Float64}, N_iter_schedule, window_T_schedule, window_N_obs_schedule, policy_schedule, window_seed)::Vector{Vector{WindowSpec}}
    rng = MersenneTwister(window_seed)
    stage_specs = [build_stage_window_specs(tspan, stage, N_iter_schedule[stage], window_T_schedule[stage],
        window_N_obs_schedule[stage], policy_schedule[stage], rng) for stage in eachindex(N_iter_schedule)]
    stage_specs
end


"""
Run the optimization and return a TrainingOutput struct with the data. 
I will add comments to relevant lines below. 
Args:
- prepared: a PreparedTraining struct (see types.jl) with the relevant training information. This has most of what we need, but not the metadata that will build our windows; this is stored in
- training: a TrainingConfig struct
- log_name: used for printing throughout the optimization
"""
function run_variable_window_stages(prepared::PreparedTraining, training::TrainingConfig; log_name)::TrainingOutput
    # Gather the metadata for training
    eta_schedule, N_iter_schedule = training.etas, training.iterations
    window_T_schedule, window_N_obs_schedule = training.window_T, training.window_N_obs
    all(all(>(0), schedule) for schedule in (
        eta_schedule,
        N_iter_schedule,
        window_T_schedule,
        window_N_obs_schedule,
    )) || error("etas, iterations, window T's, and window N obs must be > 0")
    training.loss_space == "REDUCED" && prepared.mode != :rom && error("REDUCED loss-space is only available for ROM training")
    policy_schedule = training.window_start_policy
    stage_count, total_iterations = length(eta_schedule), sum(N_iter_schedule)
    base_prob, p0, u_ref = prepared.problem, prepared.initial_parameters, prepared.reference.solution
    full_T = base_prob.tspan[2] - base_prob.tspan[1]
    stage_specs = build_window_schedule(base_prob.tspan, N_iter_schedule, window_T_schedule,
        window_N_obs_schedule, policy_schedule, training.window_seed) # build the full optimization window schedule
    window_history = reduce(vcat, stage_specs) # this is a flattened version of the above for saving purposes only

    # here, we build a validation window to calculate full trajectory loss against
    validation_spec = make_window_spec(base_prob.tspan[1], full_T, prepared.reference.N_obs; stage=0, iteration=0, policy="beginning")
    validation_window = materialize_window(u_ref, validation_spec, prepared.project)

    # here, we setup the first step of our optimization
    theta0, re = Optimisers.destructure(p0.θ)
    # current window is a Ref so that when we slip it into solve_window_loss below, we can adjust it throughout the optimization based on which window we're training over 
    # and that will adjust the optimization function, without having to redefine optf
    current_window = Ref{TrainingWindow}(materialize_window(u_ref, stage_specs[1][1], prepared.project))
    # define the optimization function: 
    optf = Optimization.OptimizationFunction(
        (theta, _) -> solve_window_loss(current_window[], 
                                        base_prob, 
                                        prepared.rebuild_parameters(p0, re, theta),
                                        TRBDF2(autodiff=AutoFiniteDiff()), 
                                        GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP()),
                                        training.loss_normalization, 
                                        prepared.Δmeasure, 
                                        prepared.reconstruct,
                                        prepared.project,
                                        training.loss_space),
        Optimization.AutoZygote(), # outer autodiff backend. Mooncake is doing the VJP, and GaussAdjoint is doing the adjoint solve through the PDE, but this is doing the outer sensitivty of the loss wrt NN/polynom. params
    )

    alg, sensalg = TRBDF2(autodiff=AutoFiniteDiff()), GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP()) # forward and reverse solvers
    initial_loss = optf(copy(theta0), nothing) # can call optf whenever we want to calculate the loss based on a parametrization; however, this will work over the current window. 
    # If we want the full trajectory *validation* loss, we need to do: 
    initial_validation_loss = solve_window_loss(validation_window, base_prob, prepared.rebuild_parameters(p0, re, copy(theta0)),
                                                alg, sensalg, training.loss_normalization, prepared.Δmeasure, prepared.reconstruct,
                                                prepared.project, training.loss_space)
    
    # save initial params and validation loss

    initial_error = training.learned_function_error ? learned_function_l2_error(prepared, prepared.rebuild_parameters(p0, re, copy(theta0)).θ, training.learned_function_error_bounds) : nothing
    !isnothing(initial_error) && hpc_log_timed(log_name, "iteration = 0, learned_function_l2_error = $initial_error")
    parameter_history = TrainingSnapshot[TrainingSnapshot(0, 0, :parameter, copy(theta0), initial_loss, initial_error)]
    validation_history = TrainingSnapshot[TrainingSnapshot(0, 0, :validation, nothing, initial_validation_loss, nothing)]
    
    # store references (so we can mutate) to the current window optimiztaion data
    latest_theta, last_training_loss = Ref(copy(theta0)), Ref(initial_loss)
    last_iteration, last_callback_iteration, iteration_offset = Ref(0), Ref(-1), Ref(0)
    last_time, result = Ref{Float64}(time()), nothing

    alg_label = "TRBDF2(autodiff=AutoFiniteDiff())"
    sensalg_label = "GaussAdjoint(MooncakeVJP)"

    hpc_log_timed(log_name, "Optimization Params: ode_algorithm = $alg_label; sensitivity_algorithm = $sensalg_label; eta = $eta_schedule; beta = $(training.beta); N_iter = $N_iter_schedule; window_T = $window_T_schedule; window_N_obs = $window_N_obs_schedule; window_start_policy = $policy_schedule; loss_normalization = $(training.loss_normalization); loss_space = $(training.loss_space); learned_function_error = $(training.learned_function_error); learned_function_error_bounds = $(training.learned_function_error_bounds); total_iterations = $total_iterations")

    # a warmup compilation run
    if training.warmup
        hpc_log_timed(log_name, "Warming up")
        Optimization.solve(Optimization.OptimizationProblem(optf, copy(theta0), nothing), OptimizationOptimisers.Adam(eta_schedule[1], training.beta); maxiters=1)
        hpc_log_timed(log_name, "Warmup complete")
    end

    curr_stage_ref = Ref(1)
    current_stage_specs = Ref{Vector{WindowSpec}}(stage_specs[1])
    """
    An internal callback function that the OptimizationOptimisers [sic] object uses to move through an optimization procedure. 
    Args:
    - state: the current state of the parameters that we're optimizing, along with the iter
    - loss: the current loss
    """
    function callback_variable_window(state, loss)
        elapsed, last_time[] = time() - last_time[], time()
        global_iteration = iteration_offset[] + state.iter
        # avoid duplicate callbacks
        (curr_stage_ref[] > 1 && state.iter == 0 || global_iteration == last_callback_iteration[]) && return false
        # okay, so we're not duplicated
        last_callback_iteration[], last_iteration[] = global_iteration, global_iteration
        
        latest_theta[], last_training_loss[] = copy(state.u), loss
        # if we're at a place to save, and we haven't saved, save 
        if global_iteration > 0 && global_iteration % training.save_frequency == 0 && parameter_history[end].iteration != global_iteration
            function_error = training.learned_function_error ? learned_function_l2_error(prepared, prepared.rebuild_parameters(p0, re, state.u).θ, training.learned_function_error_bounds) : nothing
            !isnothing(function_error) && hpc_log_timed(log_name, "iteration = $global_iteration, learned_function_l2_error = $function_error")
            push!(parameter_history, TrainingSnapshot(global_iteration, curr_stage_ref[], :parameter, copy(state.u), loss, function_error))
        end
        # if we're at a place to print, print
        if global_iteration % training.print_frequency == 0
            window = current_window[]
            hpc_log_timed(log_name, "iteration = $global_iteration, loss = $loss, stage = $(curr_stage_ref[]), window_start = $(window.spec.t_start), window_end = $(window.spec.t_end), last iteration = $(round(elapsed; digits=2)) s")
        end
        # build the next window to optimize over 
        if state.iter > 0
            specs = current_stage_specs[]
            next_window = min(state.iter + 1, length(specs))
            current_window[] = materialize_window(u_ref, specs[next_window], prepared.project)
        end
        false
    end

    # now we begin the optimization
    for stage in 1:stage_count
        # use this ref to adjust callback fxn
        curr_stage_ref[] = stage
        current_stage_specs[] = stage_specs[stage]

    
        # view to current window; thus we've changed opft 
        current_window[] = materialize_window(u_ref, current_stage_specs[][1], prepared.project)
        stage_start = time()
        hpc_log_timed(log_name, "Stage $stage / $stage_count started: eta = $(eta_schedule[stage]); N_iter = $(N_iter_schedule[stage]); window_T = $(window_T_schedule[stage]); window_N_obs = $(window_N_obs_schedule[stage]); window_start_policy = $(policy_schedule[stage])")

        

        # note that we make our way through a whole stage with one 'solve' call, because the callback function is doing the work of updating the current window.
        result = Optimization.solve(Optimization.OptimizationProblem(optf, copy(latest_theta[]), nothing),
            OptimizationOptimisers.Adam(eta_schedule[stage], training.beta); maxiters=N_iter_schedule[stage], callback=callback_variable_window)
        iteration_offset[] += N_iter_schedule[stage]
        stage_validation_loss = solve_window_loss(validation_window, base_prob, prepared.rebuild_parameters(p0, re, latest_theta[]),
                                                  alg, sensalg, training.loss_normalization, prepared.Δmeasure, prepared.reconstruct,
                                                  prepared.project, training.loss_space)
        push!(validation_history, TrainingSnapshot(iteration_offset[], stage, :validation, nothing, stage_validation_loss, nothing))
        hpc_log_timed(log_name, "Stage $stage / $stage_count complete; validation_loss = $stage_validation_loss; elapsed = $(round(time() - stage_start; digits=2)) s")
    end

    final_training_loss = last_training_loss[]
    final_full_trajectory_loss = last(validation_history).loss
    final_iteration = last_iteration[]
    if final_iteration == 0
        final_iteration = total_iterations
    end

    final_error = parameter_history[end].iteration == final_iteration ? parameter_history[end].learned_function_error :
        (training.learned_function_error ? learned_function_l2_error(prepared, prepared.rebuild_parameters(p0, re, latest_theta[]).θ, training.learned_function_error_bounds) : nothing)
    parameter_history[end].iteration != final_iteration && !isnothing(final_error) && hpc_log_timed(log_name, "iteration = $final_iteration, learned_function_l2_error = $final_error")
    final_snapshot = TrainingSnapshot(final_iteration, stage_count, :parameter, copy(latest_theta[]), final_training_loss, final_error)
    if parameter_history[end].iteration == final_iteration
        parameter_history[end] = final_snapshot
    else
        push!(parameter_history, final_snapshot)
    end
    TrainingOutput(result, copy(latest_theta[]), training, parameter_history, final_training_loss,
                   final_full_trajectory_loss, window_history, validation_history)
end
