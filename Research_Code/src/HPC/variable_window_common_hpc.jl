if !isdefined(@__MODULE__, :build_window_schedule)

"""Build a lightweight deterministic physical-time window specification."""
function make_window_spec(t_start, window_T, n_obs::Integer; stage::Integer, batch::Integer, window::Integer, policy::AbstractString)
    n_obs > 0 || error("WINDOW_N_OBS entries must be positive")
    t_end = t_start + window_T
    return (;
        stage,
        batch,
        window,
        policy,
        t_start,
        t_end,
        window_T,
        n_obs,
        t_obs=collect(LinRange(t_start + window_T / n_obs, t_end, n_obs)),
    )
end

"""Materialize reference states for one precomputed window specification."""
function materialize_window(u_ref, spec)
    return merge(spec, (;
        u0=copy(u_ref(spec.t_start)),
        u_ref_obs=[u_ref(ti) for ti in spec.t_obs],
    ))
end

"""Materialize all full-order reference data for one optimizer iteration."""
materialize_batch(u_ref, specs) = [materialize_window(u_ref, spec) for spec in specs]

"""Precompute deterministic batches of window specifications for one optimization stage."""
function build_stage_window_specs(tspan, stage::Integer, n_iter::Integer, window_T, n_obs::Integer, policy::AbstractString, batch_size::Integer, rng)
    t0, tfinal = tspan
    n_iter > 0 || error("N_iter entries must be positive")
    window_T > 0 || error("window_T entries must be positive")
    window_T <= tfinal - t0 || error("window_T entries cannot exceed the full trajectory length")
    batch_size > 0 || error("batch_size entries must be positive")

    batches = Vector{Vector{Any}}(undef, n_iter)
    latest_start = tfinal - window_T
    for batch in 1:n_iter
        batch_specs = Vector{Any}(undef, batch_size)
        for window in 1:batch_size
            t_start = policy == "beginning" ? t0 : t0 + rand(rng) * (latest_start - t0)
            batch_specs[window] = make_window_spec(t_start, window_T, n_obs; stage, batch, window, policy)
        end
        batches[batch] = batch_specs
    end
    return batches
end

"""Build all staged window specifications and a flattened history."""
function build_window_schedule(tspan, N_iter_schedule, window_T_schedule, window_N_obs_schedule, policy_schedule, batch_size_schedule, window_seed)
    rng = MersenneTwister(window_seed)
    stage_specs = Vector{Any}(undef, length(N_iter_schedule))
    window_history = Any[]
    for stage in eachindex(N_iter_schedule)
        specs = build_stage_window_specs(
            tspan,
            stage,
            N_iter_schedule[stage],
            window_T_schedule[stage],
            window_N_obs_schedule[stage],
            policy_schedule[stage],
            batch_size_schedule[stage],
            rng,
        )
        stage_specs[stage] = specs
        append!(window_history, reduce(vcat, specs))
    end
    return stage_specs, window_history
end

"""
Run the model-independent staged variable-window optimization loop.

The parent FOM or ROM helper supplies model-specific batch materialization,
parameter reconstruction, and loss evaluation functions.
Window settings accept scalars or complete vectors with one entry per stage.
`batch_size` is the number of trajectory windows averaged per optimizer iteration.
"""
function run_variable_window_stages(
    u_ref,
    base_prob,
    p0;
    optimization_data,
    materialize_model_batch,
    rebuild_params,
    batch_loss,
    validation_N_obs,
    log_name,
    eta=5e-2,
    beta=(0.9, 0.99),
    N_iter=400,
    window_T=nothing,
    window_N_obs=nothing,
    window_start_policy="beginning",
    batch_size=1,
    loss_normalization="mean",
    window_seed=1,
    alg=TRBDF2(autodiff=AutoFiniteDiff()),
    ### ADJUSTED: Keep the shared variable-window default on Mooncake instead of Enzyme.
    sensalg=GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP()),
    warmup=true,
    save_frequency=nothing,
    print_frequency=10,
)
    eta_schedule = eta isa AbstractVector ? collect(eta) : [eta]
    N_iter_schedule = N_iter isa AbstractVector ? collect(N_iter) : [N_iter]
    stage_count = length(eta_schedule)
    total_iterations = sum(N_iter_schedule)
    full_T = base_prob.tspan[2] - base_prob.tspan[1]

    window_T_schedule = Float64.(window_T isa AbstractVector ? collect(window_T) : fill(isnothing(window_T) ? full_T : window_T, stage_count))
    window_N_obs_schedule = Int.(window_N_obs isa AbstractVector ? collect(window_N_obs) : fill(isnothing(window_N_obs) ? validation_N_obs : window_N_obs, stage_count))
    policy_schedule = window_start_policy isa AbstractVector ? collect(window_start_policy) : fill(window_start_policy, stage_count)
    batch_size_schedule = Int.(batch_size isa AbstractVector ? collect(batch_size) : fill(batch_size, stage_count))
    save_frequency = isnothing(save_frequency) ? max(1, cld(total_iterations, 10)) : save_frequency

    stage_specs, window_history = build_window_schedule(
        base_prob.tspan,
        N_iter_schedule,
        window_T_schedule,
        window_N_obs_schedule,
        policy_schedule,
        batch_size_schedule,
        window_seed,
    )
    validation_spec = make_window_spec(
        base_prob.tspan[1],
        full_T,
        validation_N_obs;
        stage=0,
        batch=0,
        window=1,
        policy="validation",
    )
    validation_batch = materialize_model_batch(u_ref, [validation_spec])

    theta0, re = Optimisers.destructure(p0.θ)
    current_batch = Ref{Any}(materialize_model_batch(u_ref, stage_specs[1][1]))
    optf = Optimization.OptimizationFunction(
        (theta, data) -> batch_loss(
            current_batch[],
            base_prob,
            rebuild_params(p0, re, theta),
            alg,
            sensalg,
            loss_normalization,
        ),
        Optimization.AutoZygote(),
    )

    initial_loss = optf(copy(theta0), optimization_data)
    initial_validation_loss = batch_loss(validation_batch, base_prob, rebuild_params(p0, re, copy(theta0)), alg, sensalg, loss_normalization)
    parameter_history = [(iteration=0, θ=copy(theta0), loss=initial_loss)]
    validation_history = [(iteration=0, stage=0, loss=initial_validation_loss)]
    latest_theta = Ref(copy(theta0))
    last_training_loss = Ref(initial_loss)
    last_iteration = Ref(0)
    last_callback_iteration = Ref(-1)
    iteration_offset = Ref(0)
    last_time = Ref{Float64}(time())
    result = nothing

    ### ADJUSTED: Print readable backend labels instead of raw SciML object internals.
    alg_label = occursin("AutoFiniteDiff", string(alg)) ? "TRBDF2(autodiff=AutoFiniteDiff())" : string(nameof(typeof(alg)))
    sensalg_label = occursin("MooncakeVJP", string(sensalg)) ? "GaussAdjoint(MooncakeVJP)" : string(nameof(typeof(sensalg)))
    hpc_log_timed(log_name, "Optimization Params: ode_algorithm = $alg_label; sensitivity_algorithm = $sensalg_label; eta = $eta_schedule; beta = $beta; N_iter = $N_iter_schedule; window_T = $window_T_schedule; window_N_obs = $window_N_obs_schedule; window_start_policy = $policy_schedule; batch_size = $batch_size_schedule; loss_normalization = $loss_normalization; total_iterations = $total_iterations")

    if warmup
        hpc_log_timed(log_name, "Warming up")
        warmup_problem = Optimization.OptimizationProblem(optf, copy(theta0), optimization_data)
        Optimization.solve(warmup_problem, OptimizationOptimisers.Adam(eta_schedule[1], beta); maxiters=1)
        hpc_log_timed(log_name, "Warmup complete")
    end

    for stage in 1:stage_count
        current_stage_specs = stage_specs[stage]
        current_batch[] = materialize_model_batch(u_ref, current_stage_specs[1])
        stage_start = time()
        hpc_log_timed(log_name, "Stage $stage / $stage_count started: eta = $(eta_schedule[stage]); N_iter = $(N_iter_schedule[stage]); window_T = $(window_T_schedule[stage]); window_N_obs = $(window_N_obs_schedule[stage]); window_start_policy = $(policy_schedule[stage]); batch_size = $(batch_size_schedule[stage])")

        function callback_variable_window(state, loss)
            now = time()
            elapsed = now - last_time[]
            last_time[] = now
            global_iteration = iteration_offset[] + state.iter

            if stage > 1 && state.iter == 0
                return false
            end
            if global_iteration == last_callback_iteration[]
                return false
            end

            last_callback_iteration[] = global_iteration
            last_iteration[] = global_iteration
            latest_theta[] = copy(state.u)
            last_training_loss[] = loss

            if global_iteration > 0 && global_iteration % save_frequency == 0 && parameter_history[end].iteration != global_iteration
                push!(parameter_history, (iteration=global_iteration, θ=copy(state.u), loss=loss))
            end
            if global_iteration % print_frequency == 0
                first_window = first(current_batch[])
                hpc_log_timed(log_name, "iteration = $global_iteration, loss = $loss, stage = $stage, window_start = $(first_window.t_start), window_end = $(first_window.t_end), last iteration = $(round(elapsed; digits=2)) s")
            end
            if state.iter > 0
                next_batch = min(state.iter + 1, length(current_stage_specs))
                current_batch[] = materialize_model_batch(u_ref, current_stage_specs[next_batch])
            end
            return false
        end

        current_problem = Optimization.OptimizationProblem(optf, copy(latest_theta[]), optimization_data)
        result = Optimization.solve(
            current_problem,
            OptimizationOptimisers.Adam(eta_schedule[stage], beta);
            maxiters=N_iter_schedule[stage],
            callback=callback_variable_window,
        )
        iteration_offset[] += N_iter_schedule[stage]
        stage_validation_loss = batch_loss(validation_batch, base_prob, rebuild_params(p0, re, latest_theta[]), alg, sensalg, loss_normalization)
        push!(validation_history, (iteration=iteration_offset[], stage=stage, loss=stage_validation_loss))
        hpc_log_timed(log_name, "Stage $stage / $stage_count complete; validation_loss = $stage_validation_loss; elapsed = $(round(time() - stage_start; digits=2)) s")
    end

    final_training_loss = last_training_loss[]
    final_full_trajectory_loss = last(validation_history).loss
    final_iteration = last_iteration[] == 0 ? total_iterations : last_iteration[]
    if parameter_history[end].iteration == final_iteration
        parameter_history[end] = (iteration=final_iteration, θ=copy(latest_theta[]), loss=final_training_loss)
    else
        push!(parameter_history, (iteration=final_iteration, θ=copy(latest_theta[]), loss=final_training_loss))
    end

    settings = (;
        ode_algorithm=string(alg),
        sensitivity_algorithm=string(sensalg),
        optimizer="Adam variable-window staged",
        η=length(eta_schedule) == 1 ? eta_schedule[1] : copy(eta_schedule),
        N_iter=length(N_iter_schedule) == 1 ? N_iter_schedule[1] : total_iterations,
        η_schedule=copy(eta_schedule),
        N_iter_schedule=copy(N_iter_schedule),
        β=beta,
        window_T=length(window_T_schedule) == 1 ? window_T_schedule[1] : copy(window_T_schedule),
        window_N_obs=length(window_N_obs_schedule) == 1 ? window_N_obs_schedule[1] : copy(window_N_obs_schedule),
        window_start_policy=length(policy_schedule) == 1 ? policy_schedule[1] : copy(policy_schedule),
        batch_size=length(batch_size_schedule) == 1 ? batch_size_schedule[1] : copy(batch_size_schedule),
        window_T_schedule=copy(window_T_schedule),
        window_N_obs_schedule=copy(window_N_obs_schedule),
        window_start_policy_schedule=copy(policy_schedule),
        batch_size_schedule=copy(batch_size_schedule),
        loss_normalization,
        window_seed,
        validation_N_obs,
        warmup,
        save_frequency,
        print_frequency,
        final_training_loss,
        final_full_trajectory_loss,
    )
    result = merge(
        NamedTuple{fieldnames(typeof(result))}(getfield(result, name) for name in fieldnames(typeof(result))),
        (; u=copy(latest_theta[]), objective=final_full_trajectory_loss),
    )
    return (; result, parameter_history, settings, final_loss=final_full_trajectory_loss, final_training_loss, final_full_trajectory_loss, window_history, validation_history)
end

end
