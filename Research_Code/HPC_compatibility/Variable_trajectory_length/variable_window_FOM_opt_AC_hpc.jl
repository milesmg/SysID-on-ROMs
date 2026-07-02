
include(joinpath(@__DIR__, "..", "..", "helper_functions", "HPC", "FOM_opt_AC_hpc.jl"))

"""
Broadcast a scalar schedule value or validate a per-stage schedule.
"""
function variable_window_schedule(values, stage_count::Integer, name::AbstractString)
    collected = values isa AbstractVector ? collect(values) : [values]
    if length(collected) == 1
        return fill(collected[1], stage_count)
    elseif length(collected) == stage_count
        return collected
    end
    error("$name must be scalar or have one entry per stage")
end

"""
Normalize a window start policy name.
"""
function normalize_window_start_policy(policy)
    normalized = lowercase(strip(String(policy)))
    normalized in ("beginning", "random") || error("WINDOW_START_POLICY must be beginning or random")
    return normalized
end

"""
Normalize the loss scaling mode.
"""
function normalize_loss_normalization(loss_normalization)
    normalized = lowercase(strip(String(loss_normalization)))
    normalized in ("mean", "sum") || error("LOSS_NORMALIZATION must be mean or sum")
    return normalized
end

"""
Return exact observation times inside `(t_start, t_end]`, excluding the known initial state.
"""
function window_observation_times(t_start, t_end, n_obs::Integer)
    n_obs > 0 || error("WINDOW_N_OBS entries must be positive")
    return collect(LinRange(t_start + (t_end - t_start) / n_obs, t_end, n_obs))
end

"""
Build a lightweight window specification with deterministic start/end times.
"""
function make_window_spec(t_start, window_T, n_obs::Integer; stage::Integer, batch::Integer, window::Integer, policy::AbstractString)
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
        t_obs=window_observation_times(t_start, t_end, n_obs),
    )
end

"""
Materialize reference states for one precomputed window specification.
"""
function materialize_window(u_ref, spec)
    return merge(spec, (;
        u0=copy(u_ref(spec.t_start)),
        u_ref_obs=[u_ref(ti) for ti in spec.t_obs],
    ))
end

"""Materialize all reference data needed for one optimizer iteration."""
materialize_batch(u_ref, specs) = [materialize_window(u_ref, spec) for spec in specs]

"""
Precompute deterministic window specifications for one optimization stage.
"""
function build_stage_window_specs(u_ref, tspan, stage::Integer, n_iter::Integer, window_T, n_obs::Integer, policy::AbstractString, windows_per_iter::Integer, rng)
    t0, tfinal = tspan
    n_iter > 0 || error("ITERS entries must be positive")
    window_T > 0 || error("WINDOW_T entries must be positive")
    window_T <= tfinal - t0 || error("WINDOW_T entries cannot exceed the full trajectory length")
    windows_per_iter > 0 || error("WINDOWS_PER_ITER entries must be positive")

    batches = Vector{Vector{Any}}(undef, n_iter)
    latest_start = tfinal - window_T
    for batch in 1:n_iter
        batch_specs = Vector{Any}(undef, windows_per_iter)
        for window in 1:windows_per_iter
            t_start = policy == "beginning" ? t0 : t0 + rand(rng) * (latest_start - t0)
            batch_specs[window] = make_window_spec(t_start, window_T, n_obs; stage, batch, window, policy)
        end
        batches[batch] = batch_specs
    end
    return batches
end

"""
Compute one window's spatial MSE loss against reference observations.
"""
function variable_window_loss(window, base_prob, p, alg, sensalg, loss_normalization::AbstractString)
    window_prob = remake(base_prob; u0=window.u0, tspan=(window.t_start, window.t_end), p=p)
    sol = solve(window_prob, alg; saveat=window.t_obs, sensealg=sensalg)
    total = zero(eltype(first(sol.u)))
    @inbounds for j in eachindex(sol.u, window.u_ref_obs)
        u_model = sol.u[j]
        u_ref = window.u_ref_obs[j]
        @simd for i in eachindex(u_model, u_ref)
            total += abs2(u_model[i] - u_ref[i])
        end
    end
    loss = 0.5 * p.Δx * total
    return loss_normalization == "mean" ? loss / length(window.u_ref_obs) : loss
end

"""
Compute the current optimizer-iteration loss by averaging or summing its scheduled windows.
"""
function variable_window_batch_loss(batch, base_prob, p, alg, sensalg, loss_normalization::AbstractString)
    total = zero(eltype(first(batch).u0))
    for window in batch
        total += variable_window_loss(window, base_prob, p, alg, sensalg, loss_normalization)
    end
    return loss_normalization == "mean" ? total / length(batch) : total
end

"""
Compute the full-trajectory validation loss for a flat NN parameter vector.
"""
function full_trajectory_validation_loss(validation_batch, base_prob, p0, theta, re, alg, sensalg, loss_normalization::AbstractString)
    params = ComponentVector(ε2=p0.ε2, Δx=p0.Δx, θ=re(theta))
    return variable_window_batch_loss(validation_batch, base_prob, params, alg, sensalg, loss_normalization)
end

"""
Run staged Adam optimization on deterministic variable-length trajectory windows.

- `eta` and `N_iter` define the existing staged optimizer schedule.
- `window_T`, `window_N_obs`, `window_start_policy`, and `windows_per_iter`
  are scalar or per-stage schedules.
- `loss_normalization="mean"` divides by observations and windows; `"sum"`
  preserves the old summed-loss scale for a single full-trajectory window.
"""
function run_variable_window_optimization(
    u_ref,
    base_prob,
    p0;
    run_params,
    eta=5e-2,
    beta=(0.9, 0.99),
    N_iter=400,
    window_T=nothing,
    window_N_obs=nothing,
    window_start_policy=["beginning"],
    windows_per_iter=1,
    loss_normalization="mean",
    window_seed=1,
    validation_N_obs=nothing,
    alg=TRBDF2(),
    sensalg=GaussAdjoint(autojacvec=ReverseDiffVJP(true)),
    warmup=true,
    save_frequency=nothing,
    print_frequency=10,
)
    eta_schedule = eta isa AbstractVector ? collect(eta) : [eta]
    N_iter_schedule = N_iter isa AbstractVector ? collect(N_iter) : [N_iter]
    length(eta_schedule) == length(N_iter_schedule) || error("eta and N_iter must have the same length")
    stage_count = length(eta_schedule)
    total_iterations = sum(N_iter_schedule)
    full_T = base_prob.tspan[2] - base_prob.tspan[1]
    validation_N_obs = isnothing(validation_N_obs) ? run_params.N_obs : validation_N_obs
    window_T = isnothing(window_T) ? [full_T] : window_T
    window_N_obs = isnothing(window_N_obs) ? [validation_N_obs] : window_N_obs

    window_T_schedule = Float64.(variable_window_schedule(window_T, stage_count, "WINDOW_T"))
    window_N_obs_schedule = Int.(variable_window_schedule(window_N_obs, stage_count, "WINDOW_N_OBS"))
    window_start_policy_schedule = normalize_window_start_policy.(variable_window_schedule(window_start_policy, stage_count, "WINDOW_START_POLICY"))
    windows_per_iter_schedule = Int.(variable_window_schedule(windows_per_iter, stage_count, "WINDOWS_PER_ITER"))
    loss_normalization = normalize_loss_normalization(loss_normalization)
    save_frequency = isnothing(save_frequency) ? max(1, cld(total_iterations, 10)) : save_frequency

    rng = MersenneTwister(window_seed)
    stage_window_specs = Vector{Any}(undef, stage_count)
    window_history = Any[]
    for stage in 1:stage_count
        specs = build_stage_window_specs(
            u_ref,
            base_prob.tspan,
            stage,
            N_iter_schedule[stage],
            window_T_schedule[stage],
            window_N_obs_schedule[stage],
            window_start_policy_schedule[stage],
            windows_per_iter_schedule[stage],
            rng,
        )
        stage_window_specs[stage] = specs
        append!(window_history, reduce(vcat, specs))
    end

    validation_spec = make_window_spec(
        base_prob.tspan[1],
        full_T,
        validation_N_obs;
        stage=0,
        batch=0,
        window=1,
        policy="validation",
    )
    validation_batch = materialize_batch(u_ref, [validation_spec])

    theta0, re = Optimisers.destructure(p0.θ)
    current_batch = Ref{Any}(materialize_batch(u_ref, stage_window_specs[1][1]))
    optimization_data = (; run_params)
    adtype = Optimization.AutoZygote()
    optf = Optimization.OptimizationFunction(
        (theta, data) -> begin
            params = ComponentVector(ε2=p0.ε2, Δx=p0.Δx, θ=re(theta))
            variable_window_batch_loss(current_batch[], base_prob, params, alg, sensalg, loss_normalization)
        end,
        adtype,
    )

    initial_loss = optf(copy(theta0), optimization_data)
    initial_validation_loss = full_trajectory_validation_loss(validation_batch, base_prob, p0, copy(theta0), re, alg, sensalg, loss_normalization)
    parameter_history = [(iteration=0, θ=copy(theta0), loss=initial_loss)]
    validation_history = [(iteration=0, stage=0, loss=initial_validation_loss)]
    latest_theta = Ref(copy(theta0))
    last_training_loss = Ref(initial_loss)
    last_iteration = Ref(0)
    last_callback_iteration = Ref(-1)
    iteration_offset = Ref(0)
    last_time = Ref{Float64}(time())
    res = nothing

    hpc_log_timed("run_variable_window_optimization", "Optimization Params: eta = $eta_schedule; beta = $beta; N_iter = $N_iter_schedule; WINDOW_T = $window_T_schedule; WINDOW_N_OBS = $window_N_obs_schedule; WINDOW_START_POLICY = $window_start_policy_schedule; WINDOWS_PER_ITER = $windows_per_iter_schedule; loss_normalization = $loss_normalization; total_iterations = $total_iterations; save_frequency = $save_frequency; print_frequency = $print_frequency; warmup = $warmup")

    if warmup
        hpc_log_timed("run_variable_window_optimization", "Warming up")
        warmup_problem = Optimization.OptimizationProblem(optf, copy(theta0), optimization_data)
        Optimization.solve(
            warmup_problem,
            OptimizationOptimisers.Adam(eta_schedule[1], beta);
            maxiters=1,
        )
        hpc_log_timed("run_variable_window_optimization", "Warmup complete")
    end

    hpc_log_timed("run_variable_window_optimization", "Beginning variable-window optimization")

    for stage in 1:stage_count
        stage_specs = stage_window_specs[stage]
        current_batch[] = materialize_batch(u_ref, stage_specs[1])
        stage_start = time()
        hpc_log_timed("run_variable_window_optimization", "Stage $stage / $stage_count started: eta = $(eta_schedule[stage]); N_iter = $(N_iter_schedule[stage]); WINDOW_T = $(window_T_schedule[stage]); WINDOW_N_OBS = $(window_N_obs_schedule[stage]); WINDOW_START_POLICY = $(window_start_policy_schedule[stage]); WINDOWS_PER_ITER = $(windows_per_iter_schedule[stage])")

        function cb_variable_window(state, loss)
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
                hpc_log_timed("run_variable_window_optimization", "iteration = $(global_iteration), loss = $loss, stage = $stage, window_start = $(first_window.t_start), window_end = $(first_window.t_end), last iteration = $(round(elapsed; digits=2)) s")
            end

            if state.iter > 0
                next_batch = min(state.iter + 1, length(stage_specs))
                current_batch[] = materialize_batch(u_ref, stage_specs[next_batch])
            end

            return false
        end

        current_problem = Optimization.OptimizationProblem(optf, copy(latest_theta[]), optimization_data)
        res = Optimization.solve(
            current_problem,
            OptimizationOptimisers.Adam(eta_schedule[stage], beta);
            maxiters=N_iter_schedule[stage],
            callback=cb_variable_window,
        )

        iteration_offset[] += N_iter_schedule[stage]
        stage_validation_loss = full_trajectory_validation_loss(validation_batch, base_prob, p0, latest_theta[], re, alg, sensalg, loss_normalization)
        push!(validation_history, (iteration=iteration_offset[], stage=stage, loss=stage_validation_loss))
        hpc_log_timed("run_variable_window_optimization", "Stage $stage / $stage_count complete; validation_loss = $stage_validation_loss; elapsed = $(round(time() - stage_start; digits=2)) s")
    end

    final_training_loss = last_training_loss[]
    final_full_trajectory_loss = last(validation_history).loss
    final_iteration = last_iteration[] == 0 ? total_iterations : last_iteration[]
    if parameter_history[end].iteration == final_iteration
        parameter_history[end] = (iteration=final_iteration, θ=copy(latest_theta[]), loss=final_training_loss)
    else
        push!(parameter_history, (iteration=final_iteration, θ=copy(latest_theta[]), loss=final_training_loss))
    end

    run_params = merge(run_params, (;
        optimizer="Adam variable-window staged",
        η=length(eta_schedule) == 1 ? eta_schedule[1] : copy(eta_schedule),
        β=beta,
        N_iter=length(N_iter_schedule) == 1 ? N_iter_schedule[1] : total_iterations,
        η_schedule=copy(eta_schedule),
        N_iter_schedule=copy(N_iter_schedule),
        window_T=length(window_T_schedule) == 1 ? window_T_schedule[1] : copy(window_T_schedule),
        window_N_obs=length(window_N_obs_schedule) == 1 ? window_N_obs_schedule[1] : copy(window_N_obs_schedule),
        window_start_policy=length(window_start_policy_schedule) == 1 ? window_start_policy_schedule[1] : copy(window_start_policy_schedule),
        windows_per_iter=length(windows_per_iter_schedule) == 1 ? windows_per_iter_schedule[1] : copy(windows_per_iter_schedule),
        window_T_schedule=copy(window_T_schedule),
        window_N_obs_schedule=copy(window_N_obs_schedule),
        window_start_policy_schedule=copy(window_start_policy_schedule),
        windows_per_iter_schedule=copy(windows_per_iter_schedule),
        loss_normalization,
        window_seed,
        validation_N_obs,
        warmup,
        save_frequency,
        print_frequency,
        final_training_loss,
        final_full_trajectory_loss,
    ))

    result = merge(
        NamedTuple{fieldnames(typeof(res))}(getfield(res, name) for name in fieldnames(typeof(res))),
        (; u=copy(latest_theta[]), objective=final_full_trajectory_loss),
    )
    return (; result, parameter_history, run_params, final_loss=final_full_trajectory_loss, final_training_loss, final_full_trajectory_loss, window_history, validation_history)
end

"""
Save variable-window FOM output under `Optimization/Data/<run_name>`.
"""
function save_variable_window_optimization_data(output, run_name::AbstractString)
    data_root = normpath(joinpath(@__DIR__, "..", "..", "Optimization", "Data"))
    run_directory = assert_run_name_available(run_name; data_root)
    mkpath(data_root)
    mkdir(run_directory)

    serialize(joinpath(run_directory, "parameter_history.jls"), output.parameter_history)
    serialize(joinpath(run_directory, "run_params.jls"), output.run_params)
    serialize(joinpath(run_directory, "window_history.jls"), output.window_history)
    serialize(joinpath(run_directory, "validation_history.jls"), output.validation_history)
    serialize(joinpath(run_directory, "evaluation_history.jls"), output.validation_history)

    saved_metadata = merge(
        output.run_params,
        (;
            saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            julia_version=VERSION,
            initial_loss=first(output.parameter_history).loss,
            final_loss=output.final_loss,
            final_training_loss=output.final_training_loss,
            final_full_trajectory_loss=output.final_full_trajectory_loss,
            parameter_snapshots=length(output.parameter_history),
            window_history_entries=length(output.window_history),
            validation_snapshots=length(output.validation_history),
        ),
    )

    open(joinpath(run_directory, "metadata.txt"), "w") do io
        for (name, value) in pairs(saved_metadata)
            print(io, name, " = ")
            show(io, value)
            println(io)
        end
    end

    return run_directory
end
