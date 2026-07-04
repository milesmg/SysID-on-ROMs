include(joinpath(@__DIR__, "..", "..", "HPC_compatibility", "hpc_logging.jl"))
include(joinpath(@__DIR__, "..", "run_name_guard.jl"))

hpc_log_package("LinearAlgebra", "Loading")
using LinearAlgebra
hpc_log_package("LinearAlgebra", "Loaded")
hpc_log_package("SparseArrays", "Loading")
using SparseArrays
hpc_log_package("SparseArrays", "Loaded")
hpc_log_package("Random", "Loading")
using Random
hpc_log_package("Random", "Loaded")
hpc_log_package("ComponentArrays", "Loading")
using ComponentArrays
hpc_log_package("ComponentArrays", "Loaded")
hpc_log_package("OrdinaryDiffEq", "Loading")
using OrdinaryDiffEq
hpc_log_package("OrdinaryDiffEq", "Loaded")
hpc_log_package("OrdinaryDiffEqSDIRK", "Loading")
using OrdinaryDiffEqSDIRK
hpc_log_package("OrdinaryDiffEqSDIRK", "Loaded")
hpc_log_package("SciMLSensitivity", "Loading")
using SciMLSensitivity
hpc_log_package("SciMLSensitivity", "Loaded")
hpc_log_package("Optimization", "Loading")
using Optimization
hpc_log_package("Optimization", "Loaded")
hpc_log_package("OptimizationOptimisers", "Loading")
using OptimizationOptimisers
hpc_log_package("OptimizationOptimisers", "Loaded")
hpc_log_package("Lux", "Loading")
using Lux
hpc_log_package("Lux", "Loaded")
hpc_log_package("Functors", "Loading")
using Functors
hpc_log_package("Functors", "Loaded")
hpc_log_package("Dates", "Loading")
using Dates
hpc_log_package("Dates", "Loaded")
hpc_log_package("Serialization", "Loading")
using Serialization
hpc_log_package("Serialization", "Loaded")

hpc_log("package-load", "Including integration_AC_hpc.jl")
include("integration_AC_hpc.jl")
hpc_log("package-load", "Included integration_AC_hpc.jl")
include("variable_window_common_hpc.jl")


### Build ROM ###


"""
Build a POD-Galerkin/DEIM ROM from state and nonlinear snapshots.

- Args: `(A, u, fu, r, m)`
- Returns: reduced operators, selected modes and indices, and singular values.
"""
function build_rom(A, u, fu, r, m)
    state_svd = svd(u; full=false)
    nonlinear_svd = svd(fu; full=false)
    U = state_svd.U[:, 1:r]
    V = nonlinear_svd.U[:, 1:m]

    p = vcat(argmax(abs.(V[:, 1])), zeros(Int, m - 1))
    for k in 1:m-1
        residual = V[:, 1:k] * (V[p[1:k], 1:k] \ V[p[1:k], k + 1]) - V[:, k + 1]
        p[k + 1] = argmax(abs.(residual))
    end

    return (;
        U,
        V,
        Ã=U' * (A * U),
        p,
        Up=U[p, :],
        B=(V[p, :]' \ (V' * U))',
        state_singular_values=state_svd.S,
        nonlinear_singular_values=nonlinear_svd.S,
    )
end


### Integrate ROM ###
"""
Evaluate the Allen-Cahn POD/DEIM neural ROM right-hand side with batched NN calls.
"""
function rhs_ac_NN_ROM!(dũ, ũ, p, t, nn, state)
    (; Ã, θ, Up, B) = p
    z = Up * ũ
    fz = Fnn_batch(z, nn, θ, state)
    dũ .= Ã * ũ + B * fz
    return nothing
end


"""
Construct the reduced neural `ODEProblem`. Its function closure stores all
ROM and reference data required by the remaining optimization workflow.
"""
function neural_ROM_problem(ũ₀, tspan, p₀, data)
    f = ODEFunction((dũ, ũ, p, t) ->
        rhs_ac_NN_ROM!(dũ, ũ, p, t, data.nn, data.state))
    return ODEProblem(f, ũ₀, tspan, p₀)
end


"""
Solve a neural ROM and reconstruct its trajectory in the full state space.
"""
function model_ROM(prob, p, alg, sensalg)
    data = prob.f.f.data
    sol = solve(remake(prob; p), alg; saveat=data.t_obs, sensealg=sensalg)
    return data.rom.U * hcat(sol.u...)
end


"""
Compute the spatially weighted squared error between the reconstructed ROM
trajectory and full-order observations stored in `prob`.
"""
function loss_ROM(prob, p, alg, sensalg)
    data = prob.f.f.data
    sol = solve(remake(prob; p), alg; saveat=data.t_obs, sensealg=sensalg)
    total = zero(eltype(first(sol.u)))
    @inbounds for j in eachindex(sol.u)
        u_model = data.rom.U * sol.u[j]
        for i in axes(data.u_ref_obs, 1)
            total += abs2(u_model[i] - data.u_ref_obs[i, j])
        end
    end
    return 0.5 * data.Δx * total
end

"""Materialize full-state references and the projected ROM initial state."""
function materialize_ROM_batch(u_ref, rom, specs)
    return [merge(window, (; u0_rom=rom.U' * window.u0)) for window in materialize_batch(u_ref, specs)]
end

"""Compute one window's reconstructed spatially weighted ROM loss."""
function variable_window_ROM_loss(window, prob, p, alg, sensalg, normalization)
    data = prob.f.f.data
    window_prob = remake(prob; u0=window.u0_rom, tspan=(window.t_start, window.t_end), p=p)
    sol = solve(window_prob, alg; saveat=window.t_obs, sensealg=sensalg)
    total = zero(eltype(first(sol.u)))
    @inbounds for j in eachindex(sol.u, window.u_ref_obs)
        u_model = data.rom.U * sol.u[j]
        u_ref = window.u_ref_obs[j]
        @simd for i in eachindex(u_model, u_ref)
            total += abs2(u_model[i] - u_ref[i])
        end
    end
    loss = 0.5 * data.Δx * total
    return normalization == "mean" ? loss / length(window.u_ref_obs) : loss
end

"""Average or sum the ROM windows scheduled for one optimizer iteration."""
function variable_window_ROM_batch_loss(batch, prob, p, alg, sensalg, normalization)
    total = zero(eltype(first(batch).u0_rom))
    for window in batch
        total += variable_window_ROM_loss(window, prob, p, alg, sensalg, normalization)
    end
    return normalization == "mean" ? total / length(batch) : total
end


### Optimize ROM ###


"""
Prepare an Allen–Cahn POD/DEIM neural ROM optimization.

- Args: `(A, u_ref, r, m)`
- Keywords:
    - `nonlinear_snapshots=nothing`: use supplied snapshots, or construct
      `-k(u^3-u)` from `u_ref`
    - `Δx=u_ref.prob.p.Δx`: spatial grid spacing
    - `N_obs=10`: number of observation times, matching the FOM default
    - `t_obs=...`: optimization observation times, matching the FOM default
    - `h=8`: neural-network hidden width
    - `seed=1`: neural-network initialization seed
- Returns: one reduced neural `ODEProblem` containing its runtime context.
"""
function prepare_ROM_optimization(
    A,
    u_ref,
    r,
    m;
    nonlinear_snapshots=nothing,
    Δx=u_ref.prob.p.Δx,
    N_obs=10,
    t_obs=collect(LinRange(
        u_ref.prob.tspan[1] + (u_ref.prob.tspan[2] - u_ref.prob.tspan[1]) / (N_obs - 1),
        u_ref.prob.tspan[2],
        N_obs - 1,
    )),
    h=8,
    seed=1,
)
    u_snapshots = hcat(u_ref.u...)
    k = u_ref.prob.p.k
    use_default_nonlinearity = isnothing(nonlinear_snapshots)
    fu_snapshots = use_default_nonlinearity ?
        .-k .* (u_snapshots .^ 3 .- u_snapshots) :
        (nonlinear_snapshots isa AbstractMatrix ? nonlinear_snapshots : hcat(nonlinear_snapshots...))
    rom = build_rom(A, u_snapshots, fu_snapshots, r, m)

    rng = MersenneTwister(seed)
    nn = Chain(Dense(1 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
    ps₀, state = Lux.setup(rng, nn)
    ps₀ = fmap(x -> Float64.(x), ps₀)

    data = (;
        rom,
        nn,
        state,
        u_ref_obs=hcat((u_ref(ti) for ti in t_obs)...),
        t_obs=copy(t_obs),
        Δx,
        N=size(u_snapshots, 1),
        ε2=u_ref.prob.p.ε2,
        k,
        full_u₀=copy(u_ref.prob.u0),
        reference_saved_times=copy(u_ref.t),
        reference_algorithm=string(nameof(typeof(u_ref.alg))),
        h,
        seed,
        requested_N_obs=N_obs,
        use_default_nonlinearity,
    )

    p₀ = ComponentVector(Ã=rom.Ã, Up=rom.Up, B=rom.B, θ=ps₀)
    ũ₀ = rom.U' * u_ref.prob.u0
    return neural_ROM_problem(ũ₀, u_ref.prob.tspan, p₀, data)
end


"""
Build an `OptimizationProblem` from a prepared neural ROM `ODEProblem`.
- arg: (
    prob;
    alg=TRBDF2(autodiff=AutoFiniteDiff()),
    sensalg=GaussAdjoint(autojacvec=EnzymeVJP(...)),
)
- Keywords:
    - `alg=TRBDF2(autodiff=AutoFiniteDiff())`: forward ROM solver
    - `sensalg=GaussAdjoint(autojacvec=EnzymeVJP(...))`: sensitivity method
- returns: optprob
    """
function set_up_ROM_optimization(
    prob;
    alg=TRBDF2(autodiff=AutoFiniteDiff()),
    sensalg=GaussAdjoint(autojacvec=EnzymeVJP(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))),
)
    p₀ = prob.p
    θ₀, re = Optimisers.destructure(p₀.θ)
    optimization_data = (;
        rom_prob=prob,
        ode_algorithm=string(nameof(typeof(alg))),
        sensitivity_algorithm=string(sensalg),
    )

    optf = Optimization.OptimizationFunction(
        (θ, data) -> begin
            params = ComponentVector(Ã=p₀.Ã, Up=p₀.Up, B=p₀.B, θ=re(θ))
            loss_ROM(prob, params, alg, sensalg)
        end,
        Optimization.AutoZygote(),
    )

    return Optimization.OptimizationProblem(optf, copy(θ₀), optimization_data)
end


"""
Run an Adam ROM optimization with flushed phase timing logs for HPC jobs.

- Args: `(optprob; η=5e-2, β=(0.9, 0.99), N_iter=400,
  warmup=true, save_frequency=nothing, print_frequency=50)`
    - `η` and `N_iter` can also be same-length vectors for staged learning rates
- Returns: result, parameter history, final loss, the originating ROM problem, and optimizer settings, with `result.u` set from the latest callback state.
"""
function run_ROM_optimization(
    optprob;
    η=5e-2,
    β=(0.9, 0.99),
    N_iter=400,
    warmup=true,
    save_frequency=nothing,
    print_frequency=50,
)

    η_schedule = η isa AbstractVector ? collect(η) : [η]
    N_iter_schedule = N_iter isa AbstractVector ? collect(N_iter) : [N_iter]
    length(η_schedule) == length(N_iter_schedule) || error("η and N_iter must have the same length")
    total_iterations = sum(N_iter_schedule)
    last_time = Ref{Float64}(time())
    save_frequency = isnothing(save_frequency) ? max(1, cld(total_iterations, 10)) : save_frequency

    hpc_log_timed("run_ROM_optimization", "Optimization Params: η = $η_schedule; β = $β; N_iter = $N_iter_schedule; total_iterations = $total_iterations; save_frequency = $save_frequency; print_frequency = $print_frequency; warmup = $warmup")

    initial_loss_start = time()
    hpc_log_timed("run_ROM_optimization", "Computing initial loss")
    initial_loss = optprob.f(optprob.u0, optprob.p)
    hpc_log_timed("run_ROM_optimization", "Initial loss = $initial_loss; elapsed = $(round(time() - initial_loss_start; digits=2)) s")

    parameter_history = [(iteration=0, θ=copy(optprob.u0), loss=initial_loss)]
    last_iteration = Ref(0)
    iteration_offset = Ref(0)
    stage_index = Ref(1)
    latest_θ = Ref(optprob.u0)

    function callback(state, loss)
        now = time()
        elapsed = now - last_time[]
        last_time[] = now
        global_iteration = iteration_offset[] + state.iter

        if stage_index[] > 1 && state.iter == 0
            return false
        end

        last_iteration[] = global_iteration
        latest_θ[] = state.u

        if global_iteration > 0 && global_iteration % save_frequency == 0 && parameter_history[end].iteration != global_iteration
            push!(parameter_history, (iteration=global_iteration, θ=copy(state.u), loss))
        end

        if global_iteration % print_frequency == 0
            hpc_log_timed("run_ROM_optimization", "iteration = $(global_iteration), loss = $loss, last iteration = $(round(elapsed; digits=2)) s")
        end

        return false
    end

    if warmup
        warmup_start = time()
        hpc_log_timed("run_ROM_optimization", "Warming up")
        Optimization.solve(
            optprob,
            OptimizationOptimisers.Adam(η_schedule[1], β);
            maxiters=1,
        )
        hpc_log_timed("run_ROM_optimization", "Warmup complete; elapsed = $(round(time() - warmup_start; digits=2)) s")
    end

    last_time[] = time()

    hpc_log_timed("run_ROM_optimization", "Beginning optimization")
    current_optprob = optprob
    result = nothing
    for stage in eachindex(η_schedule)
        stage_index[] = stage
        stage_start = time()
        hpc_log_timed("run_ROM_optimization", "Stage $stage / $(length(η_schedule)) started: η = $(η_schedule[stage]); N_iter = $(N_iter_schedule[stage])")
        result = Optimization.solve(
            current_optprob,
            OptimizationOptimisers.Adam(η_schedule[stage], β);
            maxiters=N_iter_schedule[stage],
            callback,
        )
        hpc_log_timed("run_ROM_optimization", "Stage $stage / $(length(η_schedule)) complete; elapsed = $(round(time() - stage_start; digits=2)) s")
        iteration_offset[] += N_iter_schedule[stage]
        if stage < length(η_schedule)
            hpc_log_timed("run_ROM_optimization", "Rebuilding OptimizationProblem for next stage")
            current_optprob = Optimization.OptimizationProblem(current_optprob.f, copy(latest_θ[]), current_optprob.p)
        end
    end
    hpc_log_timed("run_ROM_optimization", "Completed optimization")

    final_loss_start = time()
    hpc_log_timed("run_ROM_optimization", "Computing final loss")
    final_loss = current_optprob.f(latest_θ[], current_optprob.p)
    hpc_log_timed("run_ROM_optimization", "Final loss = $final_loss; elapsed = $(round(time() - final_loss_start; digits=2)) s")

    final_iteration = last_iteration[] == 0 ? total_iterations : last_iteration[]
    if parameter_history[end].iteration == final_iteration
        parameter_history[end] = (iteration=final_iteration, θ=copy(latest_θ[]), loss=final_loss)
    else
        push!(parameter_history, (iteration=final_iteration, θ=copy(latest_θ[]), loss=final_loss))
    end

    run_settings = (;
        ode_algorithm=optprob.p.ode_algorithm,
        sensitivity_algorithm=optprob.p.sensitivity_algorithm,
        optimizer=length(η_schedule) == 1 ? "Adam" : "Adam staged",
        η=length(η_schedule) == 1 ? η_schedule[1] : copy(η_schedule),
        β,
        N_iter=length(N_iter_schedule) == 1 ? N_iter_schedule[1] : total_iterations,
        η_schedule=copy(η_schedule),
        N_iter_schedule=copy(N_iter_schedule),
        warmup,
        save_frequency,
        print_frequency,
    )

    result = merge(NamedTuple{fieldnames(typeof(result))}(getfield(result, name) for name in fieldnames(typeof(result))), (; u=copy(latest_θ[]), objective=final_loss))

    return (;
        result,
        parameter_history,
        final_loss,
        rom_prob=optprob.p.rom_prob,
        run_settings,
    )
end

"""
Run staged Adam ROM training on deterministic variable-length windows.

Window settings are scalar or same-length schedules with `eta`/`N_iter`.
Defaults use the whole trajectory, one window per iteration, and mean loss.
"""
function run_variable_window_ROM_optimization(
    u_ref,
    prob;
    eta=5e-2,
    beta=(0.9, 0.99),
    N_iter=400,
    window_T=nothing,
    window_N_obs=nothing,
    window_start_policy="beginning",
    windows_per_iter=1,
    loss_normalization="mean",
    window_seed=1,
    validation_N_obs=prob.f.f.data.requested_N_obs,
    alg=TRBDF2(autodiff=AutoFiniteDiff()),
    sensalg=GaussAdjoint(autojacvec=EnzymeVJP(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))),
    warmup=true,
    save_frequency=nothing,
    print_frequency=10,
)
    rom = prob.f.f.data.rom
    core = run_variable_window_stages(
        u_ref,
        prob,
        prob.p;
        optimization_data=(; rom_prob=prob),
        materialize_model_batch=(reference, specs) -> materialize_ROM_batch(reference, rom, specs),
        rebuild_params=(p, re, theta) -> ComponentVector(Ã=p.Ã, Up=p.Up, B=p.B, θ=re(theta)),
        batch_loss=variable_window_ROM_batch_loss,
        validation_N_obs,
        log_name="run_variable_window_ROM_optimization",
        eta,
        beta,
        N_iter,
        window_T,
        window_N_obs,
        window_start_policy,
        windows_per_iter,
        loss_normalization,
        window_seed,
        alg,
        sensalg,
        warmup,
        save_frequency,
        print_frequency,
    )
    return (;
        core.result,
        core.parameter_history,
        core.final_loss,
        core.final_training_loss,
        core.final_full_trajectory_loss,
        rom_prob=prob,
        run_settings=core.settings,
        core.window_history,
        core.validation_history,
    )
end


### Save ROM optimization ###


"""
Save the minimal raw data needed to reconstruct a ROM optimization under
`Optimization/Data/<run_name>`.

Derived capture ratios, reduced operators, grids, and projection errors are
not saved because they can be reconstructed from the saved modes, singular
values, indices, and scalar problem parameters.
"""
function save_ROM_optimization_data(output, run_name::AbstractString)
    data_root = normpath(joinpath(@__DIR__, "..", "..", "Optimization", "Data"))
    run_directory = assert_run_name_available(run_name; data_root)
    mkpath(data_root)
    mkdir(run_directory)

    prob = output.rom_prob
    data = prob.f.f.data
    rom = data.rom
    rom_data = (;
        N=data.N,
        ε2=data.ε2,
        k=data.k,
        Δx=data.Δx,
        tspan=prob.tspan,
        t_obs=data.t_obs,
        u₀=data.full_u₀,
        reference_saved_times=data.reference_saved_times,
        r=size(rom.U, 2),
        m=size(rom.V, 2),
        spatial_modes=rom.U,
        deim_modes=rom.V,
        deim_indices=rom.p,
        state_singular_values=rom.state_singular_values,
        nonlinear_singular_values=rom.nonlinear_singular_values,
        h=data.h,
        activation="tanh",
        seed=data.seed,
        use_default_nonlinearity=data.use_default_nonlinearity,
    )

    serialize(joinpath(run_directory, "parameter_history.jls"), output.parameter_history)
    serialize(joinpath(run_directory, "rom_data.jls"), rom_data)

    metadata = (;
        saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        julia_version=VERSION,
        N=rom_data.N,
        ε2=rom_data.ε2,
        k=rom_data.k,
        Δx=rom_data.Δx,
        tspan=rom_data.tspan,
        N_obs=length(rom_data.t_obs),
        reference_steps=length(rom_data.reference_saved_times) - 1,
        r=rom_data.r,
        m=rom_data.m,
        h=rom_data.h,
        activation=rom_data.activation,
        seed=rom_data.seed,
        reference_algorithm=data.reference_algorithm,
        output.run_settings...,
        initial_loss=first(output.parameter_history).loss,
        final_loss=output.final_loss,
        parameter_snapshots=length(output.parameter_history),
    )

    open(joinpath(run_directory, "metadata.txt"), "w") do io
        for (name, value) in pairs(metadata)
            print(io, name, " = ")
            show(io, value)
            println(io)
        end
    end

    return run_directory
end

"""Save variable-window ROM output under the standard run directory."""
function save_variable_window_ROM_optimization_data(output, run_name::AbstractString)
    run_directory = save_ROM_optimization_data(output, run_name)
    serialize(joinpath(run_directory, "window_history.jls"), output.window_history)
    serialize(joinpath(run_directory, "validation_history.jls"), output.validation_history)
    serialize(joinpath(run_directory, "evaluation_history.jls"), output.validation_history)
    return run_directory
end
