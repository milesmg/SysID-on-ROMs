include(joinpath(@__DIR__, "..", "..", "HPC_compatibility", "hpc_logging.jl"))
### ADJUSTED: Load the run-name guard from its moved Misc. directory.
include(joinpath(@__DIR__, "..", "Misc.", "run_name_guard.jl"))

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
hpc_log_package("LinearSolve", "Loading")
using LinearSolve
hpc_log_package("LinearSolve", "Loaded")
hpc_log_package("OrdinaryDiffEq", "Loading")
using OrdinaryDiffEq
hpc_log_package("OrdinaryDiffEq", "Loaded")
hpc_log_package("OrdinaryDiffEqSDIRK", "Loading")
using OrdinaryDiffEqSDIRK
hpc_log_package("OrdinaryDiffEqSDIRK", "Loaded")
hpc_log_package("OrdinaryDiffEqLowOrderRK", "Loading")
using OrdinaryDiffEqLowOrderRK
hpc_log_package("OrdinaryDiffEqLowOrderRK", "Loaded")
hpc_log_package("SciMLSensitivity", "Loading")
using SciMLSensitivity
hpc_log_package("SciMLSensitivity", "Loaded")
hpc_log_package("ADTypes", "Loading")
using ADTypes
hpc_log_package("ADTypes", "Loaded")
hpc_log_package("Zygote", "Loading")
using Zygote
hpc_log_package("Zygote", "Loaded")
hpc_log_package("Enzyme", "Loading")
using Enzyme
hpc_log_package("Enzyme", "Loaded")
hpc_log_package("Optimization", "Loading")
using Optimization
hpc_log_package("Optimization", "Loaded")
hpc_log_package("OptimizationOptimisers", "Loading")
using OptimizationOptimisers
hpc_log_package("OptimizationOptimisers", "Loaded")
hpc_log_package("OptimizationOptimJL", "Loading")
using OptimizationOptimJL
hpc_log_package("OptimizationOptimJL", "Loaded")
hpc_log_package("LineSearches", "Loading")
using LineSearches
hpc_log_package("LineSearches", "Loaded")
hpc_log_package("Lux", "Loading")
using Lux
hpc_log_package("Lux", "Loaded")
hpc_log_package("Functors", "Loading")
using Functors
hpc_log_package("Functors", "Loaded")
hpc_log_package("Plots", "Loading")
using Plots
hpc_log_package("Plots", "Loaded")
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


#### Train Neural Network ####


"""Compute one window's spatially weighted FOM loss."""
function variable_window_loss(window, prob, p, alg, sensalg, normalization)
    window_prob = remake(prob; u0=window.u0, tspan=(window.t_start, window.t_end), p=p)
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
    return normalization == "mean" ? loss / length(window.u_ref_obs) : loss
end

"""Average or sum the FOM windows scheduled for one optimizer iteration."""
function variable_window_batch_loss(batch, prob, p, alg, sensalg, normalization)
    total = zero(eltype(first(batch).u0))
    for window in batch
        total += variable_window_loss(window, prob, p, alg, sensalg, normalization)
    end
    return normalization == "mean" ? total / length(batch) : total
end

"""
Get the parameters for a PDE-based optimization in order.

- args:
    - `N=256`: number of spatial grid points
    - `L=1.0`: spatial-domain length
    - `ε2=1e-2`: diffusion parameter
    - `tspan=(0.0, 2.0)`: integration time span
    - `N_obs=10`: number of observation times
    - `u₀=nothing`: initial state; constructs the default tanh profile if omitted
    - `h=8`: hidden-layer width
    - `seed=1`: neural-network initialization seed

- returns: `(; prob, p₀, t_obs, nn, state, x, u₀, run_params)`
    - `prob`: neural ODE problem with initial parameters `p₀`
    - `p₀`: `ComponentVector(ε2=ε2, Δx=Δx, θ=ps₀)`
    - `t_obs`: observation times
    - `nn`: Lux neural-network architecture
    - `state`: Lux model state
    - `x`: spatial grid
    - `u₀`: initial PDE state
    - `run_params`: named tuple containing the grid, PDE, initial-state,
      observation-time, neural-network architecture, and random-seed settings
"""
function prepare_for_optimization(;N=256,
                                L=1.0, 
                                ε2 = 1e-2,
                                tspan = (0.0, 2.0),
                                N_obs = 10,
                                u₀ = nothing,
                                h = 8,
                                seed = 1,
                                )

    Δx = L/(N+1)
    x = L*Δx*collect(1:N)

    u₀ = isnothing(u₀) ? tanh.((x .- L/2) / sqrt(2ε2)) : u₀

    t_obs = collect(LinRange(tspan[1] + (tspan[2]-tspan[1])/(N_obs-1), tspan[2], N_obs-1))

    rng = MersenneTwister(seed)
    nn = Chain(Dense(1 => h, tanh), Dense(h => h, tanh), Dense(h => 1))

    ps₀, state = Lux.setup(rng, nn)
    ps₀ = fmap(x -> Float64.(x), ps₀)
    p₀ = ComponentVector(ε2=ε2, Δx=Δx, θ=ps₀)

    run_params = (;
        N, L, ε2, Δx,
        x=copy(x),
        tspan,
        N_obs,
        t_obs=copy(t_obs),
        u₀=copy(u₀),
        h,
        network_architecture=(1, h, h, 1),
        activation="tanh",
        seed,
    )

    prob = neural_ODE_prob(u₀, tspan, p₀, nn, state)
    return (; prob, p₀, t_obs, nn, state, x, u₀, run_params)
end


### ADJUSTED: Keep the HPC FOM path focused on the central variable-window optimizer.

"""
Run staged Adam FOM training on deterministic variable-length windows.

Window settings are scalar or same-length schedules with `eta`/`N_iter`.
Defaults use the whole trajectory, a batch size of one, and mean loss.
"""
function run_variable_window_optimization(
    u_ref,
    prob,
    p₀;
    run_params,
    eta=5e-2,
    beta=(0.9, 0.99),
    N_iter=400,
    window_T=nothing,
    window_N_obs=nothing,
    window_start_policy="beginning",
    batch_size=1,
    loss_normalization="mean",
    window_seed=1,
    validation_N_obs=run_params.N_obs,
    alg=TRBDF2(autodiff=AutoFiniteDiff()),
    sensalg=GaussAdjoint(autojacvec=EnzymeVJP(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))),
    warmup=true,
    save_frequency=nothing,
    print_frequency=10,
)
    core = run_variable_window_stages(
        u_ref,
        prob,
        p₀;
        optimization_data=(; run_params),
        materialize_model_batch=materialize_batch,
        rebuild_params=(p, re, theta) -> ComponentVector(ε2=p.ε2, Δx=p.Δx, θ=re(theta)),
        batch_loss=variable_window_batch_loss,
        validation_N_obs,
        log_name="run_variable_window_optimization",
        eta,
        beta,
        N_iter,
        window_T,
        window_N_obs,
        window_start_policy,
        batch_size,
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
        run_params=merge(run_params, core.settings),
        core.final_loss,
        core.final_training_loss,
        core.final_full_trajectory_loss,
        core.window_history,
        core.validation_history,
    )
end


"""
Save an optimization output and its propagated `run_params` under
`Optimization/Data/<run_name>`.
"""
function save_optimization_data(output, run_name::AbstractString)
    data_root = normpath(joinpath(@__DIR__, "..", "..", "Optimization", "Data"))
    run_directory = assert_run_name_available(run_name; data_root)
    mkpath(data_root)
    mkdir(run_directory)

    serialize(joinpath(run_directory, "parameter_history.jls"), output.parameter_history)
    serialize(joinpath(run_directory, "run_params.jls"), output.run_params)

    saved_metadata = merge(
        output.run_params,
        (;
            saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            julia_version=VERSION,
            initial_loss=first(output.parameter_history).loss,
            final_loss=output.final_loss,
            parameter_snapshots=length(output.parameter_history),
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

"""Save variable-window FOM output under the standard run directory."""
function save_variable_window_optimization_data(output, run_name::AbstractString)
    run_directory = save_optimization_data(output, run_name)
    serialize(joinpath(run_directory, "window_history.jls"), output.window_history)
    serialize(joinpath(run_directory, "validation_history.jls"), output.validation_history)
    serialize(joinpath(run_directory, "evaluation_history.jls"), output.validation_history)
    return run_directory
end
