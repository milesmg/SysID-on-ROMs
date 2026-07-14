### ADJUSTED: Add nonlocal Cahn-Hilliard FOM training helpers separate from Allen-Cahn code.
include(joinpath(@__DIR__, "..", "Tools", "hpc_logging.jl"))
### ADJUSTED: Load the run-name guard from its moved Misc. directory.
include(joinpath(@__DIR__, "..", "..", "Misc.", "run_name_guard.jl"))

hpc_log_package("LinearAlgebra", "Loading")
using LinearAlgebra
hpc_log_package("LinearAlgebra", "Loaded")
hpc_log_package("SparseArrays", "Loading")
using SparseArrays
hpc_log_package("SparseArrays", "Loaded")
hpc_log_package("Statistics", "Loading")
using Statistics
hpc_log_package("Statistics", "Loaded")
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
hpc_log_package("ADTypes", "Loading")
using ADTypes
hpc_log_package("ADTypes", "Loaded")
hpc_log_package("Zygote", "Loading")
using Zygote
hpc_log_package("Zygote", "Loaded")
hpc_log_package("Mooncake", "Loading")
using Mooncake
hpc_log_package("Mooncake", "Loaded")
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

include("variable_window_common_hpc.jl")


##### CH Grid And Operator Helpers #####

"""Validate the CH spatial dimension flag."""
function ch_validate_dimension(dimension)
    dim = Int(dimension)
    dim in (1, 2) || error("dimension must be 1 or 2; got $dimension")
    return dim
end

"""Validate the CH learner name."""
function ch_validate_learner(learner)
    learner_name = lowercase(strip(string(learner)))
    learner_name in ("nn", "polynomial") || error("learner must be nn or polynomial; got $learner")
    return learner_name
end

"""Return periodic grid coordinates and spacing on `[0, L)`."""
function ch_periodic_grid(N::Integer, L)
    Δx = L / N
    x = Δx .* collect(0:N-1)
    return (; Δx, x)
end

"""Return the flattened state length for an `N`-point-per-axis CH grid."""
ch_expected_state_length(N::Integer, dimension) = N ^ ch_validate_dimension(dimension)

"""Return the spatial integration weight for a flattened 1D or 2D CH state."""
ch_spatial_measure(Δx, dimension) = Δx ^ ch_validate_dimension(dimension)

"""Infer the per-axis grid size for a flattened 1D or square 2D CH state."""
function ch_grid_size(state_length::Integer, dimension)
    dim = ch_validate_dimension(dimension)
    if dim == 1
        return state_length
    end
    N = round(Int, sqrt(state_length))
    N * N == state_length || error("2D states must have square flattened length; got $state_length")
    return N
end

"""Return a copied flattened CH initial condition after shape checking."""
function ch_normalize_initial_condition(u₀, N::Integer, dimension)
    dim = ch_validate_dimension(dimension)
    expected_length = ch_expected_state_length(N, dim)
    if dim == 1
        ndims(u₀) == 1 && length(u₀) == expected_length ||
            error("dimension=1 requires a vector initial condition of length $N; got size $(size(u₀))")
        return copy(u₀)
    end
    valid_vector = ndims(u₀) == 1 && length(u₀) == expected_length
    valid_matrix = ndims(u₀) == 2 && size(u₀) == (N, N)
    (valid_vector || valid_matrix) ||
        error("dimension=2 requires a vector of length $(N^2) or an $N x $N matrix initial condition; got size $(size(u₀))")
    return copy(vec(u₀))
end

"""Construct a small-amplitude periodic CH initial condition with prescribed mean."""
function ch_default_initial_condition(N::Integer, L, dimension; mean_c=0.0, amplitude=0.1)
    dim = ch_validate_dimension(dimension)
    grid = ch_periodic_grid(N, L)
    x = grid.x
    if dim == 1
        return mean_c .+ amplitude .* cos.(4π .* x ./ L)
    end
    u = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        u[i, j] = mean_c + amplitude * cos(4π * x[i] / L) * cos(4π * x[j] / L)
    end
    return vec(u)
end

"""Construct the 1D periodic positive Laplacian `D = -Delta`."""
function ch_positive_laplacian_1d_matrix(N::Integer, Δx)
    scale = one(Δx) / Δx^2
    rows = Int[]
    cols = Int[]
    vals = typeof(scale)[]
    for i in 1:N
        append!(rows, (i, i, i))
        append!(cols, (i, i == 1 ? N : i - 1, i == N ? 1 : i + 1))
        append!(vals, (2scale, -scale, -scale))
    end
    return sparse(rows, cols, vals, N, N)
end

"""Construct the flattened 1D or 2D periodic positive Laplacian `D = -Delta`."""
function ch_positive_laplacian_matrix(N::Integer, Δx, dimension)
    dim = ch_validate_dimension(dimension)
    D1 = ch_positive_laplacian_1d_matrix(N, Δx)
    if dim == 1
        return D1
    end
    I_N = sparse(I, N, N)
    return kron(I_N, D1) + kron(D1, I_N)
end

"""Return `u .- mean_c` without tracing a dynamic mean through the RHS."""
ch_zero_mean(u, mean_c) = u .- mean_c

"""Return the fixed cubic chemical-potential derivative `f(c)=c^3-c`."""
ch_cubic_values(u) = u .^ 3 .- u


##### CH Learned Nonlinearities #####

"""Evaluate a Lux scalar NN on every entry of a CH state as one batch."""
function ch_nn_batch(u, nn, θ, state)
    x = reshape(u, 1, length(u))
    y, _ = Lux.apply(nn, x, θ, state)
    return vec(y)
end

"""Evaluate `sum(coefficients[j+1] * u^j for j=0:degree)` by Horner's rule."""
function ch_polynomial_value(u, coefficients)
    value = zero(u + first(coefficients))
    @inbounds for j in length(coefficients):-1:1
        value = value * u + coefficients[j]
    end
    return value
end

"""Evaluate the learned CH polynomial nonlinearity at every entry of `u`."""
ch_polynomial_values(u, coefficients) = [ch_polynomial_value(ui, coefficients) for ui in u]

"""Initial coefficient vector for a CH polynomial learned nonlinearity."""
function ch_initial_polynomial_coefficients(polynomial_degree::Integer)
    polynomial_degree >= 0 || error("polynomial_degree must be nonnegative")
    return zeros(Float64, polynomial_degree + 1)
end


##### CH FOM RHS And Problem Setup #####

"""Evaluate the fixed-cubic nonlocal CH reference RHS."""
function rhs_ch!(du, c, p, t)
    return rhs_ch!(du, c, p, t, p.D)
end

"""Evaluate the fixed-cubic nonlocal CH reference RHS with an explicitly supplied operator."""
function rhs_ch!(du, c, p, t, D)
    Dc = D * c
    D2c = D * Dc
    f_values = ch_cubic_values(c)
    Df = D * f_values
    zero_mean_c = ch_zero_mean(c, p.mean_c)
    @inbounds @simd for i in eachindex(du)
        du[i] = -p.ε2 * D2c[i] - Df[i] - p.sigma * zero_mean_c[i]
    end
    return nothing
end

"""Evaluate the NN-learned nonlocal CH FOM RHS."""
function rhs_ch_NN!(du, c, p, t, D, nn, state)
    Dc = D * c
    D2c = D * Dc
    f_values = ch_nn_batch(c, nn, p.θ, state)
    Df = D * f_values
    zero_mean_c = ch_zero_mean(c, p.mean_c)
    @inbounds @simd for i in eachindex(du)
        du[i] = -p.ε2 * D2c[i] - Df[i] - p.sigma * zero_mean_c[i]
    end
    return nothing
end

"""Evaluate the polynomial-learned nonlocal CH FOM RHS."""
function rhs_ch_polynomial!(du, c, p, t, D)
    Dc = D * c
    D2c = D * Dc
    f_values = ch_polynomial_values(c, p.θ)
    Df = D * f_values
    zero_mean_c = ch_zero_mean(c, p.mean_c)
    @inbounds @simd for i in eachindex(du)
        du[i] = -p.ε2 * D2c[i] - Df[i] - p.sigma * zero_mean_c[i]
    end
    return nothing
end

"""Construct a CH ODEProblem for fixed, NN, or polynomial local nonlinearities."""
function ch_ODE_problem(u₀, tspan, p₀, D; learner="fixed", nn=nothing, state=nothing)
    learner_name = lowercase(strip(string(learner)))
    rhs! = if learner_name == "fixed"
        (du, u, p, t) -> rhs_ch!(du, u, p, t, D)
    elseif learner_name == "nn"
        (du, u, p, t) -> rhs_ch_NN!(du, u, p, t, D, nn, state)
    elseif learner_name == "polynomial"
        (du, u, p, t) -> rhs_ch_polynomial!(du, u, p, t, D)
    else
        error("learner must be fixed, nn, or polynomial; got $learner")
    end
    ### ADJUSTED: Match the full CH RHS sparsity pattern more closely than the pure biharmonic stencil.
    jac_prototype = D * D + D + sparse(I, size(D, 1), size(D, 2))
    return ODEProblem(ODEFunction(rhs!; jac_prototype), u₀, tspan, p₀)
end

"""
Build a fixed-cubic nonlocal CH reference solution.

Returns `(; u_ref, prob, p₀, u₀, D, x, y, t, tspan, Δx, Δt, dimension, state_shape, mean_c, sigma)`.
"""
function build_ch_reference(;
    N=128,
    L=1.0,
    ε2=1e-2,
    sigma,
    tfinal=1.0,
    reference_dt_factor=0.1,
    dimension=1,
    u₀=nothing,
    mean_c=0.0,
)
    dimension = ch_validate_dimension(dimension)
    grid = ch_periodic_grid(N, L)
    Δx = grid.Δx
    x = grid.x
    u₀ = isnothing(u₀) ?
        ch_default_initial_condition(N, L, dimension; mean_c) :
        ch_normalize_initial_condition(u₀, N, dimension)
    mean_c = mean(u₀)
    D = ch_positive_laplacian_matrix(N, Δx, dimension)
    tspan = (0.0, tfinal)
    Δt = reference_dt_factor * Δx^4 / max(ε2, eps(Float64))
    save_count = min(max(2, floor(Int, (tspan[2] - tspan[1]) / Δt) + 1), 500)
    t = collect(LinRange(tspan[1], tspan[2], save_count))
    ### ADJUSTED: Store Δx in reference parameters so downstream CH ROM metadata can mirror AC outputs.
    p₀ = (; ε2, sigma, mean_c, Δx, Δmeasure=ch_spatial_measure(Δx, dimension), D)
    prob = ch_ODE_problem(u₀, tspan, p₀, D; learner="fixed")
    hpc_log_timed("build_ch_reference", "Reference solve: N=$N, Δx=$Δx, saved_times=$(length(t)), dimension=$dimension")
    u_ref = solve(prob, TRBDF2(autodiff=AutoFiniteDiff()); saveat=t)
    return (;
        u_ref,
        prob,
        p₀,
        u₀=copy(u₀),
        D,
        x,
        y=dimension == 1 ? nothing : copy(x),
        t,
        tspan,
        Δx,
        ### ADJUSTED: Return the CH reference save spacing for the standard HPC runner diagnostics.
        Δt,
        dimension,
        state_shape=dimension == 1 ? (N,) : (N, N),
        N,
        L,
        ε2,
        sigma,
        mean_c,
    )
end

"""
Prepare a trainable nonlocal CH FOM optimization with NN or polynomial local chemistry.
"""
function prepare_CH_FOM_optimization(;
    N=128,
    L=1.0,
    ε2=1e-2,
    sigma,
    tspan=(0.0, 1.0),
    N_obs=10,
    dimension=1,
    u₀=nothing,
    mean_c=0.0,
    learner="nn",
    h=8,
    seed=1,
    polynomial_degree=3,
    polynomial_initial_coefficients=nothing,
)
    dimension = ch_validate_dimension(dimension)
    learner = ch_validate_learner(learner)
    grid = ch_periodic_grid(N, L)
    Δx = grid.Δx
    x = grid.x
    u₀ = isnothing(u₀) ?
        ch_default_initial_condition(N, L, dimension; mean_c) :
        ch_normalize_initial_condition(u₀, N, dimension)
    mean_c = mean(u₀)
    D = ch_positive_laplacian_matrix(N, Δx, dimension)
    t_obs = collect(LinRange(tspan[1] + (tspan[2] - tspan[1]) / (N_obs - 1), tspan[2], N_obs - 1))

    nn = nothing
    state = nothing
    θ₀ = if learner == "nn"
        rng = MersenneTwister(seed)
        nn = Chain(Dense(1 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
        ps₀, state₀ = Lux.setup(rng, nn)
        state = state₀
        fmap(x -> Float64.(x), ps₀)
    else
        isnothing(polynomial_initial_coefficients) ?
            ch_initial_polynomial_coefficients(polynomial_degree) :
            Float64.(collect(polynomial_initial_coefficients))
    end
    polynomial_degree = learner == "polynomial" ? length(θ₀) - 1 : nothing

    ### ADJUSTED: Keep Δx in trainable-problem parameters so rebuilt CH FOM parameters retain AC metadata fields.
    p₀ = ComponentVector(ε2=ε2, sigma=sigma, mean_c=mean_c, Δx=Δx, Δmeasure=ch_spatial_measure(Δx, dimension), θ=θ₀)
    prob = ch_ODE_problem(u₀, tspan, p₀, D; learner, nn, state)
    run_params = (;
        N,
        L,
        ε2,
        ### ADJUSTED: Preserve the AC metadata key even though CH uses sigma instead of k.
        k=nothing,
        sigma,
        Δx,
        Δmeasure=ch_spatial_measure(Δx, dimension),
        dimension,
        ### ADJUSTED: Preserve the AC boundary-condition metadata key for periodic-only CH runs.
        boundary_condition="periodic",
        state_shape=dimension == 1 ? (N,) : (N, N),
        x=copy(x),
        y=dimension == 1 ? nothing : copy(x),
        tspan,
        N_obs,
        t_obs=copy(t_obs),
        u₀=copy(u₀),
        mean_c,
        learner,
        model_type=learner,
        h=learner == "nn" ? h : nothing,
        network_architecture=learner == "nn" ? (1, h, h, 1) : nothing,
        activation=learner == "nn" ? "tanh" : "polynomial",
        polynomial_degree,
        polynomial_coefficient_order=learner == "polynomial" ? "ascending powers of u" : nothing,
        polynomial_initial_coefficients=learner == "polynomial" ? copy(θ₀) : nothing,
        seed,
    )
    return (; prob, p₀, t_obs, nn, state, D, x, u₀=copy(u₀), run_params)
end


##### CH FOM Optimization #####

"""Compute one window's spatially weighted CH FOM loss."""
function variable_window_CH_FOM_loss(window, prob, p, alg, sensalg, normalization)
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
    loss = 0.5 * p.Δmeasure * total
    return normalization == "mean" ? loss / length(window.u_ref_obs) : loss
end

"""Average or sum the CH FOM windows scheduled for one optimizer iteration."""
function variable_window_CH_FOM_batch_loss(batch, prob, p, alg, sensalg, normalization)
    total = zero(eltype(first(batch).u0))
    for window in batch
        total += variable_window_CH_FOM_loss(window, prob, p, alg, sensalg, normalization)
    end
    return normalization == "mean" ? total / length(batch) : total
end

"""Run staged Adam training for a CH FOM NN or polynomial local nonlinearity."""
function run_variable_window_CH_FOM_optimization(
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
    sensalg=GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP()),
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
        rebuild_params=(p, re, theta) -> ComponentVector(
            ε2=p.ε2,
            sigma=p.sigma,
            mean_c=p.mean_c,
            ### ADJUSTED: Preserve Δx through the shared variable-window parameter rebuild.
            Δx=p.Δx,
            Δmeasure=p.Δmeasure,
            θ=re(theta),
        ),
        batch_loss=variable_window_CH_FOM_batch_loss,
        validation_N_obs,
        ### ADJUSTED: Use a nonempty CH log name matching the AC optimizer logging style.
        log_name="run_variable_window_CH_FOM_optimization",
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


##### CH FOM Save Helpers #####

### ADJUSTED: Save CH FOM outputs with the same file layout and metadata merge as AC FOM outputs.
"""
Save a CH FOM optimization output and its propagated `run_params` under
`Optimization/Data/<run_name>`.
"""
function save_CH_FOM_optimization_data(output, run_name::AbstractString)
    data_root = normpath(joinpath(@__DIR__, "..", "..", "..", "Optimization", "Data"))
    run_directory = assert_run_name_available(run_name; data_root)
    mkpath(data_root)
    mkdir(run_directory)

    run_params = hasproperty(output.run_params, :model_type) && output.run_params.model_type == "polynomial" ?
        merge(output.run_params, (; polynomial_final_coefficients=copy(last(output.parameter_history).θ))) :
        output.run_params

    serialize(joinpath(run_directory, "parameter_history.jls"), output.parameter_history)
    serialize(joinpath(run_directory, "run_params.jls"), run_params)

    saved_metadata = merge(
        run_params,
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

### ADJUSTED: Save CH variable-window histories under the same names as AC FOM outputs.
"""Save variable-window CH FOM output under the standard run directory."""
function save_variable_window_CH_FOM_optimization_data(output, run_name::AbstractString)
    run_directory = save_CH_FOM_optimization_data(output, run_name)
    serialize(joinpath(run_directory, "window_history.jls"), output.window_history)
    serialize(joinpath(run_directory, "validation_history.jls"), output.validation_history)
    serialize(joinpath(run_directory, "evaluation_history.jls"), output.validation_history)
    return run_directory
end
