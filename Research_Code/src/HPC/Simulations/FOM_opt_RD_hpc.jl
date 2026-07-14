### ADJUSTED: Add two-species reaction-diffusion FOM training with fixed s1 and trainable s2.
include(joinpath(@__DIR__, "..", "Tools", "hpc_logging.jl"))
include(joinpath(@__DIR__, "..", "..", "Misc.", "run_name_guard.jl"))

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

##### RD Grid, State, And Operator Helpers #####

"""Validate the RD spatial dimension."""
function rd_validate_dimension(dimension)
    dim = Int(dimension)
    dim in (1, 2) || error("dimension must be 1 or 2; got $dimension")
    return dim
end

"""Validate the RD boundary condition."""
function rd_validate_boundary_condition(boundary_condition)
    bc = lowercase(replace(strip(string(boundary_condition)), "-" => "_", " " => "_"))
    bc in ("neumann", "homogeneous_neumann") || error("RD boundary_condition must be neumann; got $boundary_condition")
    return "neumann"
end

"""Validate the RD learner name."""
function rd_validate_learner(learner)
    name = lowercase(strip(string(learner)))
    name in ("fixed", "nn", "polynomial") || error("learner must be fixed, nn, or polynomial; got $learner")
    return name
end

"""Return a Neumann grid on `[0, L]`."""
function rd_grid(N::Integer, L)
    N >= 2 || error("RD requires N >= 2 for the Neumann stencil")
    Δx = L / (N - 1)
    return (; Δx, x=collect(range(zero(L), L; length=N)))
end

rd_spatial_length(N::Integer, dimension) = N ^ rd_validate_dimension(dimension)
rd_spatial_measure(Δx, dimension) = Δx ^ rd_validate_dimension(dimension)

"""Infer the per-axis grid size from a flattened spatial state."""
function rd_grid_size(state_length::Integer, dimension)
    dim = rd_validate_dimension(dimension)
    dim == 1 && return state_length
    N = round(Int, sqrt(state_length))
    N * N == state_length || error("2D states must have square flattened length; got $state_length")
    return N
end

"""Construct the negative-semidefinite 1D Neumann Laplacian."""
function rd_laplacian_1d_matrix(N::Integer, Δx)
    N >= 2 || error("RD requires N >= 2")
    scale = one(Δx) / Δx^2
    rows = Int[]
    cols = Int[]
    vals = typeof(scale)[]
    append!(rows, (1, 1))
    append!(cols, (1, 2))
    append!(vals, (-scale, scale))
    for i in 2:N-1
        append!(rows, (i, i, i))
        append!(cols, (i, i - 1, i + 1))
        append!(vals, (-2scale, scale, scale))
    end
    append!(rows, (N, N))
    append!(cols, (N, N - 1))
    append!(vals, (-scale, scale))
    return sparse(rows, cols, vals, N, N)
end

"""Construct the flattened 1D or 2D Neumann Laplacian."""
function rd_laplacian_matrix(N::Integer, Δx, dimension)
    dim = rd_validate_dimension(dimension)
    lap1 = rd_laplacian_1d_matrix(N, Δx)
    dim == 1 && return lap1
    identity_N = sparse(I, N, N)
    return kron(identity_N, lap1) + kron(lap1, identity_N)
end

"""Return the flattened two-species default initial condition."""
function rd_default_initial_condition(N::Integer, L, dimension; amplitude=0.05)
    dim = rd_validate_dimension(dimension)
    grid = rd_grid(N, L)
    x = grid.x
    n = rd_spatial_length(N, dim)
    if dim == 1
        v1 = 1 .+ amplitude .* cos.(2π .* x ./ L)
        v2 = 1 .+ amplitude .* sin.(2π .* x ./ L)
    else
        v1 = [1 + amplitude * cos(2π * x[i] / L) * cos(2π * x[j] / L) for i in 1:N, j in 1:N]
        v2 = [1 + amplitude * sin(2π * x[i] / L) * sin(2π * x[j] / L) for i in 1:N, j in 1:N]
    end
    length(vec(v1)) == n || error("internal RD initial-condition size error")
    return vcat(vec(v1), vec(v2))
end

"""Normalize a two-species initial condition to a length-`2N^d` vector."""
function rd_normalize_initial_condition(u₀, N::Integer, dimension)
    n = rd_spatial_length(N, dimension)
    if u₀ isa Tuple && length(u₀) == 2
        return vcat(vec(copy(u₀[1])), vec(copy(u₀[2])))
    elseif ndims(u₀) == 1 && length(u₀) == 2n
        return copy(u₀)
    elseif ndims(u₀) == 2 && size(u₀) == (2, n)
        return copy(vec(u₀))
    elseif rd_validate_dimension(dimension) == 2 && ndims(u₀) == 3 && size(u₀) == (2, N, N)
        return copy(vec(u₀))
    end
    error("RD initial condition must be a length-$(2n) vector, a 2 x $n matrix, or two spatial fields")
end

"""Split a stacked RD state into the two species."""
function rd_split_state(u, N::Integer, dimension)
    n = rd_spatial_length(N, dimension)
    length(u) == 2n || error("stacked RD state must have length $(2n); got $(length(u))")
    return @view(u[1:n]), @view(u[n+1:2n])
end

"""Construct the sparse Jacobian sparsity pattern for the two-field RHS."""
function rd_jacobian_prototype(lap, D1, D2)
    n = size(lap, 1)
    identity_n = sparse(I, n, n)
    return [D1 .* lap + identity_n -identity_n; identity_n D2 .* lap + identity_n]
end

##### RD Reactions And Learned Nonlinearities #####

"""Evaluate the paper's fixed first reaction component."""
rd_s1(v1, v2) = v1 - v1^3 - v2 - 0.005
rd_s1_values(v1, v2) = v1 .- v1 .^ 3 .- v2 .- 0.005

"""Evaluate the paper's true/reference second reaction component."""
rd_s2_true(v1, v2) = 10 * (v1 - v2)
rd_s2_true_values(v1, v2) = 10 .* (v1 .- v2)

"""Evaluate a Lux pointwise network on paired species values."""
function rd_nn_batch(v1, v2, nn, θ, state)
    x = vcat(reshape(v1, 1, length(v1)), reshape(v2, 1, length(v2)))
    y, _ = Lux.apply(nn, x, θ, state)
    return vec(y)
end

"""Return ascending total-degree bivariate monomial exponents."""
function rd_polynomial_exponents(degree::Integer)
    degree >= 0 || error("polynomial_degree must be nonnegative")
    ### ADJUSTED: Express the fixed total-degree ordering directly as `(i, total-i)` pairs.
    return [(i, total - i) for total in 0:degree for i in 0:total]
end

"""Evaluate a bivariate polynomial using the fixed ascending total-degree ordering."""
function rd_polynomial_value(v1, v2, coefficients, exponents)
    value = zero(v1 + v2 + first(coefficients))
    @inbounds for k in eachindex(coefficients, exponents)
        i, j = exponents[k]
        value += coefficients[k] * v1^i * v2^j
    end
    return value
end

"""Evaluate a learned bivariate polynomial at paired species values."""
function rd_polynomial_values(v1, v2, coefficients, exponents)
    return [rd_polynomial_value(v1[i], v2[i], coefficients, exponents) for i in eachindex(v1)]
end

"""Return zero initial coefficients for a total-degree bivariate polynomial."""
rd_initial_polynomial_coefficients(degree::Integer) = zeros(Float64, length(rd_polynomial_exponents(degree)))

##### RD FOM RHS And Problem Setup #####

"""Evaluate the fixed two-species reference reaction-diffusion RHS."""
function rhs_rd!(du, u, p, t, lap, N, dimension)
    v1, v2 = rd_split_state(u, N, dimension)
    n = length(v1)
    Δv1 = lap * v1
    Δv2 = lap * v2
    s1 = rd_s1_values(v1, v2)
    s2 = rd_s2_true_values(v1, v2)
    @inbounds for i in 1:n
        du[i] = p.D1 * Δv1[i] + s1[i]
        du[n + i] = p.D2 * Δv2[i] + s2[i]
    end
    return nothing
end

"""Evaluate the NN-learned RD RHS, learning only `s2`."""
function rhs_rd_NN!(du, u, p, t, lap, N, dimension, nn, state)
    v1, v2 = rd_split_state(u, N, dimension)
    n = length(v1)
    Δv1 = lap * v1
    Δv2 = lap * v2
    s1 = rd_s1_values(v1, v2)
    s2 = rd_nn_batch(v1, v2, nn, p.θ, state)
    @inbounds for i in 1:n
        du[i] = p.D1 * Δv1[i] + s1[i]
        du[n + i] = p.D2 * Δv2[i] + s2[i]
    end
    return nothing
end

"""Evaluate the polynomial-learned RD RHS, learning only `s2`."""
function rhs_rd_polynomial!(du, u, p, t, lap, N, dimension, exponents)
    v1, v2 = rd_split_state(u, N, dimension)
    n = length(v1)
    Δv1 = lap * v1
    Δv2 = lap * v2
    s1 = rd_s1_values(v1, v2)
    s2 = rd_polynomial_values(v1, v2, p.θ, exponents)
    @inbounds for i in 1:n
        du[i] = p.D1 * Δv1[i] + s1[i]
        du[n + i] = p.D2 * Δv2[i] + s2[i]
    end
    return nothing
end

"""Construct an RD ODE problem for fixed, NN, or polynomial `s2`."""
function rd_ODE_problem(u₀, tspan, p₀, lap, N, dimension; learner="fixed", nn=nothing, state=nothing, exponents=rd_polynomial_exponents(3))
    learner_name = rd_validate_learner(learner)
    rhs! = if learner_name == "fixed"
        (du, u, p, t) -> rhs_rd!(du, u, p, t, lap, N, dimension)
    elseif learner_name == "nn"
        (du, u, p, t) -> rhs_rd_NN!(du, u, p, t, lap, N, dimension, nn, state)
    else
        (du, u, p, t) -> rhs_rd_polynomial!(du, u, p, t, lap, N, dimension, exponents)
    end
    return ODEProblem(ODEFunction(rhs!; jac_prototype=rd_jacobian_prototype(lap, p₀.D1, p₀.D2)), u₀, tspan, p₀)
end

"""Build the fixed reference RD trajectory."""
function build_rd_reference(; N=64, L=1.0, D1=2.8e-4, D2=5.0e-2, tfinal=1.0, reference_dt_factor=0.5, dimension=2, boundary_condition="neumann", u₀=nothing)
    dimension = rd_validate_dimension(dimension)
    boundary_condition = rd_validate_boundary_condition(boundary_condition)
    grid = rd_grid(N, L)
    Δx = grid.Δx
    u₀ = isnothing(u₀) ? rd_default_initial_condition(N, L, dimension) : rd_normalize_initial_condition(u₀, N, dimension)
    lap = rd_laplacian_matrix(N, Δx, dimension)
    tspan = (0.0, tfinal)
    Δt = reference_dt_factor * Δx^2 / max(D1, D2, eps(Float64))
    save_count = min(max(2, floor(Int, (tspan[2] - tspan[1]) / Δt) + 1), 500)
    t = collect(LinRange(tspan[1], tspan[2], save_count))
    p₀ = (; D1, D2, Δx, Δmeasure=rd_spatial_measure(Δx, dimension), dimension, N, boundary_condition)
    prob = rd_ODE_problem(u₀, tspan, p₀, lap, N, dimension; learner="fixed")
    hpc_log_timed("build_rd_reference", "Reference solve: N=$N, Δx=$Δx, saved_times=$(length(t)), dimension=$dimension")
    u_ref = solve(prob, TRBDF2(autodiff=AutoFiniteDiff()); saveat=t)
    return (; u_ref, prob, p₀, u₀=copy(u₀), lap, x=grid.x, y=dimension == 1 ? nothing : copy(grid.x), t, tspan, Δx, Δt, dimension, boundary_condition, state_shape=dimension == 1 ? (2, N) : (2, N, N), N, L, D1, D2)
end

"""Prepare a trainable RD FOM with NN or polynomial `s2`."""
function prepare_RD_FOM_optimization(; N=64, L=1.0, D1=2.8e-4, D2=5.0e-2, tspan=(0.0, 1.0), N_obs=10, dimension=2, boundary_condition="neumann", u₀=nothing, learner="nn", h=8, seed=1, polynomial_degree=3, polynomial_initial_coefficients=nothing)
    dimension = rd_validate_dimension(dimension)
    boundary_condition = rd_validate_boundary_condition(boundary_condition)
    learner = rd_validate_learner(learner)
    grid = rd_grid(N, L)
    Δx = grid.Δx
    u₀ = isnothing(u₀) ? rd_default_initial_condition(N, L, dimension) : rd_normalize_initial_condition(u₀, N, dimension)
    lap = rd_laplacian_matrix(N, Δx, dimension)
    t_obs = collect(LinRange(tspan[1] + (tspan[2] - tspan[1]) / (N_obs - 1), tspan[2], max(1, N_obs - 1)))
    nn = nothing
    state = nothing
    exponents = rd_polynomial_exponents(polynomial_degree)
    θ₀ = if learner == "nn"
        rng = MersenneTwister(seed)
        nn = Chain(Dense(2 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
        ps₀, state₀ = Lux.setup(rng, nn)
        state = state₀
        fmap(x -> Float64.(x), ps₀)
    elseif learner == "polynomial"
        isnothing(polynomial_initial_coefficients) ? rd_initial_polynomial_coefficients(polynomial_degree) : Float64.(collect(polynomial_initial_coefficients))
    else
        zeros(Float64, 0)
    end
    polynomial_degree = learner == "polynomial" ? length(θ₀) == 0 ? 0 : begin
        degree = 0
        while length(rd_polynomial_exponents(degree)) < length(θ₀)
            degree += 1
        end
        degree
    end : nothing
    Δmeasure = rd_spatial_measure(Δx, dimension)
    p₀ = ComponentVector(D1=D1, D2=D2, Δx=Δx, Δmeasure=Δmeasure, θ=θ₀)
    prob = rd_ODE_problem(u₀, tspan, p₀, lap, N, dimension; learner, nn, state, exponents)
    run_params = (; equation="rd", N, L, D1, D2, ε2=nothing, k=nothing, sigma=nothing, mean_c=nothing, Δx, Δmeasure, dimension, boundary_condition, state_shape=dimension == 1 ? (2, N) : (2, N, N), x=copy(grid.x), y=dimension == 1 ? nothing : copy(grid.x), tspan, N_obs, t_obs=copy(t_obs), u₀=copy(u₀), learner, model_type=learner, state_components=("v1", "v2"), learned_component="s2", reference_reactions="s1=v1-v1^3-v2-0.005; s2=10*(v1-v2)", h=learner == "nn" ? h : nothing, network_architecture=learner == "nn" ? (2, h, h, 1) : nothing, activation=learner == "nn" ? "tanh" : "polynomial", polynomial_degree, polynomial_coefficient_order=learner == "polynomial" ? "ascending total-degree monomials (i,j)" : nothing, polynomial_initial_coefficients=learner == "polynomial" ? copy(θ₀) : nothing, seed)
    return (; prob, p₀, t_obs, nn, state, exponents, lap, x=grid.x, u₀=copy(u₀), run_params)
end

##### RD FOM Optimization #####

"""Compute one spatially weighted RD FOM window loss over both fields."""
function variable_window_RD_FOM_loss(window, prob, p, alg, sensalg, normalization)
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

"""Average or sum RD FOM window losses."""
function variable_window_RD_FOM_batch_loss(batch, prob, p, alg, sensalg, normalization)
    total = zero(eltype(first(batch).u0))
    for window in batch
        total += variable_window_RD_FOM_loss(window, prob, p, alg, sensalg, normalization)
    end
    return normalization == "mean" ? total / length(batch) : total
end

"""Run staged Adam training for an RD FOM NN or polynomial `s2`."""
function run_variable_window_RD_FOM_optimization(u_ref, prob, p₀; run_params, eta=5e-2, beta=(0.9, 0.99), N_iter=400, window_T=nothing, window_N_obs=nothing, window_start_policy="beginning", batch_size=1, loss_normalization="mean", window_seed=1, validation_N_obs=run_params.N_obs, alg=TRBDF2(autodiff=AutoFiniteDiff()), sensalg=GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP()), warmup=true, save_frequency=nothing, print_frequency=10)
    core = run_variable_window_stages(u_ref, prob, p₀; optimization_data=(; run_params), materialize_model_batch=materialize_batch, rebuild_params=(p, re, theta) -> ComponentVector(D1=p.D1, D2=p.D2, Δx=p.Δx, Δmeasure=p.Δmeasure, θ=re(theta)), batch_loss=variable_window_RD_FOM_batch_loss, validation_N_obs, log_name="run_variable_window_RD_FOM_optimization", eta, beta, N_iter, window_T, window_N_obs, window_start_policy, batch_size, loss_normalization, window_seed, alg, sensalg, warmup, save_frequency, print_frequency)
    return (; core.result, core.parameter_history, run_params=merge(run_params, core.settings), core.final_loss, core.final_training_loss, core.final_full_trajectory_loss, core.window_history, core.validation_history)
end

##### RD FOM Save Helpers #####

### ADJUSTED: Save RD FOM data with the AC/CH file layout and RD metadata fields.
"""Save RD FOM optimization data under the standard run directory."""
function save_RD_FOM_optimization_data(output, run_name::AbstractString)
    data_root = normpath(joinpath(@__DIR__, "..", "..", "..", "Optimization", "Data"))
    run_directory = assert_run_name_available(run_name; data_root)
    mkpath(data_root)
    mkdir(run_directory)
    run_params = output.run_params.model_type == "polynomial" ? merge(output.run_params, (; polynomial_final_coefficients=copy(last(output.parameter_history).θ))) : output.run_params
    serialize(joinpath(run_directory, "parameter_history.jls"), output.parameter_history)
    serialize(joinpath(run_directory, "run_params.jls"), run_params)
    saved_metadata = merge(run_params, (; saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), julia_version=VERSION, initial_loss=first(output.parameter_history).loss, final_loss=output.final_loss, parameter_snapshots=length(output.parameter_history)))
    open(joinpath(run_directory, "metadata.txt"), "w") do io
        for (name, value) in pairs(saved_metadata)
            print(io, name, " = ")
            show(io, value)
            println(io)
        end
    end
    return run_directory
end

### ADJUSTED: Save RD variable-window histories under the standard FOM filenames.
function save_variable_window_RD_FOM_optimization_data(output, run_name::AbstractString)
    run_directory = save_RD_FOM_optimization_data(output, run_name)
    serialize(joinpath(run_directory, "window_history.jls"), output.window_history)
    serialize(joinpath(run_directory, "validation_history.jls"), output.validation_history)
    serialize(joinpath(run_directory, "evaluation_history.jls"), output.validation_history)
    return run_directory
end
