### ADJUSTED: Load HPC logging from the moved Tools directory.
include(joinpath(@__DIR__, "..", "Tools", "hpc_logging.jl"))

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


##### Misc. RHS Tools #####

"""Validate and normalize the spatial dimension flag."""
function validate_ac_dimension(dimension)
    dim = Int(dimension)
    dim in (1, 2) || error("dimension must be 1 or 2; got $dimension")
    return dim
end

### ADJUSTED: Normalize Allen-Cahn boundary-condition names before choosing Laplacian stencils.
"""Validate and normalize the Allen-Cahn boundary condition flag."""
function validate_ac_boundary_condition(boundary_condition)
    bc = lowercase(strip(string(boundary_condition)))
    bc = replace(bc, "-" => "_", " " => "_")
    if bc in ("dirichlet", "homogeneous_dirichlet", "homogenous_dirichlet")
        return "homogeneous_dirichlet"
    elseif bc == "periodic"
        return "periodic"
    end
    error("boundary_condition must be homogeneous_dirichlet or periodic; got $boundary_condition")
end

"""Return the flattened state length for an `N`-point-per-axis Allen-Cahn grid."""
expected_ac_state_length(N::Integer, dimension) = N ^ validate_ac_dimension(dimension)

"""Return the spatial integration weight for a flattened 1D or 2D state."""
spatial_measure(Δx, dimension) = Δx ^ validate_ac_dimension(dimension)

### ADJUSTED: Use endpoint-excluding spacing for periodic grids while preserving Dirichlet defaults.
"""Return coordinate vectors for an `N`-point-per-axis grid on `[0, L]`."""
function ac_grid(N::Integer, L, boundary_condition="homogeneous_dirichlet")
    bc = validate_ac_boundary_condition(boundary_condition)
    Δx = bc == "periodic" ? L / N : L / (N + 1)
    x = bc == "periodic" ? Δx .* collect(0:N-1) : L * Δx * collect(1:N)
    return (; Δx, x)
end

"""Infer the per-axis grid size for a flattened 1D or square 2D state."""
function ac_grid_size(state_length::Integer, dimension)
    dim = validate_ac_dimension(dimension)
    if dim == 1
        return state_length
    end
    N = round(Int, sqrt(state_length))
    N * N == state_length || error("2D states must have square flattened length; got $state_length")
    return N
end

"""Construct the default Allen-Cahn initial condition for 1D or 2D runs."""
function default_ac_initial_condition(N::Integer, L, ε2, dimension, boundary_condition="homogeneous_dirichlet")
    dim = validate_ac_dimension(dimension)
    grid = ac_grid(N, L, boundary_condition)
    x = grid.x
    if dim == 1
        return tanh.((x .- L / 2) ./ sqrt(2ε2))
    end
    radius = L / 4
    u = Matrix{Float64}(undef, N, N)
    @inbounds for j in 1:N, i in 1:N
        r = sqrt((x[i] - L / 2)^2 + (x[j] - L / 2)^2)
        u[i, j] = tanh((r - radius) / sqrt(2ε2))
    end
    return vec(u)
end

"""Return a flattened initial condition after checking it matches `dimension`."""
function normalize_ac_initial_condition(u₀, N::Integer, dimension)
    dim = validate_ac_dimension(dimension)
    expected_length = expected_ac_state_length(N, dim)
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


"""Apply the 1D laplacian with homogenous dirichlet boundary conditions.

- args: `(du, u, invΔx2)`
"""
function lap1d!(du, u, invΔx2)
    @inbounds du[1] = (u[2] - 2u[1]) * invΔx2
    @inbounds @simd for i in 2:length(u)-1
        du[i] = (u[i-1] - 2u[i] + u[i+1]) * invΔx2
    end
    n = length(u)
    @inbounds du[n] = (u[n-1] - 2u[n]) * invΔx2
    return nothing
end

### ADJUSTED: Add the 1D periodic Laplacian stencil for AC runs.
"""Apply the 1D Laplacian with periodic boundary conditions."""
function lap1d_periodic!(du, u, invΔx2)
    n = length(u)
    @inbounds du[1] = (u[n] - 2u[1] + u[min(2, n)]) * invΔx2
    @inbounds @simd for i in 2:n-1
        du[i] = (u[i-1] - 2u[i] + u[i+1]) * invΔx2
    end
    if n > 1
        @inbounds du[n] = (u[n-1] - 2u[n] + u[1]) * invΔx2
    end
    return nothing
end

"""Apply the flattened 2D Laplacian with homogeneous Dirichlet boundary conditions."""
function lap2d!(du, u, N::Integer, invΔx2)
    @inbounds for j in 1:N, i in 1:N
        idx = i + (j - 1) * N
        value = -4u[idx]
        i > 1 && (value += u[idx - 1])
        i < N && (value += u[idx + 1])
        j > 1 && (value += u[idx - N])
        j < N && (value += u[idx + N])
        du[idx] = value * invΔx2
    end
    return nothing
end

### ADJUSTED: Add the flattened 2D periodic Laplacian stencil for AC runs.
"""Apply the flattened 2D Laplacian with periodic boundary conditions."""
function lap2d_periodic!(du, u, N::Integer, invΔx2)
    @inbounds for j in 1:N, i in 1:N
        idx = i + (j - 1) * N
        left = i == 1 ? idx + (N - 1) : idx - 1
        right = i == N ? idx - (N - 1) : idx + 1
        down = j == 1 ? idx + (N - 1) * N : idx - N
        up = j == N ? idx - (N - 1) * N : idx + N
        du[idx] = (u[left] - 4u[idx] + u[right] + u[down] + u[up]) * invΔx2
    end
    return nothing
end

### ADJUSTED: Dispatch AC Laplacian application on the selected boundary condition.
"""Apply the 1D or flattened 2D Laplacian with the selected boundary condition."""
function lap_ac!(du, u, N::Integer, dimension, invΔx2, boundary_condition="homogeneous_dirichlet")
    ### ADJUSTED: Trust setup-time boundary validation so AD does not trace string normalization.
    bc = boundary_condition
    if dimension == 1
        bc == "periodic" ? lap1d_periodic!(du, u, invΔx2) : lap1d!(du, u, invΔx2)
    else
        bc == "periodic" ? lap2d_periodic!(du, u, N, invΔx2) : lap2d!(du, u, N, invΔx2)
    end
    return nothing
end

"""
Construct the sparse 1D Allen–Cahn diffusion matrix with homogeneous
Dirichlet boundary conditions.

- args: `(N, ε2, Δx)`
- returns: `A`, where `A*u == ε2*lap1d!(du, u, 1/Δx^2)`
"""
function get_lap1d_matrix(N, ε2, Δx)
    scale = ε2 / Δx^2
    return spdiagm(
        -1 => fill(scale, N - 1),
         0 => fill(-2scale, N),
         1 => fill(scale, N - 1),
    )
end

### ADJUSTED: Add sparse 1D periodic diffusion matrices for ROM construction.
"""Construct the sparse 1D Allen-Cahn diffusion matrix with periodic boundary conditions."""
function get_lap1d_periodic_matrix(N, ε2, Δx)
    scale = ε2 / Δx^2
    rows = Int[]
    cols = Int[]
    vals = typeof(scale)[]
    for i in 1:N
        append!(rows, (i, i, i))
        append!(cols, (i, i == 1 ? N : i - 1, i == N ? 1 : i + 1))
        append!(vals, (-2scale, scale, scale))
    end
    return sparse(rows, cols, vals, N, N)
end

"""Construct the sparse flattened 2D Allen-Cahn diffusion matrix."""
function get_lap2d_matrix(N, ε2, Δx, boundary_condition="homogeneous_dirichlet")
    ### ADJUSTED: Reuse the selected 1D sparse stencil when building 2D diffusion matrices.
    bc = validate_ac_boundary_condition(boundary_condition)
    L1 = bc == "periodic" ? get_lap1d_periodic_matrix(N, one(ε2), Δx) : get_lap1d_matrix(N, one(ε2), Δx)
    I_N = sparse(I, N, N)
    return ε2 * (kron(I_N, L1) + kron(L1, I_N))
end

### ADJUSTED: Thread boundary-condition selection through sparse AC diffusion matrices.
"""Construct the 1D or flattened 2D Allen-Cahn sparse diffusion matrix."""
function get_lap_ac_matrix(N, ε2, Δx, dimension, boundary_condition="homogeneous_dirichlet")
    dim = validate_ac_dimension(dimension)
    bc = validate_ac_boundary_condition(boundary_condition)
    if dim == 1
        return bc == "periodic" ? get_lap1d_periodic_matrix(N, ε2, Δx) : get_lap1d_matrix(N, ε2, Δx)
    end
    return get_lap2d_matrix(N, ε2, Δx, bc)
end

##### FOM Integration and Modeling #####


"""
RHS of the Allen-Cahn equation.
- args: (du, u, p, t)

"""
function rhs_ac!(du, u, p, t)
    ε2, k, Δx = p.ε2, p.k, p.Δx
    dimension = hasproperty(p, :dimension) ? p.dimension : 1
    N = hasproperty(p, :N) ? p.N : ac_grid_size(length(u), dimension)
    ### ADJUSTED: Apply the requested reference-solve boundary condition.
    boundary_condition = hasproperty(p, :boundary_condition) ? p.boundary_condition : "homogeneous_dirichlet"
    lap_ac!(du, u, N, dimension, 1/Δx^2, boundary_condition)
    @inbounds @simd for i in eachindex(du)
        du[i] = ε2*du[i] - k*(u[i]^3 - u[i])
    end
    return nothing
end


#### Integrate with Neural Network ####

"""
Define scalar NN evaluation for compatibility/debugging.
- args: (u, nn, θ, state)

NOTES: 
    - this is likely much more expensive than broadcasting a polynomial
    - the production RHS uses `Fnn_batch` to avoid one Lux call per grid point
"""
function Fnn(u, nn, θ, state)
    x = reshape([u], 1, :)
    y, _ = Lux.apply(nn, x, θ, state)
    y[1]
end

"""
Evaluate the NN on a full state vector as one Lux batch.

- args: `(u, nn, θ, state)`
- returns: vector-like NN outputs with one value per entry of `u`
"""
function Fnn_batch(u, nn, θ, state)
    x = reshape(u, 1, length(u))
    y, _ = Lux.apply(nn, x, θ, state)
    return vec(y)
end

"""
Calculate RHS of AC with batched NN evaluation and parameters as argument.
- args: (du, u, p, t, nn, state, dimension, N, boundary_condition)
"""
function rhs_ac_NN!(du, u, p, t, nn, state, dimension=1, N=length(u), boundary_condition="homogeneous_dirichlet")
    (; ε2, Δx, θ) = p
    ### ADJUSTED: Apply the requested neural-solve boundary condition.
    lap_ac!(du, u, N, dimension, 1 / Δx^2, boundary_condition)
    f = Fnn_batch(u, nn, θ, state)
    @inbounds @simd for i in eachindex(du)
        du[i] = ε2 * du[i] + f[i]
    end
    return nothing
end

"""
Set up an in-place Neural ODE problem
- args: (u₀, tspan, p₀, nn, state)
    - (; ε2, Δx, θ) = p
"""
function neural_ODE_prob(u₀, tspan, p₀, nn, state; N=nothing, dimension=1, boundary_condition="homogeneous_dirichlet")
    dim = validate_ac_dimension(dimension)
    ### ADJUSTED: Close over the selected boundary condition without putting strings in trainable parameters.
    bc = validate_ac_boundary_condition(boundary_condition)
    grid_N = isnothing(N) ? ac_grid_size(length(u₀), dim) : N
    u₀ = normalize_ac_initial_condition(u₀, grid_N, dim)
    rhs! = (du, u, p, t) -> rhs_ac_NN!(du, u, p, t, nn, state, dim, grid_N, bc)
    jac_prototype = dim == 1 && bc == "homogeneous_dirichlet" ?
        Tridiagonal(
            zeros(eltype(u₀), grid_N - 1),
            zeros(eltype(u₀), grid_N),
            zeros(eltype(u₀), grid_N - 1),
        ) :
        get_lap_ac_matrix(grid_N, one(eltype(u₀)), one(eltype(u₀)), dim, bc)
    f = ODEFunction(rhs!; jac_prototype)
    return ODEProblem(f, u₀, tspan, p₀)
end


"""
Solve a forward neural PDE given some parameters
- args:(prob, p, alg, t_obs, sensalg)
    - p should have the structure: ComponentVector(ε2=ε2, Δx=Δx, θ=θ)
"""
function model_FNN(prob, p, alg, t_obs, sensalg)
    solve(remake(prob; p=p), alg; saveat=t_obs, sensealg=sensalg)
end
