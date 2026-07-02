### ADJUSTED: Use shared flushed logging for package-load diagnostics.
include(joinpath(@__DIR__, "..", "..", "HPC_compatibility", "hpc_logging.jl"))

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


##### Misc. RHS Tools #####


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

##### FOM Integration and Modeling #####


"""
RHS of the Allen-Cahn equation.
- args: (du, u, p, t)

"""
function rhs_ac!(du, u, p, t)
    ε2, k, Δx = p.ε2, p.k, p.Δx
    lap1d!(du, u, 1/Δx^2)
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
- args: (du, u, p, t, nn, state)
"""
function rhs_ac_NN!(du, u, p, t, nn, state)
    (; ε2, Δx, θ) = p
    lap1d!(du, u, 1 / Δx^2)
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
function neural_ODE_prob(u₀, tspan, p₀, nn, state)
    N = length(u₀)
    rhs! = (du, u, p, t) -> rhs_ac_NN!(du, u, p, t, nn, state)
    jac_prototype = Tridiagonal(
        zeros(eltype(u₀), N - 1),
        zeros(eltype(u₀), N),
        zeros(eltype(u₀), N - 1),
    )
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
