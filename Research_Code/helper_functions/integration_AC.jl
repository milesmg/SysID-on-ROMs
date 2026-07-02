using LinearAlgebra
using SparseArrays
using Random
using ComponentArrays
using LinearSolve
using OrdinaryDiffEq
using OrdinaryDiffEqSDIRK
using OrdinaryDiffEqLowOrderRK
using SciMLSensitivity
using ADTypes
using Zygote
using Enzyme
using Optimization
using OptimizationOptimisers
using OptimizationOptimJL
using LineSearches
using Lux
using Functors
using Plots



##### Misc. RHS Tools #####


"""Apply the 1D laplacian with homogenous dirichlet boundary conditions 
    args: (du, u, invΔx2)"""
function lap1d!(du, u, invΔx2)
    du[1] = (u[2] - 2u[1]) * invΔx2
    for i in 2:length(u)-1
        du[i] = (u[i-1] - 2u[i] + u[i+1]) * invΔx2
    end
    n = length(u)
    du[n] = (u[n-1] - 2u[n]) * invΔx2
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
RHS of the allen cahn equation
- args: (du, u, p, t)

"""
function rhs_ac!(du, u, p, t)
    ε2, k, Δx = p.ε2, p.k, p.Δx
    lap1d!(du, u, 1/Δx^2)
    for i in eachindex(du)
        du[i] = ε2*du[i] - k*(u[i]^3 - u[i])
    end
    return nothing
end


#### Integrate with Neural Network ####

"""
Define NN function that broadcasts over a current state vector
- args: (u, nn, θ, state)

NOTES: 
    - this is likely much more expensive than broadcasting a polynomial
    - it would be faster to batch this, but this function will work
"""
function Fnn(u, nn, θ, state)
    x = reshape([u], 1, :)
    y, _ = Lux.apply(nn, x, θ, state)
    y[1]
end

"""
Calculate RHS of AC with NN and parameters as argument
- args: (du, u, p, t, nn, state)
"""
### ADJUSTED: Use the neural RHS with only the model inputs.
function rhs_ac_NN!(du, u, p, t, nn, state)
    (; ε2, Δx, θ) = p
    lap1d!(du, u, 1 / Δx^2)
    for i in eachindex(du)
        du[i] = ε2 * du[i] + Fnn(u[i], nn, θ, state)
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
    ### ADJUSTED: Build the neural ODE closure with only the model context it needs.
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
