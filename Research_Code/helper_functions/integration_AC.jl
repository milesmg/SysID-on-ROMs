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


"""Generate the 1D laplacian with homogenous dirichlet boundary conditions 
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

"""
Purpose: this function does exactly what it seems like; set up a forwards problem, equipped with the right integrators 
both forward and reverse, given some new parameters 

- note that p here is ε2, k, Δx = p.ε2, p.k, p.Δx

"""
function model(p)
    solve(remake(prob; p=p), alg; dt=Δt, saveat=t_obs, sensealg=sensalg)
end

model(p, dt; adaptive=true) = solve(remake(prob; p=p), alg; dt=dt, adaptive=adaptive, saveat=t_obs, sensealg=sensalg)


function model_f(prop, θ, p, t=t_obs)
    p = ComponentVector(ε2=ε2, k=k, Δx=Δx, θ=θ)
    solve(remake(prob; p=p), alg; dt=Δt, saveat=t, sensealg=sensalg)
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

### ADJUSTED: Count each Fnn call made by the neural ODE without changing the existing evaluation method.
function Fnn(u, nn, θ, state, evaluation_count)
    evaluation_count[] += 1
    return Fnn(u, nn, θ, state)
end

"""
Calculate RHS of AC with NN and parameters as argument
- args: (du, u, p, t, nn, state)
"""
function rhs_ac_NN!(du, u, p, t, nn, state, evaluation_count)
    (; ε2, Δx, θ) = p
    lap1d!(du, u, 1 / Δx^2)
    for i in eachindex(du)
        du[i] = ε2 * du[i] - Fnn(u[i], nn, θ, state, evaluation_count)
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
    ### ADJUSTED: Give each neural ODE problem its own Fnn evaluation counter.
    nn_evaluation_count = Ref(0)
    rhs! = (du, u, p, t) -> rhs_ac_NN!(du, u, p, t, nn, state, nn_evaluation_count)
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
