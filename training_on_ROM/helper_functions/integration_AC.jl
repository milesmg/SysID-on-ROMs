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


#### Misc. General Tools ###

function help(f)
    doc = Base.Docs.doc(f)
    show(stdout, MIME"text/plain"(), doc)
    println()
    return nothing
end


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

"""
Calculate RHS of AC with NN and parameters as argument
- args: (du, u, p, t, nn, state)
"""
function rhs_ac_NN!(du, u, p, t, nn, state)
    (; ε2, Δx, θ) = p
    lap1d!(du, u, 1 / Δx^2)
    for i in eachindex(du)
        du[i] = ε2 * du[i] - Fnn(u[i], nn, θ, state)
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

#### Train Neural Network ####


"""
a loss function (MSE) to train our model with
- args:(u_ref_obs,prob, θ, alg, t_obs, sensalg)
    - params should have the structure: ComponentVector(ε2=ε2, Δx=Δx, θ=θ)

"""
function loss_ref_F(u_ref_obs,prob, p, alg, t_obs, sensalg)
    r = model_FNN(prob, p, alg, t_obs, sensalg).u - u_ref_obs
    return 0.5 * p.Δx * sum(sum(abs2, ri) for ri in r)
end

"""
Get the parameters for a PDE-based optimization in order

- args: 
    - (;N=256, 
    - L=1.0, 
    - ε2 = 1e-2, 
    - tspan = (0.0, 2.0), 
    - N_obs = 10,
    - u₀ = nothing (constructs the default tanh profile),
    - h = 8,)
    - seed = 1,


- returns: (; prob, p₀, t_obs, nn, state, x, u₀)
    - prob is a neural_ODE_prob with parameters p₀
    - p₀ = ComponentVector(ε2=ε2, Δx=Δx, θ=ps₀)
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

    prob = neural_ODE_prob(u₀, tspan, p₀, nn, state)
    return (; prob, p₀, t_obs, nn, state, x, u₀)
end


"""
Builds an optimization problem based on your previous parametrizations
- args: 
    - u_ref (the whole solution object)
    - prob (the neural ODE problem)
    - t_obs,
    - p₀;
    - alg = TRBDF2(),
    - sensalg = GaussAdjoint(autojacvec=ReverseDiffVJP(true)),


- returns: optprob
"""
function set_up_optimization(
    u_ref,
    prob,
    t_obs,
    p₀;
    alg = TRBDF2(),
    sensalg = GaussAdjoint(autojacvec=ReverseDiffVJP(true)),
    )

    adtype = Optimization.AutoZygote()
    θ₀, re = Optimisers.destructure(p₀.θ)
    ### ADJUSTED: Count complete neural-PDE objective evaluations without differentiating the counter update.
    model_evaluation_count = Ref(0)

    u_ref_obs = [u_ref(ti) for ti in t_obs]

    optf = Optimization.OptimizationFunction(
        (θ, evaluation_count) -> begin
            Zygote.ignore() do
                evaluation_count[] += 1
            end
            params = ComponentVector(ε2=p₀.ε2, Δx=p₀.Δx, θ=re(θ))
            loss_ref_F(
                u_ref_obs,
                prob,
                params,
                alg,
                t_obs,
                sensalg,
            )
        end,
        adtype,
    )

    ### ADJUSTED: Store the evaluation counter as the fixed OptimizationProblem data.
    return Optimization.OptimizationProblem(optf, copy(θ₀), model_evaluation_count)
end


"""
Runs a full (Adam) optimization and returns parameter and model-evaluation histories.

- args: (optprob;η = 5e-2,β = (0.9, 0.99), N_iter = 400, warmup = true, save_frequency = nothing)

- returns: (; result, parameter_history, evaluation_history)
"""
function run_full_optimization(optprob;η = 5e-2,
                        β = (0.9, 0.99),
                        N_iter = 400,
                        warmup = true,
                        save_frequency = nothing,
                        )

    println("Optimization Params: η = $η; β = $β; N_iter = $N_iter")

    # run optimization with printing
    last_time = Ref{Float64}(time())
    ### ADJUSTED: Default parameter snapshots to ten evenly spaced saves over the requested run.
    save_frequency = isnothing(save_frequency) ? max(1, cld(N_iter, 10)) : save_frequency
    parameter_history = [(iteration=0, θ=copy(optprob.u0))]
    evaluation_history = NamedTuple{(:iteration, :count), Tuple{Int, Int}}[]
    model_evaluation_count = optprob.p
    previous_evaluation_count = Ref(0)
    last_iteration = Ref(0)

    function cb_nn(state, loss)
        now = time()
        elapsed = now - last_time[]
        last_time[] = now
        last_iteration[] = state.iter

        ### ADJUSTED: Record model evaluations since the previous callback and periodically copy NN parameters.
        push!(evaluation_history, (
            iteration=state.iter,
            count=model_evaluation_count[] - previous_evaluation_count[],
        ))
        previous_evaluation_count[] = model_evaluation_count[]

        ### ADJUSTED: Avoid duplicate parameter snapshots when Optimization.jl repeats a final callback.
        if state.iter > 0 && state.iter % save_frequency == 0 && parameter_history[end].iteration != state.iter
            push!(parameter_history, (iteration=state.iter, θ=copy(state.u)))
        end

        if state.iter % 10 == 0
            println(
                "iteration = $(state.iter), loss = $loss, ",
                "last iteration = $(round(elapsed; digits=2)) s"
            )
        end

        return false
    end

    if warmup
        println("Warming up")

        # Warm-up compilation
        Optimization.solve(
            optprob,
            OptimizationOptimisers.Adam(η, β);
            maxiters=1,
        )

        println("Warmup Complete")
    end

    ### ADJUSTED: Exclude warmup evaluations and time from the returned training histories.
    model_evaluation_count[] = 0
    previous_evaluation_count[] = 0
    last_time[] = time()
    
    println("\nBeginning Optimization\n")

    # Timed full solve
    @time res = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(η, β);
        maxiters=N_iter,
        callback=cb_nn,
    )

    println("\nCompleted Optimization\n")
    ### ADJUSTED: Always retain the final parameters when the run ends between snapshot intervals.
    if parameter_history[end].iteration != last_iteration[]
        push!(parameter_history, (iteration=last_iteration[], θ=copy(res.u)))
    end

    return (; result=res, parameter_history, evaluation_history)
end
