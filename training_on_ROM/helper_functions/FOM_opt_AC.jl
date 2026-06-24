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

    u_ref_obs = [u_ref(ti) for ti in t_obs]

    optf = Optimization.OptimizationFunction(
        (θ, _) -> begin
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

    return Optimization.OptimizationProblem(optf, copy(θ₀))
end


"""
Runs a full (Adam) optimization

- args: (optprob;η = 5e-2,β = (0.9, 0.99), N_iter = 400, warmup = true,)

- returns: the result of the optimization problem
"""
function run_full_optimization(optprob;η = 5e-2,
                        β = (0.9, 0.99),
                        N_iter = 400,
                        warmup = true,
                        )

    println("Optimization Params: η = $η; β = $β; N_iter = $N_iter")

    # run optimization with printing
    last_time = Ref{Float64}(time())

    function cb_nn(state, loss)
        now = time()
        elapsed = now - last_time[]
        last_time[] = now

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
    
    println("\nBeginning Optimization\n")

    # Timed full solve
    @time res = Optimization.solve(
        optprob,
        OptimizationOptimisers.Adam(η, β);
        maxiters=N_iter,
        callback=cb_nn,
    )

    println("\nCompleted Optimization\n")
    return res
end
