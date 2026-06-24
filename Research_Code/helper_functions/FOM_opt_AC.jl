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
using Dates
using Serialization

include("integration_AC.jl")


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
    run_params,
    alg = TRBDF2(),
    sensalg = GaussAdjoint(autojacvec=ReverseDiffVJP(true)),
    )

    adtype = Optimization.AutoZygote()
    θ₀, re = Optimisers.destructure(p₀.θ)
    model_evaluation_count = prob.f.f.nn_evaluation_count

    u_ref_obs = [u_ref(ti) for ti in t_obs]

    run_params = merge(run_params, (;
        ode_algorithm=string(nameof(typeof(alg))),
        sensitivity_algorithm=string(sensalg),
        reference_algorithm=string(nameof(typeof(u_ref.alg))),
        reference_tspan=u_ref.prob.tspan,
        reference_parameters=u_ref.prob.p,
        reference_saved_times=copy(u_ref.t),
        reference_steps=length(u_ref.t) - 1,
    ))
    optimization_data = (; model_evaluation_count, run_params)

    optf = Optimization.OptimizationFunction(
        (θ, data) -> begin
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

    return Optimization.OptimizationProblem(optf, copy(θ₀), optimization_data)
end


"""
Runs a full (Adam) optimization and returns parameter and model-evaluation histories.

- args: (optprob;η = 5e-2,β = (0.9, 0.99), N_iter = 400, warmup = true, save_frequency = nothing)

- returns: (; result=res, parameter_history, evaluation_history, run_params). 
    - Both parameter_history and evaluation_history save [iterate, (θ or # of evals), loss] at save_frequency
"""
function run_full_optimization(optprob;η = 5e-2,
                        β = (0.9, 0.99),
                        N_iter = 400,
                        warmup = true,
                        save_frequency = nothing,
                        )

    # run optimization with printing
    last_time = Ref{Float64}(time())
    save_frequency = isnothing(save_frequency) ? max(1, cld(N_iter, 10)) : save_frequency
    initial_loss = optprob.f(optprob.u0, optprob.p)
    parameter_history = [(iteration=0, θ=copy(optprob.u0), loss=initial_loss)]
    evaluation_history = NamedTuple{(:iteration, :count, :loss), Tuple{Int, Int, typeof(initial_loss)}}[]
    model_evaluation_count = optprob.p.model_evaluation_count
    previous_evaluation_count = Ref(0)
    last_iteration = Ref(0)
    last_loss = Ref(initial_loss)

    println("Optimization Params: η = $η; β = $β; N_iter = $N_iter; save_frequency = $save_frequency; warmup = $warmup")


    function cb_nn(state, loss)
        now = time()
        elapsed = now - last_time[]
        last_time[] = now
        last_iteration[] = state.iter
        last_loss[] = loss

        push!(evaluation_history, (
            iteration=state.iter,
            count=model_evaluation_count[] - previous_evaluation_count[],
            loss=loss,
        ))
        previous_evaluation_count[] = model_evaluation_count[]

        if state.iter > 0 && state.iter % save_frequency == 0 && parameter_history[end].iteration != state.iter
            push!(parameter_history, (iteration=state.iter, θ=copy(state.u), loss=loss))
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
    if parameter_history[end].iteration != last_iteration[]
        push!(parameter_history, (iteration=last_iteration[], θ=copy(res.u), loss=last_loss[]))
    end

    run_params = merge(optprob.p.run_params, (;
        optimizer="Adam",
        η,
        β,
        N_iter,
        warmup,
        save_frequency,
    ))
    return (; result=res, parameter_history, evaluation_history, run_params)
end


"""
Save an optimization output and its propagated `run_params` under
`Optimization/Data/<run_name>`.
"""
function save_optimization_data(output, run_name::AbstractString)
    data_root = normpath(joinpath(@__DIR__, "..", "Optimization", "Data"))
    run_directory = joinpath(data_root, run_name)
    mkdir(run_directory)

    serialize(joinpath(run_directory, "parameter_history.jls"), output.parameter_history)
    serialize(joinpath(run_directory, "evaluation_history.jls"), output.evaluation_history)
    serialize(joinpath(run_directory, "run_params.jls"), output.run_params)

    saved_metadata = merge(
        output.run_params,
        (;
            saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
            julia_version=VERSION,
            initial_loss=first(output.parameter_history).loss,
            final_loss=last(output.parameter_history).loss,
            parameter_snapshots=length(output.parameter_history),
            evaluation_records=length(output.evaluation_history),
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
