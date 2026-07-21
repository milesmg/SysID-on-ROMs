# Here, we build the loss function for our optimization; this is the objective through which we backpropagate

"""
Compute the loss based on learned and ref. snapshots 
Args:
    - model_states: learned snapshots
    - reference_states: true snapshots
    - Δmeasure: L2 weighting
    - normalization: do we normalize loss by number of snapshots? (default = yes)
"""
function weighted_solution_loss(model_states, reference_states, Δmeasure, normalization)
    total = zero(eltype(first(model_states)))
    @inbounds for j in eachindex(model_states, reference_states)
        model_state = model_states[j]
        reference_state = reference_states[j]
        # added @inbounds 
        # Is this computation screwing me O(N) style in backprop? 
        @inbounds @simd for i in eachindex(model_state, reference_state)
            total += abs2(model_state[i] - reference_state[i])
        end
    end
    loss = 0.5 * Δmeasure * total
    normalization == "mean" ? loss / length(reference_states) : loss
end

"""
This is the function that actually does the forward solving during an optimization. 
Args:
    - window, a instance of the TrainingWindow struct (see types.jl)
    - prob is the optimization problem object
    - p is the parameters
    - alg defaults to TRBDF2(autodiff=AutoFiniteDiff()), the forward solver
    - sensalg defaults to GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP()), which solves the adjoint
    - normalization and Δmeasure are how we normalize/scale the loss; see function above
    - reconstruct takes a state and produces something that can be compared to the reference; eg projecting a ROM state up, or adding back avg. mass in Cahn Hilliard
    - project takes a reference and produces something that can be compared to the reduced state
    - loss_space: "REDUCED" or "FULL"; do we project a ROM up before we compute the loss?
"""
function solve_window_loss(window::TrainingWindow, prob, p, alg, sensalg, normalization, Δmeasure, reconstruct, project, loss_space)
    window_prob = remake(prob; u0=window.model_u0, tspan=(window.spec.t_start, window.spec.t_end), p=p)
    sol = solve(window_prob, alg; saveat=window.spec.t_obs, sensealg=sensalg)
    if loss_space == "REDUCED" && !isnothing(project)
        weighted_solution_loss(sol.u, [project(state) for state in window.reference_observations], Δmeasure, normalization)
    else
        weighted_solution_loss([reconstruct(state, prob, p) for state in sol.u], window.reference_observations, Δmeasure, normalization)
    end
end

"""Compute the learned-versus-reference reaction-function L2 error over common scalar bounds.
Args:
- prepared: a PreparedTraining struct (will tell us how our learner is rebuilt from θ)
- θ: parameters for our learned function
- bounds: the bounds for the domain of our integral"""
function learned_function_l2_error(prepared::PreparedTraining, θ, bounds)
    lower, upper = bounds
    grid, parameters = prepared.grid, prepared.config.parameters
    x = collect(LinRange(lower, upper, grid.N))
    Δx = (upper - lower) / (grid.N - 1)
    if prepared.equation_name == "ac"
        learned, reference = ac_values(x, prepared.learner, θ), ac_fixed_values(x, parameters.k)
        return sqrt(Δx * sum(abs2, learned .- reference))
    elseif prepared.equation_name == "ch"
        learned, reference = ch_values(x, prepared.learner, θ), ch_fixed_values(x)
        return sqrt(Δx * sum(abs2, learned .- reference))
    elseif prepared.equation_name == "rd"
        v1 = repeat(x; inner=grid.N)
        v2 = repeat(x; outer=grid.N)
        learned, reference = rd_values(v1, v2, prepared.learner, θ), rd_s2.(v1, v2)
        return sqrt(Δx^2 * sum(abs2, learned .- reference))
    end
end
