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
    - normalization and Δmeasure are how we noramlize/scale the loss; see function above
    - reconstruct takes a state and produces something that can be compared to the reference; eg projecting a ROM state up, or adding back avg. mass in Cahn Hilliard
"""
function solve_window_loss(window::TrainingWindow, prob, p, alg, sensalg, normalization, Δmeasure, reconstruct)
    window_prob = remake(prob; u0=window.model_u0, tspan=(window.spec.t_start, window.spec.t_end), p=p)
    sol = solve(window_prob, alg; saveat=window.spec.t_obs, sensealg=sensalg)
    model_states = [reconstruct(state, prob, p) for state in sol.u]
    weighted_solution_loss(model_states, window.reference_observations, Δmeasure, normalization)
end


## TO ADD:
function l2_error_learned_function(grid, learner)

end

# actually, I think this can be done by adjusting the 'reconstruct' passed into solve_window_loss
function reduced_weighted_solution_loss()

end


