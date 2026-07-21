"""
Build a LearnerSetup struct; see types.jl
"""
function build_learner(learner, input_dim::Integer, h::Integer, seed::Integer, polynomial_degree::Integer)::LearnerSetup
    learner_name = lowercase(string(learner))
    if learner_name == "nn"
        rng = MersenneTwister(seed)
        nn = Chain(Dense(input_dim => h, tanh), Dense(h => h, tanh), Dense(h => 1))
        parameters, state = Lux.setup(rng, nn)
        return LearnerSetup(learner_name, nn, state, fmap(x -> Float64.(x), parameters),
                            h, seed, nothing, "tanh")
    elseif learner_name == "polynomial"
        coefficients = zeros(Float64, polynomial_degree + 1)
        return LearnerSetup(learner_name, nothing, nothing, coefficients,
                            nothing, seed, polynomial_degree, "polynomial")
    end
    error("learner must be nn or polynomial")
end

"""
Evaluate a neural network on a full state vector
    - nn is the Lux nn object, with parameters θ and state 'state'
    - u, or v1,v2 if in RD, is the state vector
    - reshape from N x 1 (or N x 2) to 1 x N (or 2 x N) so Lux sees this as its features x samples layout 
    - relatively cheap, since reshape and vec are views, though I could preallocate the memory for this...
"""
nn_values(u, nn, θ, state) = vec(first(Lux.apply(nn, reshape(u, 1, length(u)), θ, state)))
nn_values(v1, v2, nn, θ, state) = vec(first(Lux.apply(nn, hcat(v1, v2)', θ, state)))

"""
Evaluate polynomials with Horner's rule
"""
function polynomial_value_horner(u, coefficients)
    value = zero(u + coefficients[1]) # select a numeric type compatible with coefficients and u
    @inbounds for j in length(coefficients):-1:1
        value = value * u + coefficients[j]
    end
    value
end

"""Evaluate a polynomial (with horner's rule) on a state vector"""
polynomial_values(u, coefficients) = [polynomial_value_horner(ui, coefficients) for ui in u]
