# Build tooling to save data

"""Gets the path for the Data directory"""
function optimization_data_root()
    ### ADJUSTED: Save all run output in the repository-level Data directory.
    normpath(joinpath(@__DIR__, "..", "..", "Data"))
end

"""
Actually save the data to the relevant files
Args:
    - run_name: the name of the run, which becomes the name of the directory (directory must not exist prior to running!)
    - parameter_history: history of saved parameters throughout the optimization
    - metadata: exactly what it sounds like; (name,value) pairs of metadata, eg. (N,64)
    - extra_serialized: other stuff, like parameter_history, that we want stored as a .jls file 
"""
function save_optimization_run(run_name::AbstractString; parameter_history, metadata,
                               run_params=nothing, extra_serialized=NamedTuple())
    data_root = optimization_data_root()
    run_directory = assert_run_name_available(run_name; data_root)
    mkpath(data_root)
    mkdir(run_directory)
    serialize(joinpath(run_directory, "parameter_history.jls"), parameter_history)
    !isnothing(run_params) && serialize(joinpath(run_directory, "run_params.jls"), run_params)
    for (name, value) in pairs(extra_serialized)
        serialize(joinpath(run_directory, string(name, ".jls")), value)
    end
    open(joinpath(run_directory, "metadata.txt"), "w") do io
        for (name, value) in pairs(metadata)
            print(io, name, " = ")
            show(io, value)
            println(io)
        end
    end
    run_directory
end

"""
Saves the windows and validation losses
Args:
- run_directory: the path to save
- output: a TrainingOutput struct; see types.jl
"""
### ADJUSTED: Save the validation history once rather than emitting a duplicate evaluation file.
function save_training_histories(run_directory, output::TrainingOutput)
    for (name, value) in (("window_history", output.window_history),
                          ("validation_history", output.validation_history))
        serialize(joinpath(run_directory, name * ".jls"), value)
    end
    run_directory
end

"""
Builds an unstructured named tuple of the run parameters to be saved
Args: 
- prepared: a PreparedTraining struct (see types.jl) which contains all of the relevant training parameters, including ROM/FOM projectors, initial condits, etc 
- output: a TrainingOutput struct containing the results of the PreparedTraining
"""
function serialized_run_parameters(prepared::PreparedTraining, output::TrainingOutput)
    config, grid, learner, reference = prepared.config, prepared.grid, prepared.learner, prepared.reference
    parameters = config.parameters
    equation = prepared.equation_name
    if equation == "ac"
        ε2, k = parameters.ε2, parameters.k
        sigma, mean_c, D1, D2 = nothing, nothing, nothing, nothing
        state_shape = grid.state_shape
        state_components, learned_component, reference_reactions = nothing, nothing, nothing
    elseif equation == "rd"
        ### ADJUSTED: Initialize every serialized equation parameter for reaction-diffusion runs.
        ε2, k, sigma, mean_c, D1, D2 = nothing, nothing, nothing, nothing, parameters.D1, parameters.D2
        state_shape = (2, grid.state_shape...)
        state_components, learned_component = ("v1", "v2"), "s2"
        reference_reactions = "s1=v1-v1^3-v2-0.005; s2=10*(v1-v2)"
    elseif equation == "ch"
        ### ADJUSTED: Initialize every serialized equation parameter for Cahn-Hilliard runs.
        ε2, k, sigma, mean_c, D1, D2 = parameters.ε2, nothing, parameters.sigma, reference.mean_state, nothing, nothing
        state_shape = grid.state_shape
        state_components, learned_component, reference_reactions = nothing, nothing, nothing
    end

    y = nothing
    if !isnothing(grid.y)
        y = copy(grid.y)
    end
    network_architecture = nothing
    if learner.kind == "nn"
        input_dimension = 1
        if equation == "rd"
            input_dimension = 2
        end
        network_architecture = (input_dimension, learner.h, learner.h, 1)
    end
    polynomial_coefficient_order = nothing
    polynomial_initial_coefficients, polynomial_final_coefficients = nothing, nothing
    if learner.kind == "polynomial"
        polynomial_coefficient_order = "ascending powers of u"
        if equation == "rd"
            polynomial_coefficient_order = "ascending total-degree monomials (i,j)"
        end
        polynomial_initial_coefficients = copy(learner.θ)
        polynomial_final_coefficients = copy(output.final_theta)
    end
    r, m = nothing, nothing
    if !isnothing(prepared.rom)
        r, m = size(prepared.rom.state_modes, 2), size(prepared.rom.nonlinear_modes, 2)
    end
    (; equation, N=config.N, L=config.L, ε2, k, sigma, mean_c, D1, D2,
       Δx=grid.Δx, Δmeasure=spatial_measure(grid), dimension=grid.dimension,
       boundary_condition=grid.boundary_condition, state_shape, x=copy(grid.x), y,
       tspan=reference.tspan, N_obs=config.N_obs, t_obs=copy(reference.times),
       u₀=copy(reference.initial_state), learner=learner.kind,
       state_components, learned_component, reference_reactions, h=learner.h,
       network_architecture, activation=learner.activation, polynomial_degree=learner.polynomial_degree,
       polynomial_coefficient_order, polynomial_initial_coefficients, polynomial_final_coefficients,
       seed=learner.seed, reference_dt=reference.Δt, reference_dt_factor=config.reference_dt_factor,
       reference_save_count=length(reference.times), initial_condition=config.initial_condition,
       variable_window_training=true, r, m)
end

"""
Builds an unstructured named tuple of the training parameters to be saved
Args: 
- prepared: a PreparedTraining struct (see types.jl) which contains all of the relevant training parameters, including ROM/FOM projectors, initial condits, etc 
- output: a TrainingOutput struct containing the results of the PreparedTraining
"""
function serialized_training_parameters(prepared::PreparedTraining, output::TrainingOutput)
    training = output.training
    η = copy(training.etas)
    if length(η) == 1
        η = η[1]
    end
    N_iter = sum(training.iterations)
    window_T = copy(training.window_T)
    if length(window_T) == 1
        window_T = window_T[1]
    end
    window_N_obs = copy(training.window_N_obs)
    if length(window_N_obs) == 1
        window_N_obs = window_N_obs[1]
    end
    window_start_policy = copy(training.window_start_policy)
    if length(window_start_policy) == 1
        window_start_policy = window_start_policy[1]
    end
    ### ADJUSTED: Preserve complete stage schedules alongside compact scalar metadata.
    (; ode_algorithm="TRBDF2(autodiff=AutoFiniteDiff())",
       sensitivity_algorithm="GaussAdjoint(MooncakeVJP)", optimizer="Adam variable-window staged",
       η, N_iter, η_schedule=copy(training.etas), N_iter_schedule=copy(training.iterations), β=training.beta,
       window_T, window_N_obs, window_start_policy,
       window_T_schedule=copy(training.window_T), window_N_obs_schedule=copy(training.window_N_obs),
       window_start_policy_schedule=copy(training.window_start_policy),
       loss_normalization=training.loss_normalization, window_seed=training.window_seed,
       validation_N_obs=prepared.reference.N_obs, warmup=training.warmup,
       save_frequency=training.save_frequency, print_frequency=training.print_frequency,
       final_training_loss=output.final_training_loss,
       final_full_trajectory_loss=output.final_full_trajectory_loss)
end

"""
Save data from a FOM run
Args: 
- prepared: a PreparedTraining struct (see types.jl) which contains all of the relevant training parameters, including ROM/FOM projectors, initial condits, etc 
- output: a TrainingOutput struct containing the results of the PreparedTraining
- run_name: the name of the run; already checked to make sure no collisions upon saving
"""
function save_fom_run(prepared::PreparedTraining, output::TrainingOutput, run_name::AbstractString)
    run_params = merge(serialized_run_parameters(prepared, output), serialized_training_parameters(prepared, output))
    metadata = merge(run_params, (;
        saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        julia_version=VERSION,
        initial_loss=first(output.validation_history).loss,
        final_loss=output.final_full_trajectory_loss,
        parameter_snapshots=length(output.parameter_history),
    ))
    run_directory = save_optimization_run(run_name;
        parameter_history=output.parameter_history, run_params, metadata)
    save_training_histories(run_directory, output)
end

"""
Save data from a ROM run
Args: 
- prepared: a PreparedTraining struct (see types.jl) which contains all of the relevant training parameters, including ROM/FOM projectors, initial condits, etc 
- output: a TrainingOutput struct containing the results of the PreparedTraining
- run_name: the name of the run; already checked to make sure no collisions upon saving
"""
function save_rom_run(prepared::PreparedTraining, output::TrainingOutput, run_name::AbstractString)
    run_params = merge(serialized_run_parameters(prepared, output), serialized_training_parameters(prepared, output))
    rom = something(prepared.rom)
    rom_data = merge(run_params, (; spatial_modes=rom.state_modes, deim_modes=rom.nonlinear_modes,
        deim_indices=rom.deim_indices, state_singular_values=rom.state_singular_values,
        nonlinear_singular_values=rom.nonlinear_singular_values,
        deim_components=rom.components, deim_spatial_indices=rom.spatial_indices))
    metadata = merge(run_params, (;
        saved_at=Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
        julia_version=VERSION,
        initial_loss=first(output.validation_history).loss,
        final_loss=output.final_full_trajectory_loss,
        parameter_snapshots=length(output.parameter_history),
    ))
    run_directory = save_optimization_run(run_name;
        parameter_history=output.parameter_history,
        metadata,
        extra_serialized=(; rom_data))
    save_training_histories(run_directory, output)
end


"""Fail if `Data/<run_name>` already exists."""
function assert_run_name_available(run_name::AbstractString; data_root=optimization_data_root())
    run_directory = joinpath(data_root, run_name)
    isdir(run_directory) && error("Run directory already exists: $run_directory")
    return run_directory
end
