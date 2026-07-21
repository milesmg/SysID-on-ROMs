### This is the big pipeline file. No fun. 


"""
Return equation specifications based on which equation we're dealing with. EquationSpec struct is def'd in the types.jl file. 
"""
function equation_spec(name)::EquationSpec
    if name == "ac"
        return ac_spec()
    elseif name == "ch"
        return ch_spec()
    elseif name == "rd"
        return rd_spec()
    end
    error("equation must be ac, ch, or rd")
end

"""
Build a RunConfig struct (see types.jl), which stores many optimization params, based on a give EquationSpec struct (types.jl), 
    which stores equation-specific parameters; also taking into account params passed in via cli
Args:
    - options: a dict with defaults and params passed in via cli
    - spec: an EquationSpec struct, built by the equation_spec function
"""
function run_configuration(options, spec::EquationSpec)::RunConfig
    N = get_int(options, "N", spec.default_N)
    L = get_float(options, "L", 1.0)
    tfinal = get_float(options, "tfinal", spec.default_tfinal)
    N_obs = get_int(options, "N-obs", 10)
    h = get_int(options, "h", 8)
    seed = get_int(options, "seed", 1)
    dimension = get_int(options, "dimension", spec.default_dimension)
    boundary_condition = get_string(options, "boundary-condition", spec.default_boundary_condition)
    learner = lowercase(get_string(options, "learner", "nn"))
    polynomial_degree = get_int(options, "polynomial-degree", 3)
    reference_dt_factor = get_float(options, "reference-dt-factor", 0.5)
    initial_condition = get_string(options, "initial-condition", "default")
    RunConfig(N, L, tfinal, N_obs, h, seed, dimension, boundary_condition, learner,
              polynomial_degree, reference_dt_factor,
              initial_condition, 
              # note that the parse_parameters function is equation specific, as parametrizations depend on eqns
              spec.parse_parameters(options))
end

"""Takes equation information from EquationSpec and run configuration from RunConfig and builds a LearnerSetup struct,
    which contains the relevant learner type, input size (eg. 2-dim vs 1-dim for R-D vs A-C), degree, etc. """
initialize_learner(spec::EquationSpec, config::RunConfig)::LearnerSetup =
    build_learner(config.learner, spec.input_dim, config.h, config.seed, config.polynomial_degree)

"""This is the function that does everything. I'll comment it line-by-line
Args: 
    - mode: either :fom or :rom
    - args: these are the command line arguments passed in as a vector of strings. 
"""
function run_training(mode::Symbol, args)
    options = parse_cli(args) # parse arguments passed in via command line and turn them into a nice dictionary
    name = lowercase(get_string(options, "equation", "ac")) # which equation are we modeling?
    spec = equation_spec(name) # build the specifications based on that equation; eg. get the proper parameters, and find the default settings
    config = run_configuration(options, spec) # combine our defaults with the options dictionary to configure a run
    grid = spatial_grid(config.N, config.L, config.dimension, config.boundary_condition) # build our spatial discretization
    u₀ = materialize_initial_condition(spec, grid, config.initial_condition, config) # build an initial condition
    reference = spec.reference(config, grid, u₀) # build reference trajectory, the true u
    init = initialize_learner(spec, config) # combine our settings with equation-specific requirements to build our machine learning model, whether it be a NN or a polynomial
    prepared = spec.model(mode, config, grid, reference, init) # this is a PreparedTraining struct; it's designed to plug-and-play into an optimizer. It has all the FOM/ROM specific data, projectors, initial conditions, params, etc.
    training = parse_training_options(options, config.tfinal, config.N_obs, config.seed) # this takes the options dictionary, and a few other data, and gets all the relevant training parametrizations (see cli.jl)
    run_name = get_string(options, "run-name", timestamp_run_name("$(uppercase(string(mode)))_$(uppercase(spec.name))_hpc")) # default name is very generic and will likely collide, which is fine
    assert_run_name_available(run_name) # if run name is not available, break immediately; you don't want to run and have nowhere to save. Note that I moved this function into saving.jl
    setup_runtime() # set threading; see cli.jl
    output = run_variable_window_stages(prepared, training; log_name="run_$(mode)_$(spec.name)") # this is doing the very heavy lifting. See variable_windows.jl, where the whole optimization is actually run. 
    if mode == :fom # pretty clear what's going on here
        save_dir = save_fom_run(prepared, output, run_name)
    else
        save_dir = save_rom_run(prepared, output, run_name)
    end
    println("Saved $(uppercase(string(mode))) output to: ", save_dir)
    println("Final full-trajectory loss: ", output.final_full_trajectory_loss)
end
