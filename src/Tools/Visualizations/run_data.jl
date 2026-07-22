"""Load the serialized artifacts written for one FOM or ROM training run."""
function load_visualization_run(run_dir::AbstractString)
    parameter_history = deserialize(joinpath(run_dir, "parameter_history.jls"))
    mode = isfile(joinpath(run_dir, "rom_data.jls")) ? :rom : :fom
    data_name = mode == :rom ? "rom_data.jls" : "run_params.jls"
    (; run_dir=abspath(run_dir), mode, data=deserialize(joinpath(run_dir, data_name)), parameter_history)
end

"""Return one run directory or every immediate child run directory selected from `input_path`."""
function visualization_run_directories(input_path::AbstractString; dir_of_dirs=false, mode=:any)
    candidates = dir_of_dirs ? filter(isdir, readdir(input_path; join=true)) : [input_path]
    runs = filter(path -> isfile(joinpath(path, "parameter_history.jls")), candidates)
    filter(path -> mode == :any || load_visualization_run(path).mode == mode, sort(runs))
end

"""Return a named-tuple value, falling back to `default` for older saved runs."""
visualization_value(data, name::Symbol, default) = hasproperty(data, name) && !isnothing(getproperty(data, name)) ? getproperty(data, name) : default

"""Return a run equation name, inferring it from older serialized physical parameters when needed."""
### ADJUSTED: Infer AC, CH, or RD for legacy run artifacts that omit the equation field.
function visualization_equation(data)
    hasproperty(data, :equation) && return String(data.equation)
    !isnothing(visualization_value(data, :D1, nothing)) && return "rd"
    !isnothing(visualization_value(data, :k, nothing)) && return "ac"
    !isnothing(visualization_value(data, :sigma, nothing)) && return "ch"
    error("saved run metadata does not identify its equation")
end

"""Return the final trainable parameter vector saved for a current or legacy run."""
### ADJUSTED: Accept older parameter-history entries that predate the `kind` field.
function final_visualization_theta(run)
    snapshots = filter(snapshot -> hasproperty(snapshot, :θ) && !isnothing(snapshot.θ) &&
        (!hasproperty(snapshot, :kind) || snapshot.kind == :parameter), run.parameter_history)
    last(snapshots).θ
end

"""Rebuild the immutable `RunConfig` needed to replay one current or legacy saved run."""
function visualization_run_config(data)
    visualization_equation(data)
    parameters = EquationParameters(; ε2=visualization_value(data, :ε2, 0.0),
        k=visualization_value(data, :k, 0.0), sigma=visualization_value(data, :sigma, 0.0),
        mean_c=visualization_value(data, :mean_c, 0.0), D1=visualization_value(data, :D1, 0.0),
        D2=visualization_value(data, :D2, 0.0), r=visualization_value(data, :r, 0),
        m=visualization_value(data, :m_requested, visualization_value(data, :m, 0)),
        forced_deim_split=visualization_value(data, :forced_deim_split, false))
    # ### ADJUSTED: Legacy 2D artifacts store total state size in N and the per-axis count in grid_N.
    grid_N = visualization_value(data, :grid_N, data.N)
    RunConfig(grid_N, visualization_value(data, :L, 1.0), last(data.tspan),
        visualization_value(data, :N_obs, length(data.t_obs)),
        something(visualization_value(data, :h, nothing), 8), data.seed, data.dimension,
        data.boundary_condition, data.learner,
        something(visualization_value(data, :polynomial_degree, nothing), 3),
        visualization_value(data, :reference_dt_factor, 0.5),
        visualization_value(data, :initial_condition, "default"), parameters)
end

"""Build the reusable reference/model context for a saved FOM or ROM run without solving the learned trajectory."""
function prepare_visualization_run(run_dir::AbstractString)
    run = load_visualization_run(run_dir)
    config = visualization_run_config(run.data)
    spec = equation_spec(visualization_equation(run.data))
    grid = spatial_grid(config.N, config.L, config.dimension, config.boundary_condition)
    reference = spec.reference(config, grid, Float64.(run.data.u₀))
    learner = initialize_learner(spec, config)
    prepared = spec.model(run.mode, config, grid, reference, learner)
    θ, re = Optimisers.destructure(prepared.initial_parameters.θ)
    final_theta = final_visualization_theta(run)
    parameters = prepared.rebuild_parameters(prepared.initial_parameters, re, final_theta)
    (; run, config, spec, grid, reference, prepared, initial_theta=θ, final_theta, parameters)
end

"""Return the full-order reference frames and saved times for one prepared visualization run."""
function solve_reference_trajectory(context; times=context.reference.times)
    selected_times = collect(times)
    frames = hcat((context.reference.solution(time) for time in selected_times)...)
    (; run_dir=context.run.run_dir, times=selected_times, frames, grid=context.grid, equation=context.spec.name, mode=:reference)
end

"""Solve the final learned FOM or ROM and return reconstructed full-order frames at `times`."""
function solve_learned_trajectory(context; times=context.reference.times)
    selected_times = collect(times)
    solution = solve(remake(context.prepared.problem; p=context.parameters), TRBDF2(autodiff=AutoFiniteDiff()); saveat=selected_times)
    frames = hcat((context.prepared.reconstruct(state, context.prepared.problem, context.parameters) for state in solution.u)...)
    (; run_dir=context.run.run_dir, times=selected_times, frames, grid=context.grid, equation=context.spec.name, mode=context.run.mode)
end

"""Build plot-ready reference and learned frames once without retaining solver internals."""
### ADJUSTED: Return only reusable frames so notebook data caches do not retain large ODE solutions.
function solve_run_trajectories(run_dir::AbstractString; times=nothing)
    context = prepare_visualization_run(run_dir)
    selected_times = isnothing(times) ? context.reference.times : times
    reference = solve_reference_trajectory(context; times=selected_times)
    learned = solve_learned_trajectory(context; times=selected_times)
    frame_count = min(length(reference.times), length(learned.times))
    (; reference=(; reference..., times=reference.times[1:frame_count], frames=reference.frames[:, 1:frame_count]),
      learned=(; learned..., times=learned.times[1:frame_count], frames=learned.frames[:, 1:frame_count]))
end

"""Return the A-C or R-D local visualization-data directory for one equation."""
### ADJUSTED: Save notebook plots beside their visualization notebooks rather than inside run directories.
function visualization_output_directory(equation::AbstractString)
    visualization_group = lowercase(equation) == "rd" ? "R-D" : "A-C"
    output_dir = joinpath(VISUALIZATION_ROOT, "Untracked", "Visualize_results", visualization_group, "Data")
    mkpath(output_dir)
    output_dir
end

"""Save one static plot in the local A-C or R-D visualization-data directory and return its path."""
function save_run_plot(run_dir::AbstractString, plot_object, name::AbstractString; equation=nothing)
    resolved_equation = isnothing(equation) ? visualization_equation(load_visualization_run(run_dir).data) : String(equation)
    path = joinpath(visualization_output_directory(resolved_equation), name)
    savefig(plot_object, path)
    path
end

"""Compute the final learned-function L2 error stored or reconstructed for a saved run."""
### ADJUSTED: Fall back to reconstruction when legacy parameter snapshots have no saved error field.
function final_function_l2_error(run_dir::AbstractString; bounds=(-1.0, 1.0))
    context = prepare_visualization_run(run_dir)
    snapshot = last(filter(item -> hasproperty(item, :θ) && !isnothing(item.θ) &&
        (!hasproperty(item, :kind) || item.kind == :parameter), context.run.parameter_history))
    stored_error = hasproperty(snapshot, :learned_function_error) ? snapshot.learned_function_error : nothing
    !isnothing(stored_error) ? stored_error : learned_function_l2_error(context.prepared, context.parameters.θ, bounds)
end

"""Compute and print the final learned-function L2 error based only on a run directory path."""
function print_final_function_l2_error(run_dir::AbstractString; bounds=(-1.0, 1.0))
    error_value = final_function_l2_error(run_dir; bounds)
    println("$(basename(run_dir)): final learned-function L2 error = $error_value")
    error_value
end
