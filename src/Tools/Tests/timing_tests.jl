# ### ADJUSTED: Load the existing training stack when timing tools are included from a notebook.
if !isdefined(@__MODULE__, :EquationSpec)
    const TIMING_TEST_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
    for file in ("bootstrap.jl", "types.jl", "grids.jl", "laplacian.jl", "initial_conditions.jl",
                 "learners.jl", "losses.jl", "reduction.jl", "saving.jl", "cli.jl", "variable_windows.jl")
        include(joinpath(TIMING_TEST_ROOT, "src", "Core", file))
    end
    include(joinpath(TIMING_TEST_ROOT, "src", "Equations", "allen_cahn.jl"))
    include(joinpath(TIMING_TEST_ROOT, "src", "Core", "pipeline.jl"))
end

"""Print a timing-suite progress message with the local wall-clock date and time."""
### ADJUSTED: Make long timing sweeps observable without inspecting the Julia process externally.
timing_log(message) = println("[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ", message)

"""Format one parameter combination for timestamped timing-suite progress messages."""
timing_case_label(combo) = "N=$(combo.N), h=$(combo.h), window_N_obs=$(combo.window_N_obs), window_T=$(combo.window_T), r=$(combo.r), m=$(combo.m), rom=$(combo.rom), loss_space=$(combo.loss_space)"

"""Return the local directory used to persist one named timing test."""
### ADJUSTED: Keep each timing sweep's schedule and measurements together in an ignored local directory.
function timing_results_directory(test_name)
    name = String(test_name)
    isempty(name) && error("test_name cannot be empty")
    basename(name) == name && name != "." && name != ".." || error("test_name must be one directory name")
    directory = joinpath(TIMING_TEST_ROOT, "Untracked", "Tests", "timing_test_results", name)
    mkpath(directory)
    directory
end

"""Return the unique test name shared by every combination in a timing schedule."""
function timing_test_name(schedule)
    isempty(schedule) && error("timing schedule is empty")
    names = unique(getproperty.(schedule, :test_name))
    length(names) == 1 || error("all timing combinations must have the same test_name")
    only(names)
end

"""Write one pipe-delimited table without timestamps for a schedule or timing result file."""
function write_timing_table(path, rows; columns=propertynames(first(rows)))
    open(path, "w") do io
        println(io, join(string.(columns), " | "))
        for row in rows
            println(io, join(string.(getproperty(row, column) for column in columns), " | "))
        end
    end
    path
end

"""Persist a schedule's exact parameter combinations as `cases.txt`."""
function save_timing_cases(schedule)
    directory = timing_results_directory(timing_test_name(schedule))
    path = write_timing_table(joinpath(directory, "cases.txt"), schedule)
    timing_log("Saved $(length(schedule)) timing case(s) to $path.")
    path
end

"""Persist one timing metric as readable text and a Julia-serialized result table."""
function save_timing_results(schedule, results, metric)
    directory = timing_results_directory(timing_test_name(schedule))
    columns = (:metric, :N, :h, :window_N_obs, :window_T, :r, :m, :rom, :loss_space,
        :mean_seconds, :std_seconds, :min_seconds, :max_seconds, :mean_bytes)
    text_path = write_timing_table(joinpath(directory, "$(metric)_results.txt"), results; columns)
    serialized_path = joinpath(directory, "$(metric)_results.jls")
    serialize(serialized_path, results)
    timing_log("Saved $(metric) timing results to $text_path and $serialized_path.")
    results
end

"""Persist all metric tables and the complete structured output of one timing suite."""
function save_timing_suite(results)
    directory = timing_results_directory(timing_test_name(results.schedule))
    open(joinpath(directory, "results.txt"), "w") do io
        for (metric, metric_results) in pairs((forward=results.forward, loss=results.loss, adam_step=results.adam_step))
            println(io, "# $metric")
            columns = (:metric, :N, :h, :window_N_obs, :window_T, :r, :m, :rom, :loss_space,
                :mean_seconds, :std_seconds, :min_seconds, :max_seconds, :mean_bytes)
            println(io, join(string.(columns), " | "))
            for row in metric_results
                println(io, join(string.(getproperty(row, column) for column in columns), " | "))
            end
            println(io)
        end
    end
    serialized_path = joinpath(directory, "results.jls")
    serialize(serialized_path, results)
    timing_log("Saved complete timing suite to $(joinpath(directory, "results.txt")) and $serialized_path.")
    results
end

"""Print one readable row per timing parameter combination."""
function print_timing_schedule(schedule)
    isempty(schedule) && return timing_log("Timing schedule is empty.")
    columns = propertynames(first(schedule))
    timing_log("Timing schedule with $(length(schedule)) combination(s):")
    timing_log(join(string.(columns), " | "))
    for combo in schedule
        timing_log(join(string.(getproperty(combo, column) for column in columns), " | "))
    end
    schedule
end

"""Build, print, and save a named Cartesian-product 2D Allen--Cahn timing schedule."""
### ADJUSTED: Provide a compact, notebook-friendly sweep over optimization-step cost drivers.
function timing_schedule(test_name; N=[32], h=[8], window_N_obs=[10], window_T=[2.0], r=[10], m=[10],
                         rom=[false], loss_space=["FULL"], tfinal=nothing, seed=1)
    timing_results_directory(test_name)
    schedule = NamedTuple[]
    for values in Iterators.product(N, h, window_N_obs, window_T, r, m, rom, loss_space)
        N_value, h_value, n_obs_value, window_T_value, r_value, m_value, rom_value, loss_space_value = values
        resolved_loss_space = uppercase(String(loss_space_value))
        resolved_tfinal = isnothing(tfinal) ? Float64(window_T_value) : Float64(tfinal)
        resolved_loss_space in ("FULL", "REDUCED") || error("loss_space must be FULL or REDUCED")
        !rom_value && resolved_loss_space == "REDUCED" && error("REDUCED loss_space requires rom=true")
        window_T_value <= resolved_tfinal || error("window_T cannot exceed tfinal")
        push!(schedule, (; test_name=String(test_name), N=Int(N_value), h=Int(h_value), window_N_obs=Int(n_obs_value),
            window_T=Float64(window_T_value), r=Int(r_value), m=Int(m_value), rom=Bool(rom_value),
            loss_space=resolved_loss_space, tfinal=resolved_tfinal, seed=Int(seed)))
    end
    print_timing_schedule(schedule)
    save_timing_cases(schedule)
    schedule
end

"""Build the reference, optional ROM, deterministic window, and one-step objective for a timing combination."""
function prepare_timing_case(combo)
    timing_log("Preparing $(timing_case_label(combo)): building reference trajectory.")
    parameters = EquationParameters(; ε2=1e-5, k=1.0, r=combo.r, m=combo.m)
    config = RunConfig(combo.N, 1.0, combo.tfinal, combo.window_N_obs, combo.h, combo.seed, 2,
        "periodic", "nn", 3, 0.5, "2d random noise", parameters)
    spec = equation_spec("ac")
    grid = spatial_grid(config.N, config.L, config.dimension, config.boundary_condition)
    reference = spec.reference(config, grid, materialize_initial_condition(spec, grid, "default", config))
    timing_log("Preparing $(timing_case_label(combo)): building $(combo.rom ? "ROM" : "FOM") training problem.")
    prepared = spec.model(combo.rom ? :rom : :fom, config, grid, reference, initialize_learner(spec, config))
    window_spec = make_window_spec(0.0, combo.window_T, combo.window_N_obs; stage=1, iteration=1, policy="beginning")
    window = materialize_window(reference.solution, window_spec, prepared.project)
    theta, rebuild = Optimisers.destructure(prepared.initial_parameters.θ)
    parameters = prepared.rebuild_parameters(prepared.initial_parameters, rebuild, theta)
    algorithm = TRBDF2(autodiff=AutoFiniteDiff())
    sensitivity = GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP())
    objective = Optimization.OptimizationFunction(
        (trial_theta, _) -> solve_window_loss(window, prepared.problem,
            prepared.rebuild_parameters(prepared.initial_parameters, rebuild, trial_theta), algorithm, sensitivity,
            "mean", prepared.Δmeasure, prepared.reconstruct, prepared.project, combo.loss_space),
        Optimization.AutoZygote())
    optimization_problem = Optimization.OptimizationProblem(objective, copy(theta), nothing)
    timing_log("Preparing $(timing_case_label(combo)): complete; timed work has not started.")
    (; combo, prepared, window, theta, parameters, algorithm, sensitivity, optimization_problem)
end

"""Prepare all timing cases before measurement, excluding reference and ROM construction from reported times."""
function prepare_timing_cases(schedule)
    timing_log("Preparing $(length(schedule)) timing case(s); reference and ROM construction are excluded from measurements.")
    [prepare_timing_case(combo) for combo in schedule]
end

"""Solve one timing window without loss or differentiation."""
function timing_forward_solution(case)
    problem = remake(case.prepared.problem; u0=case.window.model_u0,
        tspan=(case.window.spec.t_start, case.window.spec.t_end), p=case.parameters)
    solve(problem, case.algorithm; saveat=case.window.spec.t_obs)
end

"""Evaluate the configured full or reduced trajectory loss from a previously solved window."""
function timing_window_loss(case, solution)
    # ### ADJUSTED: Time only the loss by reusing the projections prepared with each timing window.
    if case.combo.loss_space == "REDUCED" && !isnothing(case.prepared.project)
        weighted_solution_loss(solution.u, case.window.model_reference_observations, case.prepared.Δmeasure, "mean")
    else
        model_states = [case.prepared.reconstruct(state, case.prepared.problem, case.parameters) for state in solution.u]
        weighted_solution_loss(model_states, case.window.reference_observations, case.prepared.Δmeasure, "mean")
    end
end

"""Warm one labeled workload, then collect repeated `@timed` measurements without compilation time."""
function timing_summary(workload; repeats=5, label="workload")
    repeats > 0 || error("repeats must be positive")
    timing_log("$label: warm-up started; this unmeasured run compiles the exact workload.")
    workload()
    timing_log("$label: warm-up complete; collecting $repeats timed sample(s).")
    seconds, bytes = Float64[], Float64[]
    for sample in 1:repeats
        timing_log("$label: timed sample $sample/$repeats started.")
        GC.gc()
        measurement = @timed workload()
        push!(seconds, measurement.time)
        push!(bytes, measurement.bytes)
        timing_log("$label: timed sample $sample/$repeats complete in $(round(measurement.time; sigdigits=4)) s, $(measurement.bytes) bytes.")
    end
    summary = (; mean_seconds=mean(seconds), std_seconds=length(seconds) == 1 ? 0.0 : std(seconds),
       min_seconds=minimum(seconds), max_seconds=maximum(seconds), mean_bytes=mean(bytes),
       samples_seconds=seconds, samples_bytes=bytes)
    timing_log("$label: complete; mean = $(round(summary.mean_seconds; sigdigits=4)) s, std = $(round(summary.std_seconds; sigdigits=4)) s.")
    summary
end

"""Time and save repeated forward solves for every prepared case after one unmeasured warm-up."""
function time_forward_solves(schedule; cases=prepare_timing_cases(schedule), repeats=5)
    save_timing_cases(schedule)
    results = NamedTuple[]
    for case in cases
        label = "Forward solve ($(timing_case_label(case.combo)))"
        push!(results, merge(case.combo, (metric="forward",), timing_summary(() -> timing_forward_solution(case); repeats, label)))
    end
    save_timing_results(schedule, results, "forward")
    results
end

"""Time and save repeated loss evaluations for every case, excluding the preceding forward solve."""
function time_losses(schedule; cases=prepare_timing_cases(schedule), repeats=5)
    save_timing_cases(schedule)
    results = NamedTuple[]
    for case in cases
        label = "Loss evaluation ($(timing_case_label(case.combo)))"
        timing_log("$label: solving one unmeasured forward window for loss inputs.")
        solution = timing_forward_solution(case)
        timing_log("$label: forward input is ready; timing loss only.")
        push!(results, merge(case.combo, (metric="loss",), timing_summary(() -> timing_window_loss(case, solution); repeats, label)))
    end
    save_timing_results(schedule, results, "loss")
    results
end

"""Time and save one complete forward, reverse, and Adam update for every case after warm-up."""
function time_adam_steps(schedule; cases=prepare_timing_cases(schedule), repeats=5, eta=1e-3)
    save_timing_cases(schedule)
    optimizer = OptimizationOptimisers.Adam(eta, (0.9, 0.99))
    results = NamedTuple[]
    for case in cases
        label = "Adam step ($(timing_case_label(case.combo)))"
        push!(results, merge(case.combo, (metric="adam_step",),
            timing_summary(() -> Optimization.solve(case.optimization_problem, optimizer; maxiters=1); repeats, label)))
    end
    save_timing_results(schedule, results, "adam_step")
    results
end

"""Prepare cases once, save, and return forward, loss, and complete Adam-step timing tables."""
function time_optimization_steps(schedule; repeats=5, eta=1e-3)
    timing_log("Starting forward, loss, and Adam-step timing suite.")
    cases = prepare_timing_cases(schedule)
    results = (; schedule, forward=time_forward_solves(schedule; cases, repeats), loss=time_losses(schedule; cases, repeats),
        adam_step=time_adam_steps(schedule; cases, repeats, eta))
    save_timing_suite(results)
    timing_log("Timing suite complete.")
    results
end

"""Print a compact table of timing means, variability, allocations, and timing parameters."""
function print_timing_table(results)
    isempty(results) && return timing_log("Timing results are empty.")
    preferred_columns = (:metric, :N, :h, :window_N_obs, :window_T, :r, :m, :rom, :loss_space,
        :mean_seconds, :std_seconds, :min_seconds, :max_seconds, :mean_bytes)
    columns = filter(column -> column in propertynames(first(results)), preferred_columns)
    timing_log("Timing result table with $(length(results)) row(s):")
    timing_log(join(string.(columns), " | "))
    for result in results
        values = [column in (:mean_seconds, :std_seconds, :min_seconds, :max_seconds) ?
            round(getproperty(result, column); sigdigits=4) : getproperty(result, column) for column in columns]
        timing_log(join(string.(values), " | "))
    end
    results
end
