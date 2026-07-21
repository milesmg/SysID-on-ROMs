using Test

### ADJUSTED: Test the top-level source tree directly.
const SRC_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
### ADJUSTED: Include the active variable-window implementation before testing its public functions.
for file in ("Core/bootstrap.jl", "Core/types.jl", "Core/grids.jl", "Core/laplacian.jl", "Core/initial_conditions.jl", "Core/learners.jl", "Core/losses.jl", "Core/reduction.jl", "Core/saving.jl", "Core/cli.jl", "Core/variable_windows.jl", "Equations/allen_cahn.jl", "Equations/cahn_hilliard.jl", "Equations/reaction_diffusion.jl", "Core/pipeline.jl")
    include(joinpath(SRC_ROOT, file))
end

@testset "shared core" begin
    periodic = spatial_grid(4, 1.0, 1, "periodic")
    neumann = spatial_grid(4, 1.0, 1, "neumann")
    dirichlet = spatial_grid(4, 1.0, 2, "dirichlet")
    @test spatial_length(dirichlet) == 16
    @test spatial_measure(dirichlet) == dirichlet.Δx^2
    @test_throws ErrorException spatial_grid(4, 1.0, 1, "homogeneous_dirichlet")
    @test_throws ErrorException spatial_grid(4, 1.0, 1, "homogeneous_neumann")
    @test laplacian_matrix(periodic) * ones(4) ≈ zeros(4)
    @test laplacian_matrix(neumann) * ones(4) ≈ zeros(4)
    @test size(laplacian_matrix(dirichlet)) == (16, 16)
    for grid in (periodic, neumann, dirichlet)
        u, du = collect(1.0:prod(grid.state_shape)), zeros(prod(grid.state_shape))
        laplacian!(du, u, grid)
        @test du ≈ laplacian_matrix(grid) * u
    end

    ch_grid = spatial_grid(5, 1.0, 2, "periodic")
    ch_u, ch_D = collect(1.0:25.0) ./ 25, -laplacian_matrix(ch_grid)
    ch_p = EquationParameters(; ε2=.01, sigma=.2, mean_c=.1)
    ch_prob = ch_problem(ch_u, (0.0, .1), ch_p, ch_grid, ch_D, (u, _) -> u .^ 3 .- u)
    ch_du = similar(ch_u)
    ch_prob.f(ch_du, ch_u, ch_p, 0.0)
    @test ch_du ≈ -ch_p.ε2 .* (ch_D * (ch_D * ch_u)) .- ch_D * (ch_u .^ 3 .- ch_u) .- ch_p.sigma .* (ch_u .- ch_p.mean_c)

    rd_grid = spatial_grid(5, 1.0, 2, "neumann")
    rd_n, rd_lap = spatial_length(rd_grid), laplacian_matrix(rd_grid)
    rd_u, rd_du = collect(1.0:2rd_n) ./ rd_n, zeros(2rd_n)
    rd_p = EquationParameters(; D1=.1, D2=.2)
    rd_rhs!(rd_du, rd_u, rd_p, rd_grid, rd_n, (v1, v2, _) -> rd_s2.(v1, v2))
    rd_v1, rd_v2 = @view(rd_u[1:rd_n]), @view(rd_u[rd_n+1:2rd_n])
    @test rd_du[1:rd_n] ≈ rd_p.D1 .* (rd_lap * rd_v1) .+ rd_s1.(rd_v1, rd_v2)
    @test rd_du[rd_n+1:2rd_n] ≈ rd_p.D2 .* (rd_lap * rd_v2) .+ rd_s2.(rd_v1, rd_v2)

    options = parse_cli(["--etas", "1e-3,5e-4", "--iters=2,3", "--warmup", "false"])
    training = parse_training_options(options, 1.0, 4, 42)
    @test training.etas == [1e-3, 5e-4]
    @test training.iterations == [2, 3]

    args = ((0.0, 1.0), [2, 1], [0.2, 0.5], [2, 4], ["random", "beginning"], 42)
    ### ADJUSTED: Derive the saved flat history from the stage-wise schedule returned by the active API.
    stages1 = build_window_schedule(args...)
    stages2 = build_window_schedule(args...)
    history1 = reduce(vcat, stages1)
    history2 = reduce(vcat, stages2)
    @test stages1 == stages2
    @test history1 == history2
    @test length(history1) == 3
    @test all(hasproperty(window, :iteration) for window in history1)

    model = [[1.0, 2.0], [2.0, 4.0]]
    reference = [[0.0, 1.0], [1.0, 3.0]]
    @test weighted_solution_loss(model, reference, .5, "sum") ≈ 1.0
    @test weighted_solution_loss(model, reference, .5, "mean") ≈ .5

    learner = build_learner("nn", 1, 2, 1, 3)
    @test length(nn_values([0.0, 1.0], learner.nn, learner.θ, learner.state)) == 2
    modes, singular_values = pod_modes([1.0 0.0; 0.0 1.0], 2)
    @test pod_capture(singular_values, 1) ≈ .5
    @test length(unique(deim_indices(modes))) == 2

    for (name, options) in (("ac", ["--N", "8", "--tfinal", "0.02", "--r", "1", "--m", "1"]), ("ch", ["--N", "8", "--tfinal", "0.02", "--r", "1", "--m", "1"]), ("rd", ["--N", "4", "--tfinal", "0.01", "--r", "1", "--m", "1"]))
        spec = equation_spec(name)
        @test spec.default_N > 0
        @test spec.default_tfinal > 0
        config = run_configuration(parse_cli(options), spec)
        grid = spatial_grid(config.N, config.L, config.dimension, config.boundary_condition)
        state = materialize_initial_condition(spec, grid, "default", config)
        ref = spec.reference(config, grid, state)
        @test length(ref.initial_state) == spatial_length(grid; fields=spec.fields)
        @test isfinite(sum(ref.solution.u[end]))
        ### ADJUSTED: Check run-parameter serialization for each equation and both FOM/ROM prepared paths.
        for mode in (:fom, :rom)
            prepared = spec.model(mode, config, grid, ref, initialize_learner(spec, config))
            @test hasproperty(prepared, :problem)
            training = TrainingConfig([1e-3], [1], [config.tfinal], [1], ["beginning"], "mean", 1, (0.9, 0.99), false, 1, 1)
            output = TrainingOutput(nothing, Float64[], training, TrainingSnapshot[], 0.0, 0.0, WindowSpec[], TrainingSnapshot[])
            @test serialized_run_parameters(prepared, output).N == config.N
        end
    end

    config = RunConfig(4, 1.0, 0.1, 2, 2, 1, 1, "dirichlet", "nn", 3, 0.5,
                       "default", EquationParameters(; ε2=0.01))
    grid = spatial_grid(config.N, config.L, config.dimension, config.boundary_condition)
    reference = ReferenceData(nothing, nothing, config.parameters, zeros(4), nothing, [0.0, 0.1], (0.0, 0.1), 0.01, 2, 0.0)
    learner = LearnerSetup("nn", nothing, nothing, [0.0], 2, 1, nothing, "tanh")
    ### ADJUSTED: Supply the equation name required by the current PreparedTraining contract.
    prepared = PreparedTraining(config, grid, learner, reference, :fom, nothing, nothing, nothing, nothing, nothing, grid.Δx, nothing, "ac")
    training = TrainingConfig([1e-3], [1], [0.1], [2], ["beginning"], "mean", 1, (0.9, 0.99), false, 1, 1)
    parameter_history = [TrainingSnapshot(0, 0, :parameter, [0.0], 1.0)]
    validation_history = [TrainingSnapshot(0, 0, :validation, nothing, 1.0)]
    output = TrainingOutput(nothing, [0.0], training, parameter_history, 1.0, 1.0, WindowSpec[], validation_history)
    training_params = serialized_training_parameters(prepared, output)
    @test training_params.η_schedule == training.etas
    @test training_params.final_full_trajectory_loss == output.final_full_trajectory_loss
    @test serialized_run_parameters(prepared, output).N == config.N
    history_dir = mktempdir()
    save_training_histories(history_dir, output)
    ### ADJUSTED: Persist only distinct histories.
    @test all(isfile(joinpath(history_dir, name)) for name in ("window_history.jls", "validation_history.jls"))

    ### ADJUSTED: Keep differentiated FOM/ROM smoke coverage available without making the default local suite depend on Mooncake's unstable IRTools path.
    if get(ENV, "RUN_DIFFERENTIATED_SMOKE", "false") == "true"
        for (name, options) in (("ac", ["--N", "8", "--tfinal", "0.01", "--r", "1", "--m", "1"]), ("ch", ["--N", "8", "--tfinal", "0.01", "--r", "1", "--m", "1"]), ("rd", ["--N", "4", "--tfinal", "0.01", "--r", "1", "--m", "1"]))
            spec = equation_spec(name)
            config = run_configuration(parse_cli(options), spec)
            grid = spatial_grid(config.N, config.L, config.dimension, config.boundary_condition)
            ref = spec.reference(config, grid, materialize_initial_condition(spec, grid, "default", config))
            training = TrainingConfig([1e-3], [1], [config.tfinal], [1], ["beginning"], "mean", 1, (0.9, 0.99), false, 1, 1)
            for mode in (:fom, :rom)
                prepared = spec.model(mode, config, grid, ref, initialize_learner(spec, config))
                output = run_variable_window_stages(prepared, training; log_name="typed_contract_smoke")
                @test output isa TrainingOutput
                @test isfinite(output.final_full_trajectory_loss)
                @test length(output.window_history) == 1
                @test length(output.validation_history) == 2
                @test all(snapshot.kind == :parameter && !isnothing(snapshot.θ) for snapshot in output.parameter_history)
                @test all(snapshot.kind == :validation && isnothing(snapshot.θ) for snapshot in output.validation_history)
                history_dir = mktempdir()
                save_training_histories(history_dir, output)
                ### ADJUSTED: Persist only distinct histories.
                @test all(isfile(joinpath(history_dir, name)) for name in ("window_history.jls", "validation_history.jls"))
                @test serialized_run_parameters(prepared, output).N == config.N
                @test serialized_training_parameters(prepared, output).η_schedule == training.etas
            end
        end
    end
end

println("PASS: shared core tests")
