"""Build a `RunConfig` for an interactive ROM-stability experiment."""
function stability_run_config(equation::AbstractString; N=64, L=1.0, tfinal=1.0, dimension=1,
                              boundary_condition=nothing, parameters=EquationParameters(), seed=1)
    spec = equation_spec(String(equation))
    boundary = isnothing(boundary_condition) ? spec.default_boundary_condition : String(boundary_condition)
    RunConfig(N, L, tfinal, 10, 8, seed, dimension, boundary, "polynomial", 3, 0.5, "default", parameters)
end

"""Materialize a named or explicit initial state for a ROM-stability experiment."""
function stability_initial_state(spec::EquationSpec, config::RunConfig, grid::Grid; initial_condition="default", initial_state=nothing)
    isnothing(initial_state) ? materialize_initial_condition(spec, grid, initial_condition, config) : Float64.(initial_state)
end

"""Build and solve the fixed-nonlinearity Allen-Cahn ROM used for a stability comparison."""
function ac_stability_rom(reference::ReferenceData, grid::Grid, parameters::EquationParameters)
    frames = hcat(reference.solution.u...)
    U, _ = pod_modes(frames, parameters.r)
    V, _ = pod_modes(ac_fixed_values(frames, parameters.k), parameters.m)
    points = deim_indices(V)
    A = laplacian_matrix(grid; scale=parameters.ε2)
    B = deim_projection(V, points, U)
    prob = ODEProblem((du, a, _, t) -> (du .= U' * A * U * a + B * ac_fixed_values(U[points, :] * a, parameters.k)),
        U' * reference.initial_state, reference.tspan)
    solution = solve(prob, TRBDF2(); saveat=reference.times)
    (; solution, frames=hcat((U * state for state in solution.u)...))
end

"""Build and solve the fixed-nonlinearity Cahn-Hilliard ROM used for a stability comparison."""
function ch_stability_rom(reference::ReferenceData, parameters::EquationParameters)
    rom = ch_rom(reference.operator, hcat(reference.solution.u...), parameters.r, parameters.m,
        parameters.ε2, parameters.sigma, reference.mean_state)
    prob = ODEProblem((du, a, _, t) -> (du .= rom.linear_operator * a + rom.nonlinear_projection *
        ch_fixed_values(rom.mean_state .+ rom.sampled_state * a)),
        rom.state_modes' * (reference.initial_state .- rom.mean_state), reference.tspan)
    solution = solve(prob, TRBDF2(autodiff=AutoFiniteDiff()); saveat=reference.times)
    (; solution, frames=hcat((rom.mean_state .+ rom.state_modes * state for state in solution.u)...))
end

"""Build and solve the fixed-nonlinearity reaction-diffusion ROM used for a stability comparison."""
function rd_stability_rom(reference::ReferenceData, parameters::EquationParameters)
    rom = rd_rom(reference.operator, hcat(reference.solution.u...), parameters.r, parameters.m,
        parameters.D1, parameters.D2, parameters.forced_deim_split)
    prob = ODEProblem((du, a, _, t) -> begin
            v1, v2 = rom.sampled_state * a, something(rom.sampled_state_2) * a
            sampled = [rom.components[index] == 1 ? rd_s1(v1[index], v2[index]) : rd_s2(v1[index], v2[index])
                       for index in eachindex(rom.components)]
            du .= rom.linear_operator * a + rom.nonlinear_projection * sampled
        end, rom.state_modes' * reference.initial_state, reference.tspan)
    solution = solve(prob, TRBDF2(autodiff=AutoFiniteDiff()); saveat=reference.times)
    (; solution, frames=hcat((rom.state_modes * state for state in solution.u)...))
end

"""Run a full reference solve and a fixed-nonlinearity ROM solve for AC, CH, or RD stability inspection."""
function run_rom_stability(equation::AbstractString; N=64, L=1.0, tfinal=1.0, dimension=1,
                           boundary_condition=nothing, parameters=EquationParameters(), seed=1,
                           initial_condition="default", initial_state=nothing, reference_dt_factor=0.5)
    config = stability_run_config(equation; N, L, tfinal, dimension, boundary_condition, parameters, seed)
    config = RunConfig(config.N, config.L, config.tfinal, config.N_obs, config.h, config.seed, config.dimension,
        config.boundary_condition, config.learner, config.polynomial_degree, reference_dt_factor,
        initial_condition, config.parameters)
    spec = equation_spec(String(equation))
    grid = spatial_grid(config.N, config.L, config.dimension, config.boundary_condition)
    u₀ = stability_initial_state(spec, config, grid; initial_condition, initial_state)
    reference = spec.reference(config, grid, u₀)
    rom = spec.name == "ac" ? ac_stability_rom(reference, grid, parameters) :
        spec.name == "ch" ? ch_stability_rom(reference, parameters) : rd_stability_rom(reference, parameters)
    reference_trajectory = (; run_dir=joinpath(VISUALIZATION_ROOT, "Untracked", "Tests"), times=reference.times,
        frames=hcat(reference.solution.u...), grid, equation=spec.name, mode=:reference)
    rom_trajectory = (; run_dir=reference_trajectory.run_dir, times=reference.times, frames=rom.frames,
        grid, equation=spec.name, mode=:rom)
    (; config, grid, reference, rom, reference_trajectory, rom_trajectory)
end

"""Render a FOM-versus-ROM stability GIF with fixed color limits over time."""
function save_rom_stability_gif(stability; title="FOM versus ROM stability", xlabel="x", ylabel="state",
                                legend_labels=("FOM", "ROM"), colorbar=true, fps=8, max_frames=60,
                                output_dir=joinpath(VISUALIZATION_ROOT, "Untracked", "Tests", "outputs"),
                                name="rom_stability.gif")
    mkpath(output_dir)
    trajectories = (; reference=stability.reference_trajectory, learned=stability.rom_trajectory)
    frame_ids = unique(round.(Int, LinRange(1, length(stability.reference.times), min(max_frames, length(stability.reference.times)))))
    color_limits = stability.config.parameters.D1 == 0.0 ? trajectory_color_limits(trajectories, 1) :
        (trajectory_color_limits(trajectories, 1), trajectory_color_limits(trajectories, 2))
    animation = @animate for frame_index in frame_ids
        plot_trajectory_comparison(trajectories; frame_index, title, xlabel, ylabel, legend_labels,
            colorbar, color_limits, save=false)
    end
    path = joinpath(output_dir, name)
    gif(animation, path; fps)
    path
end
