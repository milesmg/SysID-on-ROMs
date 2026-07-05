### ADJUSTED: Add reusable FOM/ROM optimization visualization helpers for saved Data runs.
include(joinpath(@__DIR__, "optimization_metadata_visualizations.jl"))

using LinearAlgebra
using SparseArrays
using Random
using Serialization
using Printf
### ADJUSTED: Embed generated GIFs directly in notebook HTML displays.
using Base64
using ComponentArrays
using OrdinaryDiffEq
using OrdinaryDiffEqSDIRK
using OrdinaryDiffEqLowOrderRK
using Lux
using Functors
using OptimizationOptimisers
using Plots


"""Load the saved parameter history for a FOM or ROM run."""
load_parameter_history(run_dir::AbstractString) = deserialize(joinpath(run_dir, "parameter_history.jls"))


"""Load saved FOM run parameters."""
load_fom_run_params(run_dir::AbstractString) = deserialize(joinpath(run_dir, "run_params.jls"))


"""Load saved ROM data."""
load_rom_data(run_dir::AbstractString) = deserialize(joinpath(run_dir, "rom_data.jls"))


"""Return the final flat neural-network parameter vector from a saved history."""
final_theta(run_dir::AbstractString) = last(load_parameter_history(run_dir)).θ


"""Construct a solver algorithm from a saved metadata solver name."""
function algorithm_from_name(name::AbstractString)
    if occursin("Euler", name)
        return Euler()
    elseif occursin("TRBDF2", name)
        return TRBDF2()
    elseif occursin("Tsit5", name)
        return Tsit5()
    else
        @warn "Unknown algorithm string; falling back to TRBDF2()" name
        return TRBDF2()
    end
end


"""Return an Euler timestep compatible with saved reference times."""
function saved_time_step(save_times)
    length(save_times) > 1 || return nothing
    return minimum(diff(collect(save_times)))
end


"""Apply the 1D Laplacian with homogeneous Dirichlet boundary conditions."""
function viz_lap1d!(du, u, invΔx2)
    @inbounds du[1] = (u[2] - 2u[1]) * invΔx2
    @inbounds @simd for i in 2:length(u)-1
        du[i] = (u[i-1] - 2u[i] + u[i+1]) * invΔx2
    end
    @inbounds du[end] = (u[end-1] - 2u[end]) * invΔx2
    return nothing
end


"""Construct the 1D Allen-Cahn sparse diffusion matrix."""
function viz_lap1d_matrix(N, ε2, Δx)
    scale = ε2 / Δx^2
    return spdiagm(
        -1 => fill(scale, N - 1),
         0 => fill(-2scale, N),
         1 => fill(scale, N - 1),
    )
end


"""True Allen-Cahn RHS used for reference solves."""
function viz_rhs_ac!(du, u, p, t)
    (; ε2, k, Δx) = p
    viz_lap1d!(du, u, 1 / Δx^2)
    @inbounds @simd for i in eachindex(du)
        du[i] = ε2 * du[i] - k * (u[i]^3 - u[i])
    end
    return nothing
end


"""Build the same 1-`h`-`h`-1 tanh Lux network used in training."""
function saved_network(h::Integer, seed::Integer)
    rng = MersenneTwister(seed)
    nn = Chain(Dense(1 => h, tanh), Dense(h => h, tanh), Dense(h => 1))
    ps₀, state = Lux.setup(rng, nn)
    ps₀ = fmap(x -> Float64.(x), ps₀)
    _, re = Optimisers.destructure(ps₀)
    return (; nn, state, re)
end


"""Evaluate a saved neural nonlinearity on a vector of scalar states."""
function learned_function_values(u, θ_flat, h::Integer, seed::Integer)
    (; nn, state, re) = saved_network(h, seed)
    θ = re(θ_flat)
    x = reshape(collect(u), 1, length(u))
    y, _ = Lux.apply(nn, x, θ, state)
    return vec(y)
end


"""Evaluate the true Allen-Cahn nonlinearity `-k(u^3-u)`."""
true_function_values(u, k) = .-k .* (u .^ 3 .- u)


"""Solve the true Allen-Cahn trajectory using saved reference solver settings."""
function solve_true_trajectory(u₀, tspan, p, save_times, reference_algorithm::AbstractString)
    prob = ODEProblem(viz_rhs_ac!, u₀, tspan, p)
    alg = algorithm_from_name(reference_algorithm)
    if occursin("Euler", reference_algorithm)
        return solve(prob, alg; dt=saved_time_step(save_times), saveat=save_times)
    end
    return solve(prob, alg; saveat=save_times)
end


"""Convert an ODE solution's vector states into an `N x time` matrix."""
solution_matrix(sol) = hcat(sol.u...)


"""Solve the final learned FOM neural trajectory from a saved FOM run."""
function solve_fom_learned_trajectory(run_dir::AbstractString)
    params = load_fom_run_params(run_dir)
    θ_flat = final_theta(run_dir)
    (; nn, state, re) = saved_network(params.h, params.seed)
    θ = re(θ_flat)
    p = ComponentVector(ε2=params.ε2, Δx=params.Δx, θ=θ)

    rhs! = (du, u, p, t) -> begin
        viz_lap1d!(du, u, 1 / p.Δx^2)
        x = reshape(u, 1, length(u))
        y, _ = Lux.apply(nn, x, p.θ, state)
        fy = vec(y)
        @inbounds @simd for i in eachindex(du)
            du[i] = p.ε2 * du[i] + fy[i]
        end
        nothing
    end

    prob = ODEProblem(rhs!, params.u₀, params.tspan, p)
    alg = algorithm_from_name(params.ode_algorithm)
    return solve(prob, alg; saveat=params.reference_saved_times)
end


"""Solve the true FOM reference trajectory from saved FOM metadata."""
function solve_fom_true_trajectory(run_dir::AbstractString)
    params = load_fom_run_params(run_dir)
    p = params.reference_parameters
    return solve_true_trajectory(params.u₀, params.tspan, p, params.reference_saved_times, params.reference_algorithm)
end


"""Reconstruct ROM operators from saved modes and scalar PDE parameters."""
function reconstruct_rom_operators(rom_data)
    A = viz_lap1d_matrix(rom_data.N, rom_data.ε2, rom_data.Δx)
    U = rom_data.spatial_modes
    V = rom_data.deim_modes
    p = rom_data.deim_indices
    return (;
        U,
        V,
        Atilde=U' * (A * U),
        p,
        Up=U[p, :],
        B=(V[p, :]' \ (V' * U))',
    )
end


"""Solve the final learned ROM trajectory and reconstruct full states."""
function solve_rom_learned_trajectory(run_dir::AbstractString)
    rom_data = load_rom_data(run_dir)
    metadata = read_metadata_values(run_dir)
    ode_algorithm = strip(get(metadata, "ode_algorithm", "\"TRBDF2\""), ['"'])
    θ_flat = final_theta(run_dir)
    (; nn, state, re) = saved_network(rom_data.h, rom_data.seed)
    ops = reconstruct_rom_operators(rom_data)
    θ = re(θ_flat)
    p = ComponentVector(Atilde=ops.Atilde, Up=ops.Up, B=ops.B, θ=θ)
    ũ₀ = ops.U' * rom_data.u₀

    rhs! = (dũ, ũ, p, t) -> begin
        z = p.Up * ũ
        x = reshape(z, 1, length(z))
        y, _ = Lux.apply(nn, x, p.θ, state)
        dũ .= p.Atilde * ũ + p.B * vec(y)
        nothing
    end

    prob = ODEProblem(rhs!, ũ₀, rom_data.tspan, p)
    sol = solve(prob, algorithm_from_name(ode_algorithm); saveat=rom_data.reference_saved_times)
    return (; t=sol.t, u=ops.U * hcat(sol.u...))
end


"""Solve the true FOM reference trajectory from saved ROM metadata."""
function solve_rom_true_trajectory(run_dir::AbstractString)
    rom_data = load_rom_data(run_dir)
    metadata = read_metadata_values(run_dir)
    reference_algorithm = strip(get(metadata, "reference_algorithm", "\"Euler\""), ['"'])
    p = ComponentVector(ε2=rom_data.ε2, k=rom_data.k, Δx=rom_data.Δx)
    return solve_true_trajectory(rom_data.u₀, rom_data.tspan, p, rom_data.reference_saved_times, reference_algorithm)
end


"""Create and save an animated GIF for a trajectory matrix."""
function save_trajectory_gif(x, t, u; path::AbstractString, title::AbstractString, max_frames=120, fps=15, ylims=nothing)
    mkpath(dirname(path))
    frame_count = min(max_frames, length(t))
    frame_ids = unique(round.(Int, LinRange(1, length(t), frame_count)))
    anim = @animate for j in frame_ids
        plot(
            x,
            u[:, j];
            ylim=ylims,
            xlabel="x",
            ylabel="u",
            title="$title, t = $(@sprintf("%.3f", t[j]))",
            legend=false,
        )
    end
    gif(anim, path; fps)
    return path
end


"""Create and save an animated GIF with true and learned trajectories overlaid."""
function save_overlay_trajectory_gif(x, t, true_u, learned_u; path::AbstractString, title::AbstractString, max_frames=120, fps=15, ylims=nothing)
    ### ADJUSTED: Overlay true and learned trajectories in each animation frame.
    mkpath(dirname(path))
    time_count = min(length(t), size(true_u, 2), size(learned_u, 2))
    frame_count = min(max_frames, time_count)
    frame_ids = unique(round.(Int, LinRange(1, time_count, frame_count)))
    anim = @animate for j in frame_ids
        plot(
            x,
            true_u[:, j];
            ylim=ylims,
            xlabel="x",
            ylabel="u",
            title="$title, t = $(@sprintf("%.3f", t[j]))",
            label="true",
            color=:blue,
            linewidth=2,
        )
        plot!(
            x,
            learned_u[:, j];
            label="learned",
            color=:orange,
            linewidth=2,
            linestyle=:dash,
        )
    end
    gif(anim, path; fps)
    return path
end


"""Display a saved GIF in a notebook."""
function display_gif(path::AbstractString)
    ### ADJUSTED: Use a data URI so notebooks do not need access to local file URLs.
    encoded = base64encode(read(path))
    display("text/html", HTML("""<img src="data:image/gif;base64,$encoded" style="max-width: 100%;">"""))
    return nothing
end


"""Save and optionally display an overlaid true/learned FOM trajectory GIF."""
function fom_trajectory_gifs(run_dir::AbstractString; max_frames=120, fps=15, display_gifs=true)
    params = load_fom_run_params(run_dir)
    true_sol = solve_fom_true_trajectory(run_dir)
    learned_sol = solve_fom_learned_trajectory(run_dir)
    true_u = solution_matrix(true_sol)
    learned_u = solution_matrix(learned_sol)
    ylims = extrema(vcat(vec(true_u), vec(learned_u)))
    out_dir = joinpath(run_dir, "visualizations")
    ### ADJUSTED: Save one overlaid FOM trajectory GIF instead of separate true/learned GIFs.
    overlay_path = save_overlay_trajectory_gif(params.x, true_sol.t, true_u, learned_u; path=joinpath(out_dir, "overlaid_fom_trajectory.gif"), title="FOM trajectory", max_frames, fps, ylims)
    if display_gifs
        display_gif(overlay_path)
    end
    return (; overlay_path, true_solution=true_sol, learned_solution=learned_sol)
end


"""Save and optionally display an overlaid true/learned ROM trajectory GIF."""
function rom_trajectory_gifs(run_dir::AbstractString; max_frames=120, fps=15, display_gifs=true)
    rom_data = load_rom_data(run_dir)
    true_sol = solve_rom_true_trajectory(run_dir)
    learned_sol = solve_rom_learned_trajectory(run_dir)
    true_u = solution_matrix(true_sol)
    learned_u = learned_sol.u
    x = rom_data.Δx .* collect(1:rom_data.N)
    ylims = extrema(vcat(vec(true_u), vec(learned_u)))
    out_dir = joinpath(run_dir, "visualizations")
    ### ADJUSTED: Save one overlaid ROM trajectory GIF instead of separate true/learned GIFs.
    overlay_path = save_overlay_trajectory_gif(x, true_sol.t, true_u, learned_u; path=joinpath(out_dir, "overlaid_rom_trajectory.gif"), title="ROM trajectory", max_frames, fps, ylims)
    if display_gifs
        display_gif(overlay_path)
    end
    return (; overlay_path, true_solution=true_sol, learned_solution=learned_sol)
end


"""Return a plot of saved ROM spatial modes and nonlinear/function modes."""
function plot_rom_modes(run_dir::AbstractString; n_modes=6)
    rom_data = load_rom_data(run_dir)
    x = rom_data.Δx .* collect(1:rom_data.N)
    n_state = min(n_modes, size(rom_data.spatial_modes, 2))
    n_function = min(n_modes, size(rom_data.deim_modes, 2))

    p_state = plot(xlabel="x", ylabel="mode value", title="Spatial POD modes")
    for j in 1:n_state
        plot!(p_state, x, rom_data.spatial_modes[:, j]; label="mode $j")
    end

    p_function = plot(xlabel="x", ylabel="mode value", title="Function / DEIM modes")
    for j in 1:n_function
        plot!(p_function, x, rom_data.deim_modes[:, j]; label="mode $j")
    end

    p = plot(p_state, p_function; layout=(1, 2), size=(1000, 400))
    ### ADJUSTED: Return the plot without displaying; visualize_* functions display once.
    return p
end


"""Return a plot of the learned neural nonlinearity against the true Allen-Cahn nonlinearity."""
function plot_learned_function(run_dir::AbstractString; u_min=-1.2, u_max=1.2, n_points=400)
    θ_flat = final_theta(run_dir)
    u = collect(LinRange(u_min, u_max, n_points))
    if isfile(joinpath(run_dir, "rom_data.jls"))
        data = load_rom_data(run_dir)
        h, seed, k = data.h, data.seed, data.k
    else
        data = load_fom_run_params(run_dir)
        h, seed, k = data.h, data.seed, data.reference_parameters.k
    end
    p = plot(u, true_function_values(u, k); label="true -k(u^3-u)", xlabel="u", ylabel="f(u)", title="True vs learned nonlinearity")
    plot!(p, u, learned_function_values(u, θ_flat, h, seed); label="learned NN")
    ### ADJUSTED: Return the plot without displaying; visualize_* functions display once.
    return p
end


"""Return a plot of saved loss over training iterations."""
function plot_loss_history(run_dir::AbstractString; use_log=true)
    history = load_parameter_history(run_dir)
    iterations = getproperty.(history, :iteration)
    losses = getproperty.(history, :loss)
    yscale = use_log && all(>(0), losses) ? :log10 : :identity
    p = plot(iterations, losses; xlabel="iteration", ylabel="loss", yscale, marker=:circle, label="parameter snapshots", title="Loss history")

    evaluation_path = joinpath(run_dir, "evaluation_history.jls")
    if isfile(evaluation_path)
        evaluations = deserialize(evaluation_path)
        eval_iterations = getproperty.(evaluations, :iteration)
        eval_losses = getproperty.(evaluations, :loss)
        plot!(p, eval_iterations, eval_losses; label="evaluation history", alpha=0.45)
    end

    ### ADJUSTED: Return the plot without displaying; visualize_* functions display once.
    return p
end


"""Print singular-value capture ratios for ROM state and function modes."""
function print_singular_value_capture_table(run_dir::AbstractString; max_modes=nothing)
    rom_data = load_rom_data(run_dir)
    state_s = rom_data.state_singular_values
    function_s = rom_data.nonlinear_singular_values
    n = isnothing(max_modes) ? max(length(state_s), length(function_s)) : min(max_modes, max(length(state_s), length(function_s)))

    println("mode | state capture | state squared capture | function capture | function squared capture")
    println("-----|---------------|-----------------------|------------------|--------------------------")
    state_total = sum(state_s)
    state_sq_total = sum(abs2, state_s)
    function_total = sum(function_s)
    function_sq_total = sum(abs2, function_s)
    for j in 1:n
        state_capture = j <= length(state_s) ? sum(@view state_s[1:j]) / state_total : NaN
        state_sq_capture = j <= length(state_s) ? sum(abs2, @view state_s[1:j]) / state_sq_total : NaN
        function_capture = j <= length(function_s) ? sum(@view function_s[1:j]) / function_total : NaN
        function_sq_capture = j <= length(function_s) ? sum(abs2, @view function_s[1:j]) / function_sq_total : NaN
        @printf("%4d | %13.6f | %21.6f | %16.6f | %24.6f\n", j, state_capture, state_sq_capture, function_capture, function_sq_capture)
    end
    return nothing
end


"""Solve and display an overlaid true/learned trajectory GIF for a saved ROM run."""
function visualize_ROM_trajectories(run_dir::AbstractString; max_frames=120, fps=15)
    ### ADJUSTED: Display GIFs once and suppress the large returned solution tuple.
    rom_trajectory_gifs(run_dir; max_frames, fps, display_gifs=true)
    return nothing
end


"""Display saved ROM spatial POD modes and function/DEIM modes."""
function visualize_ROM_modes(run_dir::AbstractString; n_modes=6)
    ### ADJUSTED: Display the returned plot once and suppress notebook auto-display.
    display(plot_rom_modes(run_dir; n_modes))
    return nothing
end


"""Print saved ROM singular-value capture ratios."""
visualize_ROM_singular_values(run_dir::AbstractString; max_modes=20) =
    print_singular_value_capture_table(run_dir; max_modes)


"""Display learned ROM nonlinearity against the true Allen-Cahn nonlinearity."""
function visualize_ROM_learned_function(run_dir::AbstractString; u_min=-1.2, u_max=1.2, n_points=400)
    ### ADJUSTED: Display the returned plot once and suppress notebook auto-display.
    display(plot_learned_function(run_dir; u_min, u_max, n_points))
    return nothing
end


"""Display saved ROM loss history over training iterations."""
function visualize_ROM_loss(run_dir::AbstractString; use_log=true)
    ### ADJUSTED: Display the returned plot once and suppress notebook auto-display.
    display(plot_loss_history(run_dir; use_log))
    return nothing
end


"""Solve and display an overlaid true/learned trajectory GIF for a saved FOM run."""
function visualize_FOM_trajectories(run_dir::AbstractString; max_frames=120, fps=15)
    ### ADJUSTED: Display GIFs once and suppress the large returned solution tuple.
    fom_trajectory_gifs(run_dir; max_frames, fps, display_gifs=true)
    return nothing
end


"""State that saved FOM runs do not contain POD/DEIM modes."""
function visualize_FOM_modes(run_dir::AbstractString)
    println("FOM runs do not save POD/DEIM spatial or function modes.")
    return nothing
end


"""Display learned FOM nonlinearity against the true Allen-Cahn nonlinearity."""
function visualize_FOM_learned_function(run_dir::AbstractString; u_min=-1.2, u_max=1.2, n_points=400)
    ### ADJUSTED: Display the returned plot once and suppress notebook auto-display.
    display(plot_learned_function(run_dir; u_min, u_max, n_points))
    return nothing
end


"""Display saved FOM loss history over training iterations."""
function visualize_FOM_loss(run_dir::AbstractString; use_log=true)
    ### ADJUSTED: Display the returned plot once and suppress notebook auto-display.
    display(plot_loss_history(run_dir; use_log))
    return nothing
end


"""Display all ROM visualizations for a saved Data directory."""
function visualize_ROM(run_dir::AbstractString; max_frames=120, fps=15, n_modes=6)
    ### ADJUSTED: Display each visualization once and return nothing.
    visualize_ROM_metadata(run_dir)
    visualize_ROM_trajectories(run_dir; max_frames, fps)
    visualize_ROM_modes(run_dir; n_modes)
    visualize_ROM_singular_values(run_dir; max_modes=n_modes)
    visualize_ROM_learned_function(run_dir)
    visualize_ROM_loss(run_dir)
    return nothing
end


"""Display all FOM visualizations for a saved Data directory."""
function visualize_FOM(run_dir::AbstractString; max_frames=120, fps=15)
    ### ADJUSTED: Display each visualization once and return nothing.
    visualize_FOM_metadata(run_dir)
    visualize_FOM_trajectories(run_dir; max_frames, fps)
    visualize_FOM_modes(run_dir)
    visualize_FOM_learned_function(run_dir)
    visualize_FOM_loss(run_dir)
    return nothing
end
