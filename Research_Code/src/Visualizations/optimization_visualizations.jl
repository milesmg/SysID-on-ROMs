include(joinpath(@__DIR__, "optimization_metadata_visualizations.jl"))

using LinearAlgebra
using SparseArrays
using Random
using Serialization
using Printf
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


"""Return the final flat learned-parameter vector from a saved history."""
function final_theta(run_dir::AbstractString)
    snapshot = last(load_parameter_history(run_dir))
    return hasproperty(snapshot, :θ) ? snapshot.θ : snapshot.coefficients
end


### ADJUSTED: Add saved polynomial model helpers for learned nonlinearity visualization.
"""Return the saved learned nonlinearity type."""
function saved_model_type(data)
    if hasproperty(data, :model_type)
        return lowercase(string(data.model_type))
    elseif hasproperty(data, :learner)
        return lowercase(string(data.learner))
    end
    return "nn"
end


"""Return the final saved polynomial coefficients."""
function final_polynomial_coefficients(run_dir::AbstractString)
    snapshot = last(load_parameter_history(run_dir))
    return hasproperty(snapshot, :coefficients) ? snapshot.coefficients : snapshot.θ
end


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

"""Return the saved spatial dimension, defaulting old runs to 1D."""
saved_dimension(data) = hasproperty(data, :dimension) ? Int(data.dimension) : 1

"""Return saved reference output times, reconstructing them for compact old HPO metadata."""
function reference_saved_times(params)
    if hasproperty(params, :reference_saved_times)
        return params.reference_saved_times
    elseif hasproperty(params, :reference_save_count)
        return collect(LinRange(params.tspan[1], params.tspan[2], Int(params.reference_save_count)))
    else
        return params.t_obs
    end
end

"""Return the reference Euler step size when it was stored separately from `saveat`."""
reference_dt(params) = hasproperty(params, :reference_dt) ? params.reference_dt : nothing

"""Return the saved reference algorithm, defaulting compact HPO metadata to Euler."""
reference_algorithm(params) = hasproperty(params, :reference_algorithm) ? params.reference_algorithm : "Euler"

"""Return the saved Allen-Cahn reaction coefficient."""
saved_k(params) = hasproperty(params, :k) ? params.k : params.reference_parameters.k



"""Return the per-axis grid count for a saved 1D or flattened 2D run."""
function saved_grid_N(data)
    if hasproperty(data, :grid_N)
        return Int(data.grid_N)
    elseif hasproperty(data, :state_shape)
        return Int(first(data.state_shape))
    elseif saved_dimension(data) == 1
        return Int(data.N)
    else
        return round(Int, sqrt(data.N))
    end
end


"""Return the coordinate vectors saved for plotting a 1D or 2D state."""
function saved_grid_axes(data)
    N = saved_grid_N(data)
    x = hasproperty(data, :x) ? data.x : data.Δx .* collect(1:N)
    y = saved_dimension(data) == 1 ? nothing : (hasproperty(data, :y) && !isnothing(data.y) ? data.y : copy(x))
    return (; x, y)
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

"""Apply the flattened 2D Laplacian with homogeneous Dirichlet boundary conditions."""
function viz_lap2d!(du, u, N::Integer, invΔx2)
    @inbounds for j in 1:N, i in 1:N
        idx = i + (j - 1) * N
        value = -4u[idx]
        i > 1 && (value += u[idx - 1])
        i < N && (value += u[idx + 1])
        j > 1 && (value += u[idx - N])
        j < N && (value += u[idx + N])
        du[idx] = value * invΔx2
    end
    return nothing
end


"""Apply a 1D or flattened 2D Laplacian for visualization solves."""
function viz_lap_ac!(du, u, N::Integer, dimension, invΔx2)
    if Int(dimension) == 1
        viz_lap1d!(du, u, invΔx2)
    else
        viz_lap2d!(du, u, N, invΔx2)
    end
    return nothing
end


"""Construct the flattened 2D Allen-Cahn sparse diffusion matrix."""
function viz_lap2d_matrix(N, ε2, Δx)
    L1 = viz_lap1d_matrix(N, one(ε2), Δx)
    I_N = sparse(I, N, N)
    return ε2 * (kron(I_N, L1) + kron(L1, I_N))
end


"""Construct the 1D or flattened 2D Allen-Cahn sparse diffusion matrix."""
function viz_lap_ac_matrix(N, ε2, Δx, dimension)
    return Int(dimension) == 1 ? viz_lap1d_matrix(N, ε2, Δx) : viz_lap2d_matrix(N, ε2, Δx)
end


"""True Allen-Cahn RHS used for reference solves."""
function viz_rhs_ac!(du, u, p, t)
    (; ε2, k, Δx) = p
    dimension = hasproperty(p, :dimension) ? p.dimension : 1
    N = hasproperty(p, :N) ? p.N : (Int(dimension) == 1 ? length(u) : round(Int, sqrt(length(u))))
    viz_lap_ac!(du, u, N, dimension, 1 / Δx^2)
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


### ADJUSTED: Evaluate saved polynomial learned nonlinearities from coefficient snapshots.
"""Evaluate a learned polynomial nonlinearity on scalar states."""
function learned_polynomial_values(u, coefficients)
    values = similar(collect(u), promote_type(eltype(collect(u)), eltype(coefficients)))
    @inbounds for i in eachindex(values)
        value = zero(values[i] + first(coefficients))
        for j in length(coefficients):-1:1
            value = value * u[i] + coefficients[j]
        end
        values[i] = value
    end
    return values
end


"""Evaluate the true Allen-Cahn nonlinearity `-k(u^3-u)`."""
true_function_values(u, k) = .-k .* (u .^ 3 .- u)


"""Solve the true Allen-Cahn trajectory using saved reference solver settings."""
function solve_true_trajectory(u₀, tspan, p, save_times, reference_algorithm::AbstractString; dt=nothing)
    prob = ODEProblem(viz_rhs_ac!, u₀, tspan, p)
    alg = algorithm_from_name(reference_algorithm)
    if occursin("Euler", reference_algorithm)
        return solve(prob, alg; dt=something(dt, saved_time_step(save_times)), saveat=save_times)
    end
    return solve(prob, alg; saveat=save_times)
end


"""Convert an ODE solution's vector states into an `N x time` matrix."""
solution_matrix(sol) = hcat(sol.u...)


"""Solve the final learned FOM neural trajectory from a saved FOM run."""
function solve_fom_learned_trajectory(run_dir::AbstractString)
    params = load_fom_run_params(run_dir)
    θ_flat = final_theta(run_dir)
    dimension = saved_dimension(params)
    N = saved_grid_N(params)
    if saved_model_type(params) == "polynomial"
        ### ADJUSTED: Reconstruct learned FOM trajectories from polynomial coefficients when saved.
        p = ComponentVector(ε2=params.ε2, Δx=params.Δx, θ=θ_flat)
        rhs! = (du, u, p, t) -> begin
            viz_lap_ac!(du, u, N, dimension, 1 / p.Δx^2)
            f = learned_polynomial_values(u, p.θ)
            @inbounds @simd for i in eachindex(du)
                du[i] = p.ε2 * du[i] + f[i]
            end
            nothing
        end
        prob = ODEProblem(rhs!, params.u₀, params.tspan, p)
        alg = algorithm_from_name(params.ode_algorithm)
        return solve(prob, alg; saveat=reference_saved_times(params))
    end

    (; nn, state, re) = saved_network(params.h, params.seed)
    θ = re(θ_flat)
    p = ComponentVector(ε2=params.ε2, Δx=params.Δx, θ=θ)

    rhs! = (du, u, p, t) -> begin
        viz_lap_ac!(du, u, N, dimension, 1 / p.Δx^2)
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
    return solve(prob, alg; saveat=reference_saved_times(params))
end


"""Solve the true FOM reference trajectory from saved FOM metadata."""
function solve_fom_true_trajectory(run_dir::AbstractString)
    params = load_fom_run_params(run_dir)
    p = hasproperty(params, :reference_parameters) ?
        params.reference_parameters :
        (; ε2=params.ε2, k=hasproperty(params, :k) ? params.k : 1.0, Δx=params.Δx, N=saved_grid_N(params), dimension=saved_dimension(params))
    save_times = reference_saved_times(params)
    return solve_true_trajectory(params.u₀, params.tspan, p, save_times, reference_algorithm(params); dt=reference_dt(params))
end


"""Reconstruct ROM operators from saved modes and scalar PDE parameters."""
function reconstruct_rom_operators(rom_data)
    A = viz_lap_ac_matrix(saved_grid_N(rom_data), rom_data.ε2, rom_data.Δx, saved_dimension(rom_data))
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
    ops = reconstruct_rom_operators(rom_data)
    if saved_model_type(rom_data) == "polynomial"
        ### ADJUSTED: Reconstruct learned ROM trajectories from polynomial coefficients when saved.
        p = ComponentVector(Atilde=ops.Atilde, Up=ops.Up, B=ops.B, θ=θ_flat)
        ũ₀ = ops.U' * rom_data.u₀
        rhs! = (dũ, ũ, p, t) -> begin
            z = p.Up * ũ
            dũ .= p.Atilde * ũ + p.B * learned_polynomial_values(z, p.θ)
            nothing
        end
        prob = ODEProblem(rhs!, ũ₀, rom_data.tspan, p)
        sol = solve(prob, algorithm_from_name(ode_algorithm); saveat=reference_saved_times(rom_data))
        return (; t=sol.t, u=ops.U * hcat(sol.u...))
    end

    (; nn, state, re) = saved_network(rom_data.h, rom_data.seed)
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
    sol = solve(prob, algorithm_from_name(ode_algorithm); saveat=reference_saved_times(rom_data))
    return (; t=sol.t, u=ops.U * hcat(sol.u...))
end


"""Solve the true FOM reference trajectory from saved ROM metadata."""
function solve_rom_true_trajectory(run_dir::AbstractString)
    rom_data = load_rom_data(run_dir)
    metadata = read_metadata_values(run_dir)
    reference_algorithm = strip(get(metadata, "reference_algorithm", "\"Euler\""), ['"'])
    p = (; ε2=rom_data.ε2, k=rom_data.k, Δx=rom_data.Δx, N=saved_grid_N(rom_data), dimension=saved_dimension(rom_data))
    return solve_true_trajectory(rom_data.u₀, rom_data.tspan, p, reference_saved_times(rom_data), reference_algorithm; dt=reference_dt(rom_data))
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


"""Create and save an animated GIF comparing flattened 2D true and learned trajectories."""
function save_overlay_2d_trajectory_gif(x, y, t, true_u, learned_u; path::AbstractString, title::AbstractString, max_frames=120, fps=15, clims=nothing)
    mkpath(dirname(path))
    N = length(x)
    time_count = min(length(t), size(true_u, 2), size(learned_u, 2))
    frame_count = min(max_frames, time_count)
    frame_ids = unique(round.(Int, LinRange(1, time_count, frame_count)))
    anim = @animate for j in frame_ids
        true_frame = reshape(true_u[:, j], N, N)
        learned_frame = reshape(learned_u[:, j], N, N)
        error_frame = learned_frame .- true_frame
        p_true = heatmap(x, y, true_frame; clims, aspect_ratio=:equal, xlabel="x", ylabel="y", title="true", colorbar=false)
        p_learned = heatmap(x, y, learned_frame; clims, aspect_ratio=:equal, xlabel="x", ylabel="y", title="learned", colorbar=false)
        p_error = heatmap(x, y, error_frame; aspect_ratio=:equal, xlabel="x", ylabel="y", title="learned - true", colorbar=true)
        plot(p_true, p_learned, p_error; layout=(1, 3), size=(1200, 360), plot_title="$title, t = $(@sprintf("%.3f", t[j]))")
    end
    gif(anim, path; fps)
    return path
end


"""Display a saved GIF in a notebook."""
function display_gif(path::AbstractString)
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
    axes = saved_grid_axes(params)
    overlay_path = saved_dimension(params) == 1 ?
        save_overlay_trajectory_gif(axes.x, true_sol.t, true_u, learned_u; path=joinpath(out_dir, "overlaid_fom_trajectory.gif"), title="FOM trajectory", max_frames, fps, ylims) :
        save_overlay_2d_trajectory_gif(axes.x, axes.y, true_sol.t, true_u, learned_u; path=joinpath(out_dir, "overlaid_fom_trajectory_2d.gif"), title="FOM trajectory", max_frames, fps, clims=ylims)
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
    ylims = extrema(vcat(vec(true_u), vec(learned_u)))
    out_dir = joinpath(run_dir, "visualizations")
    axes = saved_grid_axes(rom_data)
    overlay_path = saved_dimension(rom_data) == 1 ?
        save_overlay_trajectory_gif(axes.x, true_sol.t, true_u, learned_u; path=joinpath(out_dir, "overlaid_rom_trajectory.gif"), title="ROM trajectory", max_frames, fps, ylims) :
        save_overlay_2d_trajectory_gif(axes.x, axes.y, true_sol.t, true_u, learned_u; path=joinpath(out_dir, "overlaid_rom_trajectory_2d.gif"), title="ROM trajectory", max_frames, fps, clims=ylims)
    if display_gifs
        display_gif(overlay_path)
    end
    return (; overlay_path, true_solution=true_sol, learned_solution=learned_sol)
end


"""Return a plot of saved ROM spatial modes and nonlinear/function modes."""
function plot_rom_modes(run_dir::AbstractString; n_modes=6)
    rom_data = load_rom_data(run_dir)
    axes = saved_grid_axes(rom_data)
    n_state = min(n_modes, size(rom_data.spatial_modes, 2))
    n_function = min(n_modes, size(rom_data.deim_modes, 2))

    if saved_dimension(rom_data) == 2
        N = saved_grid_N(rom_data)
        plots = Any[]
        for j in 1:n_state
            push!(plots, heatmap(axes.x, axes.y, reshape(rom_data.spatial_modes[:, j], N, N); aspect_ratio=:equal, title="POD mode $j", xlabel="x", ylabel="y", colorbar=false))
        end
        for j in 1:n_function
            push!(plots, heatmap(axes.x, axes.y, reshape(rom_data.deim_modes[:, j], N, N); aspect_ratio=:equal, title="DEIM mode $j", xlabel="x", ylabel="y", colorbar=false))
        end
        return plot(plots...; layout=(2, max(n_state, n_function)), size=(260 * max(n_state, n_function), 520))
    end

    p_state = plot(xlabel="x", ylabel="mode value", title="Spatial POD modes")
    for j in 1:n_state
        plot!(p_state, axes.x, rom_data.spatial_modes[:, j]; label="mode $j")
    end

    p_function = plot(xlabel="x", ylabel="mode value", title="Function / DEIM modes")
    for j in 1:n_function
        plot!(p_function, axes.x, rom_data.deim_modes[:, j]; label="mode $j")
    end

    p = plot(p_state, p_function; layout=(1, 2), size=(1000, 400))
    return p
end


"""Return a plot of the learned nonlinearity against the true Allen-Cahn nonlinearity."""
function plot_learned_function(run_dir::AbstractString; u_min=-1.2, u_max=1.2, n_points=400)
    θ_flat = final_theta(run_dir)
    u = collect(LinRange(u_min, u_max, n_points))
    if isfile(joinpath(run_dir, "rom_data.jls"))
        data = load_rom_data(run_dir)
    else
        data = load_fom_run_params(run_dir)
    end
    k = saved_k(data)
    p = plot(u, true_function_values(u, k); label="true -k(u^3-u)", xlabel="u", ylabel="f(u)", title="True vs learned nonlinearity")
    if saved_model_type(data) == "polynomial"
        ### ADJUSTED: Plot saved polynomial learned nonlinearities without rebuilding a Lux network.
        degree = hasproperty(data, :polynomial_degree) ? data.polynomial_degree : length(θ_flat) - 1
        plot!(p, u, learned_polynomial_values(u, θ_flat); label="learned polynomial degree $degree")
    else
        plot!(p, u, learned_function_values(u, θ_flat, data.h, data.seed); label="learned NN")
    end
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
    rom_trajectory_gifs(run_dir; max_frames, fps, display_gifs=true)
    return nothing
end


"""Display saved ROM spatial POD modes and function/DEIM modes."""
function visualize_ROM_modes(run_dir::AbstractString; n_modes=6)
    display(plot_rom_modes(run_dir; n_modes))
    return nothing
end


"""Print saved ROM singular-value capture ratios."""
visualize_ROM_singular_values(run_dir::AbstractString; max_modes=20) =
    print_singular_value_capture_table(run_dir; max_modes)


"""Display learned ROM nonlinearity against the true Allen-Cahn nonlinearity."""
function visualize_ROM_learned_function(run_dir::AbstractString; u_min=-1.2, u_max=1.2, n_points=400)
    display(plot_learned_function(run_dir; u_min, u_max, n_points))
    return nothing
end


"""Display saved ROM loss history over training iterations."""
function visualize_ROM_loss(run_dir::AbstractString; use_log=true)
    display(plot_loss_history(run_dir; use_log))
    return nothing
end


"""Solve and display an overlaid true/learned trajectory GIF for a saved FOM run."""
function visualize_FOM_trajectories(run_dir::AbstractString; max_frames=120, fps=15)
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
    display(plot_learned_function(run_dir; u_min, u_max, n_points))
    return nothing
end


"""Display saved FOM loss history over training iterations."""
function visualize_FOM_loss(run_dir::AbstractString; use_log=true)
    display(plot_loss_history(run_dir; use_log))
    return nothing
end


"""Display all ROM visualizations for a saved Data directory."""
function visualize_ROM(run_dir::AbstractString; max_frames=120, fps=15, n_modes=6)
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
    visualize_FOM_metadata(run_dir)
    visualize_FOM_trajectories(run_dir; max_frames, fps)
    visualize_FOM_modes(run_dir)
    visualize_FOM_learned_function(run_dir)
    visualize_FOM_loss(run_dir)
    return nothing
end
