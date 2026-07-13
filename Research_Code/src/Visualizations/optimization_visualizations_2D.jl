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
using LaTeXStrings


const MAX_REFERENCE_SAVED_TIMES = 500


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

### ADJUSTED: Infer 2D from serialized state size for older 2D ROM/FOM runs missing dimension metadata.
"""Return the saved spatial dimension, defaulting old scalar-state runs to 1D."""
function saved_dimension(data, run_dir::AbstractString="")
    hasproperty(data, :dimension) && return Int(data.dimension)
    occursin("2d", lowercase(run_dir)) && return 2
    state_length = if hasproperty(data, :spatial_modes)
        size(data.spatial_modes, 1)
    elseif hasproperty(data, :u₀)
        length(data.u₀)
    else
        hasproperty(data, :N) ? Int(data.N) : 0
    end
    if hasproperty(data, :N) && Int(data.N)^2 == state_length
        return 2
    end
    return 1
end

### ADJUSTED: Infer boundary conditions from serialized metadata or 2D run names.
"""Return the saved Allen-Cahn boundary condition."""
function saved_boundary_condition(data, run_dir::AbstractString="")
    hasproperty(data, :boundary_condition) && return string(data.boundary_condition)
    occursin("periodic", lowercase(run_dir)) && return "periodic"
    return "homogeneous_dirichlet"
end

### ADJUSTED: Cap saved reference output times used by local visualization tooling.
"""Return saved reference output times, reconstructing them for compact old HPO metadata."""
function reference_saved_times(params)
    times = if hasproperty(params, :reference_saved_times)
        collect(params.reference_saved_times)
    elseif hasproperty(params, :reference_save_count)
        collect(LinRange(params.tspan[1], params.tspan[2], min(Int(params.reference_save_count), MAX_REFERENCE_SAVED_TIMES)))
    else
        collect(params.t_obs)
    end
    length(times) <= MAX_REFERENCE_SAVED_TIMES && return times
    return collect(LinRange(first(times), last(times), MAX_REFERENCE_SAVED_TIMES))
end

"""Return the reference Euler step size when it was stored separately from `saveat`."""
reference_dt(params) = hasproperty(params, :reference_dt) ? params.reference_dt : nothing

### ADJUSTED: Reconstruct the stable explicit Euler reference timestep when old runs did not serialize it.
"""Return the stable explicit Euler timestep used by reference replays when no saved `reference_dt` exists."""
stable_reference_dt(p) = 0.5 * p.Δx^2 / (2 * (hasproperty(p, :dimension) ? Int(p.dimension) : 1) * p.ε2)

"""Return the saved reference algorithm, defaulting compact HPO metadata to Euler."""
reference_algorithm(params) = hasproperty(params, :reference_algorithm) ? params.reference_algorithm : "Euler"

"""Return the saved Allen-Cahn reaction coefficient."""
saved_k(params) = hasproperty(params, :k) ? params.k : params.reference_parameters.k



"""Return the per-axis grid count for a saved 1D or flattened 2D run."""
function saved_grid_N(data, run_dir::AbstractString="")
    if hasproperty(data, :grid_N)
        return Int(data.grid_N)
    elseif hasproperty(data, :state_shape)
        return Int(first(data.state_shape))
    elseif saved_dimension(data, run_dir) == 2
        if hasproperty(data, :N) && hasproperty(data, :spatial_modes) && Int(data.N)^2 == size(data.spatial_modes, 1)
            return Int(data.N)
        end
        state_length = hasproperty(data, :u₀) ? length(data.u₀) : Int(data.N)
        return round(Int, sqrt(state_length))
    else
        return Int(data.N)
    end
end


"""Return the coordinate vectors saved for plotting a 1D or 2D state."""
function saved_grid_axes(data, run_dir::AbstractString="")
    N = saved_grid_N(data, run_dir)
    x = hasproperty(data, :x) ? data.x : data.Δx .* collect(1:N)
    y = saved_dimension(data, run_dir) == 1 ? nothing : (hasproperty(data, :y) && !isnothing(data.y) ? data.y : copy(x))
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

### ADJUSTED: Add periodic 1D operators so 2D ROM boundary-condition runs reconstruct like training.
"""Apply the 1D Laplacian with periodic boundary conditions."""
function viz_lap1d_periodic!(du, u, invΔx2)
    n = length(u)
    @inbounds for i in 1:n
        left = i == 1 ? n : i - 1
        right = i == n ? 1 : i + 1
        du[i] = (u[left] - 2u[i] + u[right]) * invΔx2
    end
    return nothing
end

"""Construct the 1D Allen-Cahn sparse diffusion matrix with periodic boundaries."""
function viz_lap1d_periodic_matrix(N, ε2, Δx)
    A = viz_lap1d_matrix(N, ε2, Δx)
    scale = ε2 / Δx^2
    A[1, N] = scale
    A[N, 1] = scale
    return A
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

### ADJUSTED: Add periodic flattened 2D Laplacian support for saved boundary-condition comparisons.
"""Apply the flattened 2D Laplacian with periodic boundary conditions."""
function viz_lap2d_periodic!(du, u, N::Integer, invΔx2)
    @inbounds for j in 1:N, i in 1:N
        idx = i + (j - 1) * N
        left = (i == 1 ? N : i - 1) + (j - 1) * N
        right = (i == N ? 1 : i + 1) + (j - 1) * N
        down = i + ((j == 1 ? N : j - 1) - 1) * N
        up = i + ((j == N ? 1 : j + 1) - 1) * N
        du[idx] = (u[left] + u[right] + u[down] + u[up] - 4u[idx]) * invΔx2
    end
    return nothing
end

"""Apply a 1D or flattened 2D Laplacian for visualization solves."""
function viz_lap_ac!(du, u, N::Integer, dimension, invΔx2, boundary_condition="homogeneous_dirichlet")
    bc = string(boundary_condition)
    if Int(dimension) == 1
        bc == "periodic" ? viz_lap1d_periodic!(du, u, invΔx2) : viz_lap1d!(du, u, invΔx2)
    else
        bc == "periodic" ? viz_lap2d_periodic!(du, u, N, invΔx2) : viz_lap2d!(du, u, N, invΔx2)
    end
    return nothing
end


"""Construct the flattened 2D Allen-Cahn sparse diffusion matrix."""
function viz_lap2d_matrix(N, ε2, Δx, boundary_condition="homogeneous_dirichlet")
    L1 = string(boundary_condition) == "periodic" ?
        viz_lap1d_periodic_matrix(N, one(ε2), Δx) :
        viz_lap1d_matrix(N, one(ε2), Δx)
    I_N = sparse(I, N, N)
    return ε2 * (kron(I_N, L1) + kron(L1, I_N))
end


"""Construct the 1D or flattened 2D Allen-Cahn sparse diffusion matrix."""
function viz_lap_ac_matrix(N, ε2, Δx, dimension, boundary_condition="homogeneous_dirichlet")
    if Int(dimension) == 1
        return string(boundary_condition) == "periodic" ?
            viz_lap1d_periodic_matrix(N, ε2, Δx) :
            viz_lap1d_matrix(N, ε2, Δx)
    end
    return viz_lap2d_matrix(N, ε2, Δx, boundary_condition)
end


"""True Allen-Cahn RHS used for reference solves."""
function viz_rhs_ac!(du, u, p, t)
    (; ε2, k, Δx) = p
    dimension = hasproperty(p, :dimension) ? p.dimension : 1
    N = hasproperty(p, :N) ? p.N : (Int(dimension) == 1 ? length(u) : round(Int, sqrt(length(u))))
    boundary_condition = hasproperty(p, :boundary_condition) ? p.boundary_condition : "homogeneous_dirichlet"
    viz_lap_ac!(du, u, N, dimension, 1 / Δx^2, boundary_condition)
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


"""Solve the true Allen-Cahn trajectory using saved reference solver settings and a stable Euler fallback timestep."""
function solve_true_trajectory(u₀, tspan, p, save_times, reference_algorithm::AbstractString; dt=nothing)
    prob = ODEProblem(viz_rhs_ac!, u₀, tspan, p)
    alg = algorithm_from_name(reference_algorithm)
    if occursin("Euler", reference_algorithm)
        return solve(prob, alg; dt=something(dt, stable_reference_dt(p)), saveat=save_times)
    end
    return solve(prob, alg; saveat=save_times)
end


"""Convert an ODE solution's vector states into an `N x time` matrix."""
solution_matrix(sol) = hcat(sol.u...)


"""Solve the final learned FOM neural trajectory from a saved FOM run."""
function solve_fom_learned_trajectory(run_dir::AbstractString)
    params = load_fom_run_params(run_dir)
    θ_flat = final_theta(run_dir)
    dimension = saved_dimension(params, run_dir)
    N = saved_grid_N(params, run_dir)
    boundary_condition = saved_boundary_condition(params, run_dir)
    if saved_model_type(params) == "polynomial"
        ### ADJUSTED: Reconstruct learned 2D FOM trajectories from polynomial coefficients when saved.
        p = ComponentVector(ε2=params.ε2, Δx=params.Δx, θ=θ_flat)
        rhs! = (du, u, p, t) -> begin
            viz_lap_ac!(du, u, N, dimension, 1 / p.Δx^2, boundary_condition)
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
        ### ADJUSTED: Reuse the saved boundary condition when reconstructing learned 2D FOM trajectories.
        viz_lap_ac!(du, u, N, dimension, 1 / p.Δx^2, boundary_condition)
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
        (; ε2=params.ε2, k=hasproperty(params, :k) ? params.k : 1.0, Δx=params.Δx, N=saved_grid_N(params, run_dir), dimension=saved_dimension(params, run_dir), boundary_condition=saved_boundary_condition(params, run_dir))
    save_times = reference_saved_times(params)
    return solve_true_trajectory(params.u₀, params.tspan, p, save_times, reference_algorithm(params); dt=reference_dt(params))
end


"""Reconstruct ROM operators from saved modes and scalar PDE parameters."""
function reconstruct_rom_operators(rom_data, run_dir::AbstractString="")
    ### ADJUSTED: Reconstruct the ROM diffusion operator with inferred 2D and boundary-condition metadata.
    A = viz_lap_ac_matrix(saved_grid_N(rom_data, run_dir), rom_data.ε2, rom_data.Δx, saved_dimension(rom_data, run_dir), saved_boundary_condition(rom_data, run_dir))
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
    ops = reconstruct_rom_operators(rom_data, run_dir)
    if saved_model_type(rom_data) == "polynomial"
        ### ADJUSTED: Reconstruct learned 2D ROM trajectories from polynomial coefficients when saved.
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
    ### ADJUSTED: Include inferred boundary metadata for true 2D ROM reference reconstruction.
    p = (; ε2=rom_data.ε2, k=rom_data.k, Δx=rom_data.Δx, N=saved_grid_N(rom_data, run_dir), dimension=saved_dimension(rom_data, run_dir), boundary_condition=saved_boundary_condition(rom_data, run_dir))
    return solve_true_trajectory(rom_data.u₀, rom_data.tspan, p, reference_saved_times(rom_data), reference_algorithm; dt=reference_dt(rom_data))
end


"""Create and save an animated GIF for a trajectory matrix."""
function save_trajectory_gif(x, t, u; path::AbstractString, title::AbstractString, max_frames=120, fps=15, ylims=nothing)
    mkpath(dirname(path))
    frame_count = min(max_frames, length(t))
    frame_ids = unique(round.(Int, LinRange(1, length(t), frame_count)))
    ### ADJUSTED: Delegate 1D trajectory GIF writing so paired GIFs can share frame timing.
    return save_trajectory_gif(x, t, u, frame_ids; path, title, fps, ylims)
end

### ADJUSTED: Add a frame-id based 1D trajectory writer for synchronized separate GIFs.
"""Create and save a 1D trajectory GIF using caller-supplied frame indices."""
function save_trajectory_gif(x, t, u, frame_ids; path::AbstractString, title::AbstractString, fps=15, ylims=nothing, color=:blue)
    mkpath(dirname(path))
    anim = @animate for j in frame_ids
        plot(
            x,
            u[:, j];
            xlim=(first(x), last(x)),
            ylim=ylims,
            xlabel="x",
            ylabel="u",
            title="$title, t = $(@sprintf("%.3f", t[j]))",
            legend=false,
            color,
            linewidth=2,
            size=(650, 420),
        )
    end
    gif(anim, path; fps)
    return path
end

### ADJUSTED: Add configurable 2D GIF frame size and optional heatmap interpolation.
"""Create and save a 2D trajectory GIF using caller-supplied frame indices and optional render sizing."""
function save_2d_trajectory_gif(x, y, t, u, frame_ids; path::AbstractString, title::AbstractString, fps=15, clims=nothing, show_colorbar=false, color_scheme=:viridis, plot_size=nothing, interpolate=false)
    mkpath(dirname(path))
    N = length(x)
    frame_size = isnothing(plot_size) ? (show_colorbar ? (560, 470) : (500, 470)) : plot_size
    anim = @animate for j in frame_ids
        heatmap(
            x,
            y,
            reshape(u[:, j], N, N);
            clims,
            xlim=(first(x), last(x)),
            ylim=(first(y), last(y)),
            aspect_ratio=:equal,
            xlabel="x",
            ylabel="y",
            title="$title t = $(@sprintf("%.3f", t[j]))",
            colorbar=show_colorbar,
            color=color_scheme,
            size=frame_size,
            interpolate=interpolate,
        )
    end
    gif(anim, path; fps)
    return path
end


"""Create and save an animated GIF with true and learned trajectories overlaid."""
function save_overlay_trajectory_gif(x, t, true_u, learned_u; path::AbstractString, title::AbstractString, max_frames=120, fps=15, ylims=nothing)
    return save_overlay_trajectories_gif(x, t, true_u, [learned_u]; path, title, max_frames, fps, ylims, labels=["learned"])
end


### ADJUSTED: Add multi-ROM 1D trajectory overlays against one true trajectory.
"""Create and save an animated GIF with one true trajectory and one or more learned trajectories overlaid."""
function save_overlay_trajectories_gif(x, t, true_u, learned_us; path::AbstractString, title::AbstractString, max_frames=120, fps=15, ylims=nothing, labels=nothing, true_label="true", true_color=:blue, learned_colors=nothing)
    mkpath(dirname(path))
    time_count = minimum(vcat([length(t), size(true_u, 2)], [size(u, 2) for u in learned_us]))
    frame_count = min(max_frames, time_count)
    frame_ids = unique(round.(Int, LinRange(1, time_count, frame_count)))
    learned_labels = isnothing(labels) ? ["learned $i" for i in eachindex(learned_us)] : labels
    colors = isnothing(learned_colors) ? palette(:tab10, length(learned_us)) : learned_colors
    anim = @animate for j in frame_ids
        p = plot(
            x,
            true_u[:, j];
            xlabel="x",
            ylabel="u",
            title="$title, t = $(@sprintf("%.3f", t[j]))",
            label=true_label,
            color=true_color,
            linewidth=2,
        )
        !isnothing(ylims) && ylims!(p, ylims)
        for (u, label, color) in zip(learned_us, learned_labels, colors)
            plot!(
                p,
                x,
                u[:, j];
                label,
                color,
                linewidth=2,
                linestyle=:dash,
            )
        end
    end
    gif(anim, path; fps)
    return path
end


"""Create and save an animated GIF comparing flattened 2D true and learned trajectories."""
function save_overlay_2d_trajectory_gif(x, y, t, true_u, learned_u; path::AbstractString, title::AbstractString, max_frames=120, fps=15, clims=nothing)
    return save_overlay_2d_trajectories_gif(x, y, t, true_u, [learned_u]; path, title, max_frames, fps, clims, labels=["learned"])
end


### ADJUSTED: Add synchronized multi-ROM 2D trajectory comparisons against one true trajectory.
"""Create and save an animated GIF comparing one true 2D trajectory with one or more learned trajectories."""
function save_overlay_2d_trajectories_gif(x, y, t, true_u, learned_us; path::AbstractString, title::AbstractString, max_frames=120, fps=15, clims=nothing, labels=nothing, true_label="true", show_colorbar=false, color_scheme=:viridis, plot_size=nothing, interpolate=false)
    mkpath(dirname(path))
    N = length(x)
    time_count = minimum(vcat([length(t), size(true_u, 2)], [size(u, 2) for u in learned_us]))
    frame_count = min(max_frames, time_count)
    frame_ids = unique(round.(Int, LinRange(1, time_count, frame_count)))
    learned_labels = isnothing(labels) ? ["learned $i" for i in eachindex(learned_us)] : labels
    n_panels = 1 + length(learned_us)
    frame_size = isnothing(plot_size) ? (show_colorbar ? (360 * n_panels + 80, 360) : (360 * n_panels, 360)) : plot_size
    anim = @animate for j in frame_ids
        plots = Any[
            heatmap(x, y, reshape(true_u[:, j], N, N); clims, aspect_ratio=:equal, xlabel="x", ylabel="y", title=true_label, colorbar=false, color=color_scheme, interpolate)
        ]
        for (i, u) in enumerate(learned_us)
            push!(
                plots,
                heatmap(x, y, reshape(u[:, j], N, N); clims, aspect_ratio=:equal, xlabel="x", ylabel="y", title=learned_labels[i], colorbar=show_colorbar && i == length(learned_us), color=color_scheme, interpolate),
            )
        end
        plot(plots...; layout=(1, n_panels), size=frame_size, plot_title="$title, t = $(@sprintf("%.3f", t[j]))")
    end
    gif(anim, path; fps)
    return path
end


### ADJUSTED: Allow wide combined GIFs to display at native pixel width instead of being downscaled.
"""Display a saved GIF in a notebook."""
function display_gif(path::AbstractString; max_width="100%")
    encoded = base64encode(read(path))
    width_style = max_width == "none" ? "width: auto; max-width: none;" : "max-width: $max_width;"
    display("text/html", HTML("""
    <div style="overflow-x: auto;">
        <img src="data:image/gif;base64,$encoded" style="$width_style image-rendering: auto;">
    </div>
    """))
    return nothing
end


### ADJUSTED: Add synchronized side-by-side GIF display and optional combined-GIF saving.
"""Display GIFs side by side, optionally saving and displaying one combined side-by-side GIF."""
function display_gifs_side_by_side(paths::AbstractString...; same_gif=false, output_path=nothing, gap_px=8, duration_ms=nothing)
    if same_gif
        saved_path = save_gifs_side_by_side(paths...; output_path, gap_px, duration_ms)
        display_gif(saved_path; max_width="none")
        return saved_path
    end

    imgs = String[]
    max_width = 100 / max(length(paths), 1) - 1
    for path in paths
        encoded = base64encode(read(path))
        push!(imgs, """
        <img src="data:image/gif;base64,$encoded"
             style="max-width: $(max_width)%; vertical-align: top;">
        """)
    end

    display("text/html", HTML("""
    <div style="display: flex; gap: $(gap_px)px; align-items: flex-start;">
        $(join(imgs, "\n"))
    </div>
    """))

    return nothing
end


### ADJUSTED: Add a small helper for saving combined side-by-side GIFs without overwriting the source GIFs.
"""Save multiple GIFs as one side-by-side animated GIF and return the saved path."""
function save_gifs_side_by_side(paths::AbstractString...; output_path=nothing, gap_px=8, duration_ms=nothing)
    length(paths) >= 2 || error("At least two GIF paths are required.")
    saved_path = isnothing(output_path) ? default_side_by_side_gif_path(first(paths)) : output_path
    mkpath(dirname(saved_path))

    python_code = raw"""
import sys
from PIL import Image, ImageSequence

paths = sys.argv[1:-3]
out_path = sys.argv[-3]
gap_px = int(sys.argv[-2])
duration_arg = sys.argv[-1]
duration_ms = int(duration_arg) if duration_arg else None

frame_sets = []
widths = []
heights = []
for path in paths:
    image = Image.open(path)
    frames = [frame.convert("RGBA") for frame in ImageSequence.Iterator(image)]
    frame_sets.append(frames)
    widths.append(max(frame.width for frame in frames))
    heights.append(max(frame.height for frame in frames))
    if duration_ms is None:
        duration_ms = int(image.info.get("duration", 100))

frame_count = max(len(frames) for frames in frame_sets)
canvas_width = sum(widths) + gap_px * (len(frame_sets) - 1)
canvas_height = max(heights)
combined = []

for j in range(frame_count):
    canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))
    x_offset = 0
    for frames, width, height in zip(frame_sets, widths, heights):
        frame = frames[min(j, len(frames) - 1)]
        y_offset = (canvas_height - frame.height) // 2
        canvas.alpha_composite(frame, (x_offset, y_offset))
        x_offset += width + gap_px
    combined.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE))

combined[0].save(
    out_path,
    save_all=True,
    append_images=combined[1:],
    duration=duration_ms,
    loop=0,
    disposal=2,
)
"""

    duration_arg = isnothing(duration_ms) ? "" : string(duration_ms)
    run(Cmd(vcat(["python3", "-c", python_code], collect(String.(paths)), [saved_path, string(gap_px), duration_arg])))
    return saved_path
end


### ADJUSTED: Add default naming for combined GIFs so source animations are preserved.
"""Return a non-overwriting default path for a combined side-by-side GIF."""
function default_side_by_side_gif_path(path::AbstractString)
    stem, ext = splitext(basename(path))
    return joinpath(dirname(path), stem * "_side_by_side" * ext)
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
    axes = saved_grid_axes(params, run_dir)
    overlay_path = saved_dimension(params, run_dir) == 1 ?
        save_overlay_trajectory_gif(axes.x, true_sol.t, true_u, learned_u; path=joinpath(out_dir, "overlaid_fom_trajectory.gif"), title="FOM trajectory", max_frames, fps, ylims) :
        save_overlay_2d_trajectory_gif(axes.x, axes.y, true_sol.t, true_u, learned_u; path=joinpath(out_dir, "overlaid_fom_trajectory_2d.gif"), title="FOM trajectory", max_frames, fps, clims=ylims)
    if display_gifs
        display_gif(overlay_path)
    end
    return (; overlay_path, true_solution=true_sol, learned_solution=learned_sol)
end


### ADJUSTED: Pass optional 2D GIF render controls through the ROM trajectory helper.
"""Save and optionally display synchronized true and learned ROM trajectory GIFs."""
function rom_trajectory_gifs(run_dir::AbstractString; max_frames=120, fps=15, display_gifs=true, show_colorbar=false, true_title="FOM true trajectory", learned_title="ROM learned trajectory", color_scheme=:viridis, true_color=:blue, learned_color=:orange, plot_size=nothing, interpolate=false)
    rom_data = load_rom_data(run_dir)
    true_sol = solve_rom_true_trajectory(run_dir)
    learned_sol = solve_rom_learned_trajectory(run_dir)
    true_u = solution_matrix(true_sol)
    learned_u = learned_sol.u
    ylims = extrema(vcat(vec(true_u), vec(learned_u)))
    out_dir = joinpath(run_dir, "visualizations")
    axes = saved_grid_axes(rom_data, run_dir)
    time_count = min(length(true_sol.t), size(true_u, 2), size(learned_u, 2))
    frame_count = min(max_frames, time_count)
    frame_ids = unique(round.(Int, LinRange(1, time_count, frame_count)))
    if saved_dimension(rom_data, run_dir) == 1
        true_path = save_trajectory_gif(axes.x, true_sol.t, true_u, frame_ids; path=joinpath(out_dir, "true_rom_reference_trajectory.gif"), title=true_title, fps, ylims, color=true_color)
        learned_path = save_trajectory_gif(axes.x, true_sol.t, learned_u, frame_ids; path=joinpath(out_dir, "learned_rom_trajectory.gif"), title=learned_title, fps, ylims, color=learned_color)
    else
        true_path = save_2d_trajectory_gif(axes.x, axes.y, true_sol.t, true_u, frame_ids; path=joinpath(out_dir, "true_rom_reference_trajectory_2d.gif"), title=true_title, fps, clims=ylims, show_colorbar, color_scheme, plot_size, interpolate)
        learned_path = save_2d_trajectory_gif(axes.x, axes.y, true_sol.t, learned_u, frame_ids; path=joinpath(out_dir, "learned_rom_trajectory_2d.gif"), title=learned_title, fps, clims=ylims, show_colorbar, color_scheme, plot_size, interpolate)
    end
    if display_gifs
        display_gif(true_path)
        display_gif(learned_path)
    end
    return (; true_path, learned_path, true_solution=true_sol, learned_solution=learned_sol)
end


### ADJUSTED: Restore ROM trajectory visualization as an overlaid true-vs-learned comparison.
"""Save and optionally display an overlaid true/learned ROM trajectory GIF."""
function rom_overlay_trajectory_gif(run_dir::AbstractString; max_frames=120, fps=15, display_gifs=true, show_colorbar=false, title=nothing, labels=nothing, true_label="true", color_scheme=:viridis, true_color=:blue, learned_colors=nothing, plot_size=nothing, interpolate=false)
    rom_data = load_rom_data(run_dir)
    true_sol = solve_rom_true_trajectory(run_dir)
    learned_sol = solve_rom_learned_trajectory(run_dir)
    true_u = solution_matrix(true_sol)
    learned_u = learned_sol.u
    ylims = extrema(vcat(vec(true_u), vec(learned_u)))
    out_dir = joinpath(run_dir, "visualizations")
    axes = saved_grid_axes(rom_data, run_dir)
    overlay_title = isnothing(title) ? "FOM vs ROM" : title
    overlay_labels = isnothing(labels) ? ["learned"] : labels
    path = saved_dimension(rom_data, run_dir) == 1 ? joinpath(out_dir, "overlaid_rom_trajectory.gif") : joinpath(out_dir, "overlaid_rom_trajectory_2d.gif")
    overlay_path = saved_dimension(rom_data, run_dir) == 1 ?
        save_overlay_trajectories_gif(axes.x, true_sol.t, true_u, [learned_u]; path, title=overlay_title, max_frames, fps, ylims, labels=overlay_labels, true_label, true_color, learned_colors) :
        save_overlay_2d_trajectories_gif(axes.x, axes.y, true_sol.t, true_u, [learned_u]; path, title=overlay_title, max_frames, fps, clims=ylims, labels=overlay_labels, true_label, show_colorbar, color_scheme, plot_size, interpolate)
    if display_gifs
        display_gif(overlay_path; max_width=saved_dimension(rom_data, run_dir) == 1 ? "100%" : "none")
    end
    return (; overlay_path, true_solution=true_sol, learned_solution=learned_sol)
end


### ADJUSTED: Add multi-run ROM trajectory overlays against one true trajectory.
"""Save and optionally display one true ROM reference trajectory with learned trajectories from multiple runs."""
function rom_overlay_trajectory_gif(run_dirs::AbstractVector{<:AbstractString}; max_frames=120, fps=15, display_gifs=true, show_colorbar=false, labels=nothing, run_names=nothing, title=nothing, true_label="true", color_scheme=:viridis, true_color=:blue, learned_colors=nothing, plot_size=nothing, interpolate=false)
    run_labels = isnothing(labels) ? (isnothing(run_names) ? basename.(run_dirs) : run_names) : labels
    length(run_labels) == length(run_dirs) || error("labels must have the same length as run_dirs")
    first_data = load_rom_data(first(run_dirs))
    true_sol = solve_rom_true_trajectory(first(run_dirs))
    true_u = solution_matrix(true_sol)
    learned_us = [solve_rom_learned_trajectory(run_dir).u for run_dir in run_dirs]
    ylims = extrema(vcat(vec(true_u), vec.(learned_us)...))
    out_dir = joinpath(first(run_dirs), "visualizations")
    axes = saved_grid_axes(first_data, first(run_dirs))
    overlay_title = isnothing(title) ? "FOM vs ROMs" : title
    path = saved_dimension(first_data, first(run_dirs)) == 1 ? joinpath(out_dir, "overlaid_rom_trajectory_comparison.gif") : joinpath(out_dir, "overlaid_rom_trajectory_comparison_2d.gif")
    overlay_path = saved_dimension(first_data, first(run_dirs)) == 1 ?
        save_overlay_trajectories_gif(axes.x, true_sol.t, true_u, learned_us; path, title=overlay_title, max_frames, fps, ylims, labels=run_labels, true_label, true_color, learned_colors) :
        save_overlay_2d_trajectories_gif(axes.x, axes.y, true_sol.t, true_u, learned_us; path, title=overlay_title, max_frames, fps, clims=ylims, labels=run_labels, true_label, show_colorbar, color_scheme, plot_size, interpolate)
    if display_gifs
        display_gif(overlay_path; max_width=saved_dimension(first_data, first(run_dirs)) == 1 ? "100%" : "none")
    end
    return (; overlay_path, true_solution=true_sol, learned_solutions=learned_us)
end


### ADJUSTED: Add a static initial-condition plot matching the ROM trajectory visualization styling.
"""Return true and ROM-projected initial-condition plots, optionally saving the figure."""
function plot_rom_initial_condition(run_dir::AbstractString; show_colorbar=false, color_scheme=:viridis, colorscheme=nothing, true_title="FOM true initial condition", learned_title="ROM projected initial condition", save_path=nothing)
    rom_data = load_rom_data(run_dir)
    axes = saved_grid_axes(rom_data, run_dir)
    scheme = isnothing(colorscheme) ? color_scheme : colorscheme
    true_u0 = rom_data.u₀
    projected_u0 = rom_data.spatial_modes * (rom_data.spatial_modes' * true_u0)
    clims = extrema(vcat(vec(true_u0), vec(projected_u0)))

    if saved_dimension(rom_data, run_dir) == 1
        p = plot(
            axes.x,
            true_u0;
            xlim=(first(axes.x), last(axes.x)),
            ylim=clims,
            xlabel="x",
            ylabel="u",
            title=true_title,
            label=false,
            color=:blue,
            linewidth=2,
            size=(650, 420),
        )
        plot!(
            p,
            axes.x,
            projected_u0;
            label=false,
            color=:orange,
            linewidth=2,
            linestyle=:dash,
        )
    else
        N = saved_grid_N(rom_data, run_dir)
        p_true = heatmap(
            axes.x,
            axes.y,
            reshape(true_u0, N, N);
            clims,
            xlim=(first(axes.x), last(axes.x)),
            ylim=(first(axes.y), last(axes.y)),
            aspect_ratio=:equal,
            xlabel="x",
            ylabel="y",
            title=true_title,
            colorbar=show_colorbar,
            color=scheme,
        )
        p_learned = heatmap(
            axes.x,
            axes.y,
            reshape(projected_u0, N, N);
            clims,
            xlim=(first(axes.x), last(axes.x)),
            ylim=(first(axes.y), last(axes.y)),
            aspect_ratio=:equal,
            xlabel="x",
            ylabel="y",
            title=learned_title,
            colorbar=show_colorbar,
            color=scheme,
        )
        p = plot(p_true, p_learned; layout=(1, 2), size=show_colorbar ? (1000, 470) : (900, 470))
    end

    if !isnothing(save_path)
        mkpath(dirname(save_path))
        savefig(p, save_path)
    end

    return p
end


### ADJUSTED: Add a display-and-save wrapper for ROM initial-condition plots.
"""Display the ROM initial-condition plot and return the saved path when requested."""
function visualize_ROM_initial_condition(run_dir::AbstractString; show_colorbar=false, color_scheme=:viridis, colorscheme=nothing, true_title="FOM true initial condition", learned_title="ROM projected initial condition", save_plot=false, save_path=joinpath(run_dir, "visualizations", "rom_initial_condition.png"))
    p = plot_rom_initial_condition(run_dir; show_colorbar, color_scheme, colorscheme, true_title, learned_title, save_path=save_plot ? save_path : nothing)
    display(p)
    return save_plot ? save_path : nothing
end


"""Return a plot of saved ROM spatial modes and nonlinear/function modes."""
function plot_rom_modes(run_dir::AbstractString; n_modes=6)
    rom_data = load_rom_data(run_dir)
    axes = saved_grid_axes(rom_data, run_dir)
    n_state = min(n_modes, size(rom_data.spatial_modes, 2))
    n_function = min(n_modes, size(rom_data.deim_modes, 2))

    if saved_dimension(rom_data, run_dir) == 2
        N = saved_grid_N(rom_data, run_dir)
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
function plot_learned_function(run_dir::AbstractString; u_min=-1.2, u_max=1.2, n_points=400, title=nothing)
    if isnothing(title)
        title = "True vs learned nonlinearity"
    end
    θ_flat = final_theta(run_dir)
    u = collect(LinRange(u_min, u_max, n_points))
    if isfile(joinpath(run_dir, "rom_data.jls"))
        data = load_rom_data(run_dir)
    else
        data = load_fom_run_params(run_dir)
    end
    k = saved_k(data)
    p = plot(u, true_function_values(u, k); label="True " * L"-k(u^3-u)", xlabel=L"u", ylabel=L"f(u)", title=title)
    if saved_model_type(data) == "polynomial"
        degree = hasproperty(data, :polynomial_degree) ? data.polynomial_degree : length(θ_flat) - 1
        plot!(p, u, learned_polynomial_values(u, θ_flat); label="learned polynomial degree $degree")
    else
        plot!(p, u, learned_function_values(u, θ_flat, data.h, data.seed); label="Learned Neural "* L"f(u)")
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


### ADJUSTED: Route ROM trajectory visualization through the overlaid comparison helper.
"""Solve and display an overlaid true/learned trajectory GIF for a saved ROM run."""
function visualize_ROM_trajectories(run_dir::AbstractString; max_frames=120, fps=15, show_colorbar=false, title=nothing, true_title="true", learned_title="learned", color_scheme=:viridis, true_color=:blue, learned_color=:orange, plot_size=nothing, interpolate=false)
    rom_overlay_trajectory_gif(run_dir; max_frames, fps, display_gifs=true, show_colorbar, title, labels=[learned_title], true_label=true_title, color_scheme, true_color, learned_colors=[learned_color], plot_size, interpolate)
    return nothing
end


### ADJUSTED: Allow multiple ROM runs to be shown in one synchronized trajectory comparison.
"""Solve and display one true ROM reference trajectory with learned trajectories from multiple saved runs."""
function visualize_ROM_trajectories(run_dirs::AbstractVector{<:AbstractString}; max_frames=120, fps=15, show_colorbar=false, labels=nothing, run_names=nothing, title=nothing, true_title="true", color_scheme=:viridis, true_color=:blue, learned_colors=nothing, plot_size=nothing, interpolate=false)
    rom_overlay_trajectory_gif(run_dirs; max_frames, fps, display_gifs=true, show_colorbar, labels, run_names, title, true_label=true_title, color_scheme, true_color, learned_colors, plot_size, interpolate)
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
function visualize_ROM_learned_function(run_dir::AbstractString; u_min=-1.2, u_max=1.2, n_points=400,title=nothing)
    display(plot_learned_function(run_dir; u_min, u_max, n_points, title))
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
