"""Return a single-component frame from an AC/CH trajectory or one RD species."""
function visualization_component(frame, grid::Grid, component::Integer=1)
    n = spatial_length(grid)
    component == 1 && return frame[1:n]
    component == 2 && return frame[n+1:2n]
    error("component must be 1 or 2")
end

"""Return fixed color limits spanning all reference and learned frames for one component."""
function trajectory_color_limits(trajectories, component::Integer=1)
    grid = trajectories.reference.grid
    values = vcat(vec(visualization_component(trajectories.reference.frames, grid, component)),
                  vec(visualization_component(trajectories.learned.frames, grid, component)))
    extrema(values)
end

"""Plot one 1D component of a reference/learned trajectory comparison."""
function plot_trajectory_1d(reference, learned; frame_index=lastindex(reference.times), component=1,
                            title="Trajectory comparison", xlabel="x", ylabel="state",
                            legend_labels=("reference", "learned"), linewidth=2, save=true,
                            save_name="trajectory_comparison.png")
    grid = reference.grid
    reference_values = visualization_component(reference.frames[:, frame_index], grid, component)
    learned_values = visualization_component(learned.frames[:, frame_index], grid, component)
    time_label = round(reference.times[frame_index]; digits=4)
    plot_object = plot(grid.x, reference_values; label=legend_labels[1], linewidth,
        title="$title, t = $time_label", xlabel, ylabel)
    plot!(plot_object, grid.x, learned_values; label=legend_labels[2], linewidth, linestyle=:dash)
    # ### ADJUSTED: Route static notebook output to the equation-specific local visualization Data folder.
    save && save_run_plot(reference.run_dir, plot_object, save_name; equation=reference.equation)
    plot_object
end

"""Plot one 2D component of a reference/learned trajectory comparison with common color limits."""
function plot_trajectory_2d(reference, learned; frame_index=lastindex(reference.times), component=1,
                            title="Trajectory comparison", xlabel="x", ylabel="y",
                            legend_labels=("reference", "learned"), colorbar=true,
                            color_limits=nothing, save=true, save_name="trajectory_comparison.png")
    grid = reference.grid
    limits = isnothing(color_limits) ? trajectory_color_limits((; reference, learned), component) : color_limits
    reference_values = reshape(visualization_component(reference.frames[:, frame_index], grid, component), grid.state_shape)
    learned_values = reshape(visualization_component(learned.frames[:, frame_index], grid, component), grid.state_shape)
    time_label = round(reference.times[frame_index]; digits=4)
    left = heatmap(grid.x, grid.y, reference_values; title="$(legend_labels[1]), t = $time_label",
        xlabel, ylabel, colorbar, clims=limits)
    right = heatmap(grid.x, grid.y, learned_values; title=legend_labels[2],
        xlabel, ylabel, colorbar, clims=limits)
    plot_object = plot(left, right; layout=(1, 2), size=(900, 420), plot_title=title)
    # ### ADJUSTED: Route static notebook output to the equation-specific local visualization Data folder.
    save && save_run_plot(reference.run_dir, plot_object, save_name; equation=reference.equation)
    plot_object
end

"""Plot all available fields for one reference/learned frame, choosing 1D lines or 2D heatmaps."""
function plot_trajectory_comparison(trajectories; frame_index=lastindex(trajectories.reference.times),
                                    title="Trajectory comparison", xlabel="x", ylabel="state",
                                    legend_labels=("reference", "learned"), colorbar=true,
                                    color_limits=nothing, save=true, save_name="trajectory_comparison.png")
    reference, learned = trajectories.reference, trajectories.learned
    component_count = reference.equation == "rd" ? 2 : 1
    plots = Any[]
    for component in 1:component_count
        component_title = component_count == 1 ? title : "$title, v$component"
        component_name = component_count == 1 ? save_name : replace(save_name, ".png" => "_v$(component).png")
        limits = isnothing(color_limits) || !(color_limits isa Tuple && first(color_limits) isa Tuple) ? color_limits : color_limits[component]
        push!(plots, reference.grid.dimension == 1 ?
            plot_trajectory_1d(reference, learned; frame_index, component, title=component_title, xlabel, ylabel,
                legend_labels, save=false, save_name=component_name) :
            plot_trajectory_2d(reference, learned; frame_index, component, title=component_title, xlabel, ylabel,
                legend_labels, colorbar, color_limits=limits, save=false, save_name=component_name))
    end
    plot_object = length(plots) == 1 ? only(plots) : plot(plots...; layout=(component_count, 1), size=(920, 420 * component_count))
    # ### ADJUSTED: Route static notebook output to the equation-specific local visualization Data folder.
    save && save_run_plot(reference.run_dir, plot_object, save_name; equation=reference.equation)
    plot_object
end

"""Render and save a reference-versus-learned GIF with fixed color ranges across every frame."""
function save_trajectory_gif(trajectories; title="Trajectory comparison", xlabel="x", ylabel="state",
                             legend_labels=("reference", "learned"), colorbar=true, fps=8,
                             max_frames=60, name="trajectory_comparison.gif")
    reference = trajectories.reference
    frame_ids = unique(round.(Int, LinRange(1, length(reference.times), min(max_frames, length(reference.times)))))
    color_limits = reference.equation == "rd" ? (trajectory_color_limits(trajectories, 1), trajectory_color_limits(trajectories, 2)) :
        trajectory_color_limits(trajectories, 1)
    animation = @animate for frame_index in frame_ids
        plot_trajectory_comparison(trajectories; frame_index, title, xlabel, ylabel, legend_labels,
            colorbar, color_limits, save=false)
    end
    # ### ADJUSTED: Save GIFs beside the relevant visualization notebook rather than in the run directory.
    path = joinpath(visualization_output_directory(reference.equation), name)
    gif(animation, path; fps)
    path
end

"""Evaluate the saved final learned reaction and its fixed reference counterpart on a configurable grid."""
function learned_function_values(run_dir::AbstractString; value_range=(-1.0, 1.0), n_points=101)
    context = prepare_visualization_run(run_dir)
    values = collect(LinRange(value_range[1], value_range[2], n_points))
    θ = context.parameters.θ
    if context.spec.name == "ac"
        (; context, x=values, learned=ac_values(values, context.prepared.learner, θ), true_values=ac_fixed_values(values, context.config.parameters.k))
    elseif context.spec.name == "ch"
        (; context, x=values, learned=ch_values(values, context.prepared.learner, θ), true_values=ch_fixed_values(values))
    else
        v1 = repeat(values; inner=n_points)
        v2 = repeat(values; outer=n_points)
        (; context, x=values, learned=reshape(rd_values(v1, v2, context.prepared.learner, θ), n_points, n_points),
          true_values=reshape(rd_s2.(v1, v2), n_points, n_points))
    end
end

"""Plot the final learned reaction against the fixed reference reaction for AC, CH, or RD."""
function plot_learned_function(run_dir::AbstractString; plot_true=true, value_range=(-1.0, 1.0), n_points=101,
                               title="Learned reaction function", xlabel="input", ylabel="reaction",
                               legend_labels=("true", "learned"), colorbar=true, color_limits=nothing,
                               save=true, save_name="learned_function.png")
    values = learned_function_values(run_dir; value_range, n_points)
    if values.context.spec.name in ("ac", "ch")
        plot_object = plot(values.x, values.learned; label=legend_labels[2], linewidth=2,
            title, xlabel, ylabel)
        plot_true && plot!(plot_object, values.x, values.true_values; label=legend_labels[1], linewidth=2, linestyle=:dash)
    else
        limits = isnothing(color_limits) ? extrema(plot_true ? vcat(values.true_values, values.learned) : values.learned) : color_limits
        learned_plot = heatmap(values.x, values.x, values.learned'; title="$(legend_labels[2])", xlabel, ylabel, colorbar, clims=limits)
        if plot_true
            true_plot = heatmap(values.x, values.x, values.true_values'; title="$(legend_labels[1])", xlabel, ylabel, colorbar, clims=limits)
            plot_object = plot(true_plot, learned_plot; layout=(1, 2), size=(900, 420), plot_title=title)
        else
            plot_object = plot(learned_plot; size=(460, 420), plot_title=title)
        end
    end
    # ### ADJUSTED: Route learned-function plots to the equation-specific local visualization Data folder.
    save && save_run_plot(run_dir, plot_object, save_name; equation=values.context.spec.name)
    plot_object
end
