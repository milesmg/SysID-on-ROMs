### ADJUSTED: Add optional fixed-scale colorbars to stability visualizations.

Files edited:
- `Research_Code/src/Visualizations/optimization_visualizations.jl`: added `show_colorbar` support to `trajectory_gifs` and the 2D overlay helper, using `clims=(-1, 1)` when enabled.
- `Research_Code/Optimization/Local/test_ROM_stability.ipynb`: added the same optional `show_colorbar` flag to the notebook-local `plot_initial_conditions` function.
