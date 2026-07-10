### ADJUSTED: Remove the difference panel from in-memory FOM/ROM trajectory GIFs.

Files edited:
- `Research_Code/src/Visualizations/optimization_visualizations.jl`: added an `include_difference` keyword to the 2D overlay GIF helper and set `trajectory_gifs` to render only FOM and ROM panels for 2D stability GIFs.
