### ADJUSTED: Overlay true and learned trajectory GIFs.

Trajectory visualization now writes one GIF per FOM/ROM run with true and learned trajectories overlaid in different colors, instead of displaying separate true and learned GIFs.

Files edited:

- `Research_Code/helper_functions/Visualizations/optimization_visualizations.jl`
  - Added `save_overlay_trajectory_gif`.
  - Changed FOM trajectory GIF generation to save and display `overlaid_fom_trajectory.gif`.
  - Changed ROM trajectory GIF generation to save and display `overlaid_rom_trajectory.gif`.
  - Updated trajectory visualization docstrings to describe overlaid true/learned GIFs.

- `Research_Code/Optimization/visualize_FOM.ipynb`
  - Updated the trajectory cell comment to describe the overlaid GIF.
  - Cleared stale outputs.

- `Research_Code/Optimization/visualize_ROM.ipynb`
  - Updated the trajectory cell comment to describe the overlaid GIF.
  - Cleared stale outputs.

- `codex_logs/2026-07-01_overlaid_trajectory_visualizations.md`
  - Documented the trajectory overlay changes.
