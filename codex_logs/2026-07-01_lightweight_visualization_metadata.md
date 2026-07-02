### ADJUSTED: Split metadata visualization from heavy plotting and solver setup.

Metadata cells in the FOM and ROM notebooks now load a small metadata-only helper before loading the full visualization stack. This keeps `visualize_FOM_metadata(run_dir)` and `visualize_ROM_metadata(run_dir)` from paying the package-loading cost for plotting, Lux, and DifferentialEquations.

Files edited:

- `Research_Code/helper_functions/Visualizations/optimization_metadata_visualizations.jl`
  - Added lightweight `print_metadata`, `read_metadata_values`, `visualize_FOM_metadata`, and `visualize_ROM_metadata`.

- `Research_Code/helper_functions/Visualizations/optimization_visualizations.jl`
  - Included the metadata-only helper.
  - Removed duplicate metadata helper definitions from the heavy visualization helper.

- `Research_Code/Optimization/visualize_FOM.ipynb`
  - Changed the first include cell to load only metadata helpers.
  - Added a full visualization include cell after the metadata cell.
  - Cleared stale outputs.

- `Research_Code/Optimization/visualize_ROM.ipynb`
  - Changed the first include cell to load only metadata helpers.
  - Added a full visualization include cell after the metadata cell.
  - Added missing `### ADJUSTED:` comments to visualization cells.
  - Cleared stale outputs.

- `codex_logs/2026-07-01_lightweight_visualization_metadata.md`
  - Documented the metadata split and notebook changes.
