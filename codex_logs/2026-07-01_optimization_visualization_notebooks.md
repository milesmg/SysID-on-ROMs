Added shared FOM/ROM optimization visualization helpers and thin visualization notebooks.

Files edited:

- `Research_Code/helper_functions/Visualizations/optimization_visualizations.jl`
  - Added reusable helpers to load saved FOM/ROM runs, print metadata, reconstruct true and learned trajectories, save/display GIFs, plot ROM spatial/function modes, plot learned vs true nonlinearities, plot loss histories, and print singular-value capture tables.
  - Updated GIF display to use an HTML `<img>` tag so VS Code Julia notebooks can render saved GIF files.
- `Research_Code/Optimization/visualize_ROM.ipynb`
  - Added a thin ROM notebook that sets `run_dir`, includes the visualization helper file, runs `visualize_ROM`, and prints a longer singular-value capture table.
- `Research_Code/Optimization/visualize_FOM.ipynb`
  - Added a thin FOM notebook that sets `run_dir`, includes the visualization helper file, and runs `visualize_FOM`.
- `codex_logs/2026-07-01_optimization_visualization_notebooks.md`
  - Documented the visualization helper and notebook edits.
