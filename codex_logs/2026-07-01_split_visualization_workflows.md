Split the FOM/ROM visualization workflows into separate functions for each requested visualization output.

Files edited:

- `Research_Code/helper_functions/Visualizations/optimization_visualizations.jl`
  - Added `visualize_ROM_metadata`, `visualize_ROM_trajectories`, `visualize_ROM_modes`, `visualize_ROM_singular_values`, `visualize_ROM_learned_function`, and `visualize_ROM_loss`.
  - Added `visualize_FOM_metadata`, `visualize_FOM_trajectories`, `visualize_FOM_modes`, `visualize_FOM_learned_function`, and `visualize_FOM_loss`.
  - Kept `visualize_ROM` and `visualize_FOM` as compatibility wrappers that call the split functions in order.
- `Research_Code/Optimization/visualize_ROM.ipynb`
  - Replaced the single aggregate `visualize_ROM(...)` call with separate cells for metadata, trajectory GIFs, ROM modes, singular-value capture, learned function, and loss history.
  - Cleared saved execution outputs from the notebook.
- `Research_Code/Optimization/visualize_FOM.ipynb`
  - Replaced the single aggregate `visualize_FOM(...)` call with separate cells for metadata, trajectory GIFs, FOM mode availability, learned function, and loss history.
  - Cleared saved execution outputs from the notebook.
- `codex_logs/2026-07-01_split_visualization_workflows.md`
  - Documented the workflow split and files edited.
