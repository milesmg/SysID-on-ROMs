## ROM HPO 2D Visualization

Adjusted the 2D ROM HPO visualization path so the notebook follows the 1D notebook flow while using 2D-aware helpers.

Files edited:
- `Research_Code/Optimization/Figures/ROM_hpo_visualization_2D.ipynb`: switched the helper include to `optimization_visualizations_2D.jl` and pointed the default run at a local 2D ROM boundary-test run.
- `Research_Code/src/Visualizations/optimization_visualizations_2D.jl`: inferred legacy 2D runs from run paths, derived 2D grid size from serialized state length, and added periodic/Dirichlet reconstruction support.
- `Research_Code/src/Visualizations/optimization_metadata_visualizations.jl`: added serialized metadata summaries so older 2D runs display dimension, boundary condition, grid size, and state length even when `metadata.txt` omitted them.
