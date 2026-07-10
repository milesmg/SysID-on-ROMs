### ADJUSTED: Move stability visualization helpers into shared visualization code.

Files edited:
- `Research_Code/src/Visualizations/optimization_visualizations.jl`: added small in-memory helpers for FOM-vs-ROM overlay GIFs and ROM mode plots using existing plotting/display functions.
- `Research_Code/Optimization/Local/test_ROM_stability.ipynb`: removed duplicated notebook-local GIF/display/mode plotting helpers, updated calls to use shared helpers, and cleared stale outputs.
