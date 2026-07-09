### ADJUSTED: Added a minimal local ROM stability notebook.

Files edited:
- `Research_Code/Optimization/Local/test_ROM_stability.ipynb`
  - Replaced the placeholder notebook with a valid Julia notebook.
  - Added functions to run a FOM reference solve, build a POD/DEIM ROM for input `m,r`, run the true-nonlinearity ROM, summarize singular-value capture, and plot trajectories/modes.
  - Added support for default, vector-valued, and scalar-function initial conditions, plus a plot for comparing candidate initial conditions.
  - Reworked the notebook to use the existing HPC `build_ac_reference` and `prepare_ROM_optimization` helpers instead of duplicating FOM reference and ROM construction code.
  - Replaced the static trajectory snapshot plot with FOM-only and FOM-vs-ROM overlaid GIF generation/display.
  - Added notebook-local GIF display and a true 2D FOM-only trajectory GIF helper so the stability trajectory section produces animations for both 1D and 2D runs.
- `codex_logs/2026-07-09_local_rom_stability_notebook.md`
  - Recorded this notebook update.
