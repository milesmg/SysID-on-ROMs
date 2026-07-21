# 2026-07-07 Readable Backend Logs

## Summary
- Updated the shared variable-window optimization logger to print short ODE and sensitivity backend names instead of raw Julia/SciML object internals.

## Edited files
- `Research_Code/src/HPC/variable_window_common_hpc.jl`
- `codex_logs/2026-07-07_readable_backend_logs.md`

## Notes
- The actual optimization algorithm and sensitivity backend were not changed.
- The live log now reports `ode_algorithm = TRBDF2(autodiff=AutoFiniteDiff())` and `sensitivity_algorithm = GaussAdjoint(MooncakeVJP)` for the current Mooncake setup.
