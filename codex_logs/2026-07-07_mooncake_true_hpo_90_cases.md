# 2026-07-07 Mooncake True HPO 90-Case Sweep

## Summary
- Added row-wise `[[case]]` support to the sweep parser for explicit HPO recipes.
- Added one 90-case Mooncake FOM HPO sweep file with budgeted staged schedules.

## Edited files
- `Research_Code/HPC_compatibility/Sweeps/sweep_params.py`
  - Preserves existing Cartesian sweep behavior.
  - Adds row-wise `[[case]]` parsing and `CASE_NAME`-based labels.
- `Research_Code/Optimization/Data/Sweeps/FOM_hyperparam_opt/01_mooncake_true_hpo_90_cases.txt`
  - Contains 30 fast-screen, 30 medium, 15 batching, and 15 long/showpiece FOM recipes.
- `codex_logs/2026-07-07_mooncake_true_hpo_90_cases.md`
  - Records this change.
