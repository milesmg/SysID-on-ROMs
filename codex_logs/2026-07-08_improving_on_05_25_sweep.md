# 2026-07-08 Improving on 05-25 Sweep

## Summary
- Added a 30-case FOM exploitation sweep centered on the best practical Mooncake HPO recipe from case 25.

## Edited files
- `Research_Code/Optimization/Data/Sweeps/FOM_hyperparam_opt/06_improving_on_05-25.txt`
  - Tests shorter/longer iteration budgets, nearby window schedules, Nobs schedules, eta schedules, and seed repeats around `WINDOW_T=[0.05,0.5,2.0]`, `WINDOW_N_OBS=[2,5,10]`, and `ETAS=[0.01,0.003,0.001]`.
- `codex_logs/2026-07-08_improving_on_05_25_sweep.md`
  - Records this sweep-file addition.
