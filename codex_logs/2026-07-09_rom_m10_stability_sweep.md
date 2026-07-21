### ADJUSTED: Added a ROM m=10 stability HPO sweep.

Files edited:
- `Research_Code/Optimization/Data/Sweeps/ROM_more_hyperparam_opt_for_fu_learning/00_m10_stability_fu_learning.txt`
  - Added 90 explicit row-wise ROM cases for `M=10` across `R = [3,4,5,10,15]`.
  - Tested lower learning-rate schedules, longer taper schedules, higher late-stage `WINDOW_N_OBS`, and limited batching.
  - Set schedule-valued keys per case so 3-stage and 5-stage cases validate correctly.
- `codex_logs/2026-07-09_rom_m10_stability_sweep.md`
  - Logged this sweep addition.
