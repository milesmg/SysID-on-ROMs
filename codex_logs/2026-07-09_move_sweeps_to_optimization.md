## Move Sweeps To Optimization

Updated sweep references after moving sweep specs from `Research_Code/Optimization/Data/Sweeps` to `Research_Code/Optimization/Sweeps`.

Files edited:
- `Research_Code/src/HPC/Tools/Sweeps/hpc1_run_sweep.slurm`: updated missing-`SWEEP_FILE` guidance to the new sweep directory.
- `Research_Code/Optimization/Sweeps/ROM_backend_tests/README.md`: updated rsync, submit, inspect, and dry-run paths to the new sweep directory.
- `codex_logs/2026-07-09_polynomial_single_case_sweeps.md`: updated logged polynomial sweep paths to the new sweep directory.
