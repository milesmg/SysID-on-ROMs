### ADJUSTED: Validate variable-window schedule lengths.

Files edited:
- `Research_Code/HPC_compatibility/Variable_trajectory_length/run_variable_fom_hpc.jl`
  - Replaced the `--etas`/`--iters` length check with a combined check for `--etas`, `--iters`, `--window-T`, `--window-N-obs`, `--window-start-policy`, and `--windows-per-iter`.
