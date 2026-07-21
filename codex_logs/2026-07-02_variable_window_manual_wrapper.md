Variable-window manual wrapper fix

Files edited:
- `Research_Code/HPC_compatibility/Variable_trajectory_length/hpc1_run_variable_fom.slurm`
  - Let manual runs outside Slurm fall back to the current directory when `SLURM_SUBMIT_DIR` is unset.
- `Research_Code/HPC_compatibility/Variable_trajectory_length/run_variable_fom_hpc.jl`
  - Added a lightweight preflight schedule-length validation before loading the optimization helper packages.
