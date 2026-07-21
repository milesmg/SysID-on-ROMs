### ADJUSTED: Add variable-window FOM HPC training.

Built a new variable-window FOM training path under `Research_Code/HPC_compatibility/Variable_trajectory_length` with staged window length, window observation count, start policy, windows per optimizer iteration, deterministic seeded random starts, and mean/sum loss normalization.

Files edited:
- `Research_Code/HPC_compatibility/Variable_trajectory_length/variable_window_FOM_opt_AC_hpc.jl`
  - Added variable-window objective construction, deterministic window schedule generation, staged Adam optimization, full-trajectory validation, and compatible save output.
- `Research_Code/HPC_compatibility/Variable_trajectory_length/run_variable_fom_hpc.jl`
  - Added a Julia HPC entrypoint for variable-window FOM runs.
- `Research_Code/HPC_compatibility/Variable_trajectory_length/hpc1_run_variable_fom.slurm`
  - Added a Slurm wrapper with variable-window environment parameters and dry-run output.
- `Research_Code/HPC_compatibility/Variable_trajectory_length/example_variable_fom_sweep.txt`
  - Added an example `fom_variable` sweep file.
- `Research_Code/HPC_compatibility/Sweeps/sweep_params.py`
  - Added grouped sweep parsing and aliases for variable-window schedules.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`
  - Added the `fom_variable` sweep target and variable-window parameter logging.
