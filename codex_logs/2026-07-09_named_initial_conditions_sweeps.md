### ADJUSTED: Add named initial conditions to HPC sweeps.

Files edited:
- `Research_Code/src/HPC/Tools/Sweeps/initial_conditions.jl`: added a compact registry of named pointwise Allen-Cahn initial-condition functions plus a materializer for the active 1D/2D grid.
- `Research_Code/src/HPC/Tools/Sweeps/sweep_params.py`: added aliases so `INITIAL_CONDITION`, `INITIAL_CONDITION_NAME`, `INIT_COND`, and `U0_NAME` map to the same sweep environment key.
- `Research_Code/src/HPC/Tools/hpc1_run_fom.slurm`: forwards `INITIAL_CONDITION` to the FOM Julia entrypoint.
- `Research_Code/src/HPC/Tools/hpc1_run_rom.slurm`: forwards `INITIAL_CONDITION` to the ROM Julia entrypoint.
- `Research_Code/src/HPC/Tools/Sweeps/hpc1_run_sweep.slurm`: includes `INITIAL_CONDITION` in per-combination sweep logs.
- `Research_Code/src/HPC/Tools/run_fom_hpc.jl`: parses, materializes, logs, and saves the named initial condition for FOM runs.
- `Research_Code/src/HPC/Tools/run_rom_hpc.jl`: parses, materializes, logs, and saves the named initial condition for ROM runs.
