# Move Successful HPC Logs

### ADJUSTED: Move successful FOM/ROM run logs into the saved run directory.

- `Research_Code/HPC_compatibility/hpc1_run_fom.slurm`: after a successful direct FOM run, moves the Slurm `.out` and `.err` files into `Research_Code/Optimization/Data/<RUN_NAME>`.
- `Research_Code/HPC_compatibility/hpc1_run_rom.slurm`: after a successful direct ROM run, moves the Slurm `.out` and `.err` files into `Research_Code/Optimization/Data/<RUN_NAME>`.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`: writes each sweep combination to per-run `.out`/`.err` files and moves those files into the successful run directory, leaving failed-run logs in the central HPC logs directory.
