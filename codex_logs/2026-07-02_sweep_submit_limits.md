# Sweep Submit Limits

### ADJUSTED: Updated sweep submission to respect hpc1's submit and concurrency limits.

- `Research_Code/HPC_compatibility/Sweeps/submit_sweep.sh`: defaults to 20 submitted workers and 8 concurrent workers while accounting for existing jobs.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`: lets workers process excess combinations sequentially.
- `Research_Code/HPC_compatibility/Sweeps/README_sweeps.md`: documents worker assignment and limit overrides.
