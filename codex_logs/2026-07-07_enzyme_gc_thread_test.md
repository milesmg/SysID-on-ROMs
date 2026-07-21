# Enzyme GC Thread Test

### ADJUSTED: Added the next stability test for the accelerated FOM crash reproducer.

- `Research_Code/Optimization/Data/Sweeps/FOM_hyperparam_opt/02_enzyme_gc1_reproducer_005x.txt`: repeats the same `WINDOW_T=0.05`, `WINDOW_N_OBS=2`, `ITERS=100` FOM reproducer five times with `JULIA_NUM_GC_THREADS=1`, keeping `JULIA_NUM_THREADS=8`.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`: prints `JULIA_NUM_GC_THREADS` in sweep logs so the threading condition is visible after sync.
