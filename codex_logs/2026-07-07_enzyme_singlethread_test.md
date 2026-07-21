# Enzyme Single-Thread Test

### ADJUSTED: Added a five-run single-thread reproducer for the accelerated FOM crash case.

- `Research_Code/Optimization/Data/Sweeps/FOM_hyperparam_opt/03_enzyme_singlethread_reproducer_005x.txt`: repeats the `WINDOW_T=0.05`, `WINDOW_N_OBS=2`, `ITERS=100` FOM reproducer five times with `JULIA_NUM_THREADS=1`, `JULIA_NUM_GC_THREADS=1`, and `JULIA_BLAS_THREADS=1`.
