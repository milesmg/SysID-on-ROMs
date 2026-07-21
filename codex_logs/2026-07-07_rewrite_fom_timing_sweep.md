# Rewrite FOM Timing Sweep

### ADJUSTED: Updated the timing-only FOM sweep for the current central variable-window and accelerated-backprop workflow.

- `Research_Code/Optimization/Data/Sweeps/FOM_hyperparam_opt/00_timing_scaling.txt`: changed the target from the legacy `fom_variable` route to `fom`, renamed the run prefix, increased each timing run to 100 iterations, disabled periodic parameter-history saves during the timing loop, and kept the same `WINDOW_T` by `WINDOW_N_OBS` grid.
