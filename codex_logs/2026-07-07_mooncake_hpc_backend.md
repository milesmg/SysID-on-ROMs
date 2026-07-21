# Mooncake HPC Backend

### ADJUSTED: Swapped the active HPC FOM/ROM sensitivity backend from runtime Enzyme to Mooncake while preserving finite-difference TRBDF2 solver autodiff.

## Files Edited

- `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`: load Mooncake, document the new default, and use `GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP())`.
- `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`: load Mooncake, document the new default, and use `GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP())`.
- `Research_Code/src/HPC/variable_window_common_hpc.jl`: update the shared default sensitivity backend and print `alg`/`sensalg` in optimization logs.
- `Research_Code/src/HPC/integration_AC_hpc.jl`: remove the old Enzyme backend load from the generic integration helper.
- `Research_Code/Optimization/Data/Sweeps/FOM_hyperparam_opt/00_timing_scaling_mooncake.txt`: add a Mooncake-named copy of the timing-per-iteration sweep.

## Verification

- Parsed all edited Julia files with `Meta.parseall`.
- Constructed `TRBDF2(autodiff=AutoFiniteDiff())` and `GaussAdjoint(autojacvec=SciMLSensitivity.MooncakeVJP())` under `Julia/HPC_compatibility`.
- Ran `git diff --check` on edited files.
