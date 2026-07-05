# Remove Legacy HPC Optimizers

## Summary

Removed the unused fixed-trajectory setup and optimization functions from the
HPC FOM and ROM helpers. The HPC tooling now exposes only the central
variable-window optimization paths; full-trajectory training remains available
through their default window settings.

## Files Edited

- `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`
  - Removed `set_up_optimization` and `run_full_optimization`.
- `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`
  - Removed `set_up_ROM_optimization` and `run_ROM_optimization`.

## Tests

1. Searched active HPC source and entrypoints for all four removed names: no references remain.
2. Parsed both edited Julia files with `Meta.parseall`: passed.
3. Loaded the HPC FOM helper and verified only `run_variable_window_optimization` remains: passed.
4. Loaded the HPC ROM helper and verified only `run_variable_window_ROM_optimization` remains: passed.
