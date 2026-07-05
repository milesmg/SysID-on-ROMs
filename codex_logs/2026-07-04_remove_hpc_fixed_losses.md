# Remove Fixed-Trajectory HPC Losses

## Summary

Removed the unused fixed-observation FOM and ROM loss functions from the HPC
helpers. Variable-window losses remain unchanged.

## Files Edited

- `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`
  - Removed `loss_ref_F`.
- `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`
  - Removed `loss_ROM`.

## Tests

1. Searched active HPC source and entrypoints for both removed names: no references remain.
2. Parsed both edited Julia files with `Meta.parseall`: passed.
