# Optional Slurm Arguments

## Summary

Removed model and training defaults from the FOM and ROM Slurm wrappers. Each
wrapper now starts with an empty `ARGS` array and forwards only explicitly set
environment variables. Defaults remain centralized in `run_fom_hpc.jl` and
`run_rom_hpc.jl`.

Runtime settings such as `INSTANTIATE`, `JULIA_NUM_THREADS`, and
`JULIA_BLAS_THREADS` retain their wrapper defaults.

## Files Edited

- `Research_Code/HPC_compatibility/hpc1_run_fom.slurm`
  - Removed FOM parameter defaults and consolidated all optional CLI forwarding.
- `Research_Code/HPC_compatibility/hpc1_run_rom.slurm`
  - Removed ROM parameter defaults and consolidated all optional CLI forwarding.

## Tests

1. `bash -n` on both Slurm wrappers: passed.
2. Searched both wrappers for the removed parameter-assignment block: no assignments remain.
3. Verified FOM and ROM argument blocks cover every corresponding Julia runner option, including window schedules and run names.
