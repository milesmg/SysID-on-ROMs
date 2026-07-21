### ADJUSTED: Restrict sweep initial-condition registry to current 2D notebook options.

Files edited:
- `Research_Code/src/HPC/Tools/Sweeps/2D_initial_conditions.jl`: replaced the broader registry with the current `test_ROM_stability.ipynb` 2D initial-condition names/functions and added a `DIMENSION=2` guard for named conditions.
- `Research_Code/src/HPC/Tools/Sweeps/initial_conditions.jl`: removed after renaming the registry.
- `Research_Code/src/HPC/Tools/run_fom_hpc.jl`: updated the FOM entrypoint include path to the renamed 2D registry.
- `Research_Code/src/HPC/Tools/run_rom_hpc.jl`: updated the ROM entrypoint include path to the renamed 2D registry.
