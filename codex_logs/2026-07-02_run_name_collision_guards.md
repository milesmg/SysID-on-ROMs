### ADJUSTED: Add early run-name collision guards.

Added shared run-name collision checks so FOM/ROM runs fail before reusing an existing `Optimization/Data/<run_name>` directory. HPC entrypoints now check immediately after parsing `run_name`, before runtime setup, reference solves, and optimization setup. Local training notebooks define and check `run_name` before their setup cells, and save helpers reject existing directories as a final guard.

Files edited:

- `Research_Code/helper_functions/run_name_guard.jl`
  - Added `optimization_data_root`.
  - Added `assert_run_name_available`.

- `Research_Code/HPC_compatibility/hpc_common.jl`
  - Included the shared run-name guard helper for HPC entrypoints.

- `Research_Code/HPC_compatibility/run_fom_hpc.jl`
  - Added an early `assert_run_name_available(run_name)` check before runtime and reference-solve setup.

- `Research_Code/HPC_compatibility/run_rom_hpc.jl`
  - Added an early `assert_run_name_available(run_name)` check before runtime and reference-solve setup.

- `Research_Code/helper_functions/FOM_opt_AC.jl`
  - Included the shared guard helper.
  - Changed `save_optimization_data` to fail if the target run directory already exists.

- `Research_Code/helper_functions/ROM_opt_AC.jl`
  - Included the shared guard helper.
  - Changed `save_ROM_optimization_data` to fail if the target run directory already exists.

- `Research_Code/helper_functions/HPC/FOM_opt_AC_hpc.jl`
  - Included the shared guard helper.
  - Changed `save_optimization_data` to fail if the target run directory already exists.

- `Research_Code/helper_functions/HPC/ROM_opt_AC_hpc.jl`
  - Included the shared guard helper.
  - Changed `save_ROM_optimization_data` to fail if the target run directory already exists.

- `Research_Code/Optimization/training_on_FOM.ipynb`
  - Added an early guarded `run_name` cell before local FOM setup.
  - Updated the save cell to use `run_name`.
  - Cleared stale outputs.

- `Research_Code/Optimization/training_on_ROM.ipynb`
  - Added an early guarded `run_name` cell before local ROM setup.
  - Updated the save cell to use `run_name`.
  - Cleared stale outputs.

- `codex_logs/2026-07-02_run_name_collision_guards.md`
  - Documented the collision-guard changes.
