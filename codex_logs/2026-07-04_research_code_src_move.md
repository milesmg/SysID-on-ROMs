# Research Code `src` Move

## Summary

Updated active local, HPC, Slurm-entrypoint, and notebook paths after moving
`Research_Code/helper_functions` to `Research_Code/src` and moving local
notebooks to `Research_Code/Optimization/Local`.

`Research_Code/HPC_compatibility/Old`, historical logs, and saved optimization
data were intentionally not edited.

## Files Edited

- `Research_Code/HPC_compatibility/run_fom_hpc.jl`
  - Loads FOM and integration code from `Research_Code/src/HPC`.
- `Research_Code/HPC_compatibility/run_rom_hpc.jl`
  - Loads ROM code from `Research_Code/src/HPC`.
- `Research_Code/HPC_compatibility/hpc_common.jl`
  - Loads `run_name_guard.jl` from `Research_Code/src/Misc.`.
- `Research_Code/HPC_compatibility/README_hpc1.md`
  - Points to the active `Julia/HPC_compatibility/Project.toml`.
- `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`
  - Uses the moved run-name guard.
- `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`
  - Uses the moved run-name guard.
- `Research_Code/src/Local/FOM_opt_AC.jl`
  - Uses the moved run-name guard and correct `Optimization/Data` root.
- `Research_Code/src/Local/ROM_opt_AC.jl`
  - Uses the moved run-name guard and correct `Optimization/Data` root.
- `Research_Code/src/Misc./run_name_guard.jl`
  - Resolves the shared data root correctly from its new location.
- `Research_Code/Optimization/Local/training_on_FOM.ipynb`
- `Research_Code/Optimization/Local/training_on_ROM.ipynb`
  - Load training helpers from `Research_Code/src`.
- `Research_Code/Optimization/Local/visualize_FOM.ipynb`
- `Research_Code/Optimization/Local/visualize_ROM.ipynb`
  - Load visualization helpers from `Research_Code/src` and saved runs from `Optimization/Data`.

## Tests

1. Parsed all 15 active Julia source/entrypoint files with `Meta.parseall`: passed.
2. Validated all four moved notebooks with `jq empty`: passed.
3. Included `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`: passed.
   - Variable-window API loaded and data root resolved to `Research_Code/Optimization/Data`.
4. Included `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`: passed.
   - Variable-window ROM API loaded and data root resolved correctly.
5. Included both local FOM and ROM helpers in separate Julia processes: passed.
   - Both resolved the shared data root correctly.
6. Ran standard FOM and ROM entrypoints with intentional schedule mismatches: passed.
   - Both loaded the moved source trees and reached expected schedule validation.
7. Searched active source, runner, Slurm/sweep, notebook, and README files for stale `helper_functions` references: none found.
8. Validated tracked edits with `git diff --check`: passed.
