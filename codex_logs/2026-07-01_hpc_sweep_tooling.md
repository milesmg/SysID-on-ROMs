### ADJUSTED: Add hpc1 FOM/ROM sweep tooling.

Built sweep tooling under `Research_Code/HPC_compatibility/Sweeps` to expand line-oriented parameter files into Cartesian-product Slurm array tasks. Each task exports one concrete FOM/ROM parameter combination, creates a unique `RUN_NAME`, and delegates to the existing hpc1 FOM or ROM wrapper.

Files edited:

- `Research_Code/HPC_compatibility/Sweeps/sweep_params.py`
  - Added dependency-free sweep parsing.
  - Added grouped schedule handling for `ETAS`, `ITERS`, and `BETA`.
  - Added `count`, `target`, `list`, and `env` commands.
  - Added shell-safe environment export generation for one array task.

- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`
  - Added generic Slurm array wrapper for FOM/ROM sweeps.
  - Added per-task environment expansion from a sweep text file.
  - Added unique sweep run-name generation.
  - Added `DRY_RUN=true` support for safe task inspection.

- `Research_Code/HPC_compatibility/Sweeps/submit_sweep.sh`
  - Added submit helper that computes the array bounds from the sweep file.
  - Added optional `MAX_CONCURRENT` throttling.
  - Added `DRY_RUN=true` support for safe submission inspection.

- `Research_Code/HPC_compatibility/Sweeps/example_rom_sweep.txt`
  - Added ROM sweep example using nested grouped `ITERS` and `ETAS` schedules.

- `Research_Code/HPC_compatibility/Sweeps/example_fom_sweep.txt`
  - Added FOM sweep example.

- `Research_Code/HPC_compatibility/Sweeps/README_sweeps.md`
  - Documented sweep file format, submission, inspection, and dry-run commands.

- `codex_logs/2026-07-01_hpc_sweep_tooling.md`
  - Documented the sweep tooling changes.
