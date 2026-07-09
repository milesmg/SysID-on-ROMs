# HPC README Julia Depot Fix

### ADJUSTED: Updated the direct Julia call example to use the same repo-local depot and project as the hpc1 Slurm wrappers.

## Files Edited

- `Research_Code/HPC_compatibility/README_hpc1.md`: replaced the bare direct Julia command with a `JULIA_DEPOT_PATH="$PWD/Julia/depot:"` template and warned against bare `--project` calls for hpc1 diagnostics.
