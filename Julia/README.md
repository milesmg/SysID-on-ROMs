# Repo-local Julia for hpc1

This directory contains the Julia setup used by the hpc1 Slurm wrappers.

- `HPC_compatibility/Project.toml` and `Manifest.toml` are the locked Julia environment for the batch scripts.
- `depot/` is created on hpc1 by `setup_hpc_julia.sh` and stores packages/artifacts/compiled caches.
- `julia-1.12.6/` is downloaded/extracted into this repo by `setup_hpc_julia.sh`.

Run this once on hpc1 after syncing the repo:

```bash
cd ~/Brookhaven
bash Julia/setup_hpc_julia.sh
```

The HPC Slurm wrappers default to this directory, so ordinary `sbatch` commands use the same Julia binary, project, and depot every time. No Julia runtime or depot outside this repo is used.
