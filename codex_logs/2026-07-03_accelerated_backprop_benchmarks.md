# Accelerated Backpropagation Benchmarks

### ADJUSTED: Added isolated FOM benchmarks for Julia sensitivity backends and SimpleChains.

- `Julia/HPC_compatibility/Project.toml` and `Manifest.toml`: added Adapt, Mooncake, and SimpleChains.
- `Research_Code/HPC_compatibility/accelerated_backprop/accelerated_backprop_FOM_hpc.jl`: added backend selection, equivalent SimpleChains construction, expected compatibility reporting, pairwise and directional correctness checks, timings, and saved results.
- `Research_Code/HPC_compatibility/accelerated_backprop/run_accelerated_backprop_hpc.jl`: added the benchmark CLI entrypoint.
- `Research_Code/HPC_compatibility/accelerated_backprop/hpc1_run_accelerated_backprop.slurm`: added the repo-local Julia HPC wrapper.
- `Research_Code/HPC_compatibility/accelerated_backprop/test_accelerated_backprop.jl`: added full local network/backend matrix tests.
- `Research_Code/HPC_compatibility/accelerated_backprop/README.md`: documented scope, differences, tests, and execution.
- `codex_logs/2026-07-03_accelerated_backprop_test_results.md`: recorded all local test and diagnostic results, including expected failures.

## Publish status

The work is on local branch `2026-07-02-accelerated-backprop`. Nothing was staged,
committed, or pushed because this Codex sandbox cannot create `.git/index.lock`.
The GitHub CLI is also not installed. Existing unrelated working-tree changes were
left untouched.
