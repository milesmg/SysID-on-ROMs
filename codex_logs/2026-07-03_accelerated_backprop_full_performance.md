# Accelerated Backpropagation Full Performance Matrix

### ADJUSTED: Record the complete local package/tool benchmark, confirmation runs, CSV outputs, and conclusions.

## Scope

- Julia `1.12.6`, eight Julia threads, one BLAS thread.
- FOM size `N=256`, network `(1,8,8,1)`, mean variable-window loss.
- Workloads: `T={0.05,2.0}` crossed with `Nobs={10,50}`.
- Full matrix: three steady-state timing samples per successful row.
- Confirmation matrix: seven samples for the production baseline and leading candidates.
- Original solver mode: Lux with `TRBDF2()` and all eleven declared VJP choices.
- Finite solver mode: Lux and SimpleChains with `TRBDF2(autodiff=AutoFiniteDiff())` and all eleven VJP choices.

The full matrices produced 132 rows: 72 successful, 60 expected unsupported,
and zero unexpected failures. The confirmation runs added 32 rows: 24
successful, eight expected unsupported, and zero unexpected failures.

## Result

The fastest complete configuration was Lux with runtime-activity Enzyme and
finite-difference TRBDF2 state Jacobians:

| Rank | Network | VJP | Solver state Jacobian | Full-matrix geometric speedup |
| --- | --- | --- | --- | ---: |
| 1 | Lux | runtime Enzyme | finite difference | 51.28x |
| 2 | Lux | runtime Enzyme | production | 49.48x |
| 3 | SimpleChains | compiled ReverseDiff | finite difference | 33.74x |
| 4 | Lux | Mooncake | finite difference | 31.25x |

The independent seven-sample confirmation gave geometric speedups of `49.50x`,
`45.99x`, `31.68x`, and `30.94x`, respectively. The robust conclusion is that
runtime Enzyme is the important change. The two solver state-Jacobian modes are
close enough that their small ordering difference should be rechecked on the
cluster CPU before changing the production solver mode.

Seven-sample workload medians for the overall winner were:

| T | Nobs | Seconds/gradient | Speedup vs current production |
| ---: | ---: | ---: | ---: |
| 0.05 | 10 | 0.01718 | 38.58x |
| 0.05 | 50 | 0.06624 | 43.27x |
| 2.0 | 10 | 0.02111 | 60.57x |
| 2.0 | 50 | 0.07397 | 59.35x |

The current production configuration is Lux with compiled ReverseDiff. Its
confirmation medians were `0.6630`, `2.8666`, `1.2788`, and `4.3899` seconds
for the same workloads. Runtime Enzyme has a substantial one-time compilation
cost on its first invocation, but the steady-state savings repay that cost in
roughly tens to low hundreds of optimizer iterations, depending on `Nobs`.

All successful gradients matched compiled ReverseDiff with maximum relative
error `5.68e-15`. The largest independent directional finite-difference error
was `0.00101`.

## CSV Outputs

All outputs are under:

`Research_Code/Optimization/Data/BackpropBenchmarks/2026-07-03_full_performance_matrix`

- `all_experiment_results.csv`: every raw row from both full matrices and both confirmation runs; 164 rows.
- `all_results.csv`: the 132-row full matrix used for the primary ranking.
- `rankings.csv`: overall full-matrix ranking for 18 complete supported configurations.
- `workload_rankings.csv`: all 72 successful full-matrix rows ranked within workload.
- `confirmation_results.csv`: all 32 confirmation rows.
- `confirmation_rankings.csv`: overall seven-sample confirmation ranking.
- `confirmation_workload_rankings.csv`: confirmation rows ranked within workload.
- Each run directory also contains its original `benchmark_results.csv` and serialized metadata/results.

## Compatibility Findings

- `autojacvec=true` and Tracker are not implemented by this GaussAdjoint path.
- Default Enzyme fails on the existing Lux batch function; runtime activity fixes it.
- Mooncake requires finite-difference solver state Jacobians.
- SimpleChains requires finite-difference solver state Jacobians.
- SimpleChains does not compose with Tracker, Enzyme, or Mooncake in this path.
- Reactant requires a separate Reactant-array FOM port and was recorded as unsupported.

## Files Edited

- `Research_Code/HPC_compatibility/accelerated_backprop/accelerated_backprop_FOM_hpc.jl`: parameterized production versus finite solver state Jacobians and saved that mode per row.
- `Research_Code/HPC_compatibility/accelerated_backprop/run_accelerated_backprop_hpc.jl`: added `--solver-autodiff` parsing and logging.
- `Research_Code/HPC_compatibility/accelerated_backprop/hpc1_run_accelerated_backprop.slurm`: added `SOLVER_AUTODIFF` forwarding.
- `Research_Code/HPC_compatibility/accelerated_backprop/test_accelerated_backprop.jl`: added solver-mode construction checks.
- `Research_Code/HPC_compatibility/accelerated_backprop/rank_backprop_results.py`: added raw CSV combination, overall ranking, and per-workload ranking.
- `Research_Code/HPC_compatibility/accelerated_backprop/README.md`: documented solver-mode selection and result location.
- `codex_logs/2026-07-03_accelerated_backprop_full_performance.md`: added this experiment record.

## Verification

- Both solver-mode CLI smoke tests passed.
- Python compilation, Julia parsing, Slurm shell syntax, and `git diff --check` passed before the full runs.
- Both full matrices completed with zero unexpected failures.
- Both seven-sample confirmation runs completed with zero unexpected failures.
- Every generated aggregate CSV was read back with Python's CSV parser and row-count checked.
