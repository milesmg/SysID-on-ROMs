# Accelerated Backpropagation Benchmarks

### ADJUSTED: Add an isolated benchmark path for FOM sensitivity backends and SimpleChains.

This directory reuses the variable-window FOM loss, ODE, reference construction,
and 1-8-8-1 Lux network. It varies only the network kernels and
`GaussAdjoint.autojacvec` setting.

## Compared backends

- automatic SciMLSensitivity selection
- finite-difference full Jacobian
- ForwardDiff full Jacobian
- ReverseDiff VJP, compiled and uncompiled
- Zygote VJP
- Tracker VJP
- Enzyme VJP, with default and runtime-activity modes
- Mooncake VJP
- Reactant VJP compatibility probe

Both the original Lux network and the same 1-8-8-1 network running through
Lux's `SimpleChainsLayer` adapter are tested. The SimpleChains parameter vector
is populated from the Lux parameters so both implementations start as the same
mathematical function.

The current local compatibility matrix is:

| Backend | Lux | SimpleChains |
| --- | --- | --- |
| automatic | supported | supported |
| finite-difference Jacobian | supported | supported |
| ForwardDiff Jacobian | unsupported by this GaussAdjoint implementation | unsupported |
| ReverseDiff | supported | supported |
| compiled ReverseDiff | supported | supported |
| Zygote | supported | supported |
| Tracker | unsupported by this GaussAdjoint implementation | unsupported |
| default Enzyme | requires runtime activity | unsupported |
| runtime Enzyme | supported | unsupported by SimpleChains pointer kernels |
| Mooncake | supported | unsupported by SimpleChains `llvmcall` kernels |
| Reactant | requires a Reactant-compiled FOM, not CPU `Array`s | same |

## Differences from FOM training

- No optimizer updates are performed.
- Compiled ReverseDiff, matching current FOM training, is the comparison gradient.
- A central directional finite difference independently checks gradient scale.
- `TRBDF2(autodiff=AutoFiniteDiff())` is used for internal state Jacobians because
  SimpleChains and Mooncake do not accept the default ForwardDiff dual states.
- `--solver-autodiff production` restores the original `TRBDF2()` behavior for
  matched Lux comparisons; the default remains `finite_diff`.
- Loss time, first-call time, steady-state gradient time, and allocations are saved separately.
- Results are written after each backend to preserve completed measurements.
- Known incompatibilities are `unsupported`; only unexpected failures return a nonzero status.

`first_call_seconds` is order-dependent in a multi-backend process. Run one VJP
per job when comparing cold compilation costs.

## Local test

```bash
julia --project=Julia/HPC_compatibility \
  Research_Code/HPC_compatibility/accelerated_backprop/test_accelerated_backprop.jl
```

## Local smoke benchmark

```bash
julia --project=Julia/HPC_compatibility \
  Research_Code/HPC_compatibility/accelerated_backprop/run_accelerated_backprop_hpc.jl \
  --N 8 --tfinal 0.02 --N-obs 1 \
  --window-T 0.01 --window-N-obs 1 --repeats 1 \
  --run-name accelerated_backprop_local_smoke
```

## HPC benchmark

From the repository root on hpc1:

```bash
sbatch Research_Code/HPC_compatibility/accelerated_backprop/hpc1_run_accelerated_backprop.slurm
```

Outputs are saved under `Research_Code/Optimization/Data/BackpropBenchmarks`.

The completed local `N=256` performance matrix and rankings are under
`Research_Code/Optimization/Data/BackpropBenchmarks/2026-07-03_full_performance_matrix`.
