# Accelerated Backpropagation Test Results

### ADJUSTED: Record every local compatibility test, including diagnostic failures, for continuation by another agent.

## Environment

- Branch: `2026-07-02-accelerated-backprop`
- Julia: `1.12.6`
- Project: `Julia/HPC_compatibility`
- SciMLSensitivity: `7.112.1`
- SimpleChains: `0.4.8`
- Mooncake: `0.5.36`

## Package and adapter probes

1. Loaded SciMLSensitivity, Enzyme, Mooncake, SimpleChains, Adapt, and Lux from the HPC project.
   Result: pass after adding Adapt, Mooncake, and SimpleChains to the project and manifest.
2. Checked SciMLSensitivity VJP symbols and constructors.
   Result: `EnzymeVJP`, `TrackerVJP`, and `ReactantVJP` are exported; `MooncakeVJP` exists but is not exported and must be called as `SciMLSensitivity.MooncakeVJP()`.
3. Differentiated a direct SimpleChains network with Zygote.
   Result: failed with a stack overflow; this direct route was not used.
4. Adapted the existing Lux `Chain(1,8,8,1)` through `Lux.ToSimpleChainsAdaptor` and copied every Lux weight and bias into the SimpleChains parameter vector.
   Result: pass; Lux and adapted-SimpleChains function outputs differed by only `2.22e-16`.

## Parser and shell checks

1. `bash -n Research_Code/HPC_compatibility/accelerated_backprop/hpc1_run_accelerated_backprop.slurm`
   Result: pass before and after the final argument additions.
2. Parsed all three Julia files with `Meta.parseall`.
   Result: the original parse probe passed, but the first runtime test exposed an ambiguous generator expression in CSV output. Parenthesizing the generator fixed it.
3. Re-ran `Meta.parseall` on `accelerated_backprop_FOM_hpc.jl`, `run_accelerated_backprop_hpc.jl`, and `test_accelerated_backprop.jl` after the compatibility edits.
   Result: pass for all three files.

## Initial end-to-end matrix attempts

1. Ran `test_accelerated_backprop.jl` with `autojacvec=true` as the reference gradient.
   Result: failed before the matrix because SciMLSensitivity reports `autojacvec choice true is not supported by GaussAdjoint`.
2. Inspected the installed `gauss_adjoint.jl` implementation.
   Result: `false`, ReverseDiff, Zygote, Enzyme, Mooncake, and Reactant branches exist; `true` reaches the unsupported error; Tracker is marked TODO. This conflicts with the broad constructor documentation for `true` and Tracker.
3. Re-ran the matrix with `ReverseDiffVJP(true)` as a temporary reference and the production `TRBDF2()` solver.
   Lux results:
   - pass: automatic, finite-difference Jacobian, ReverseDiff, compiled ReverseDiff, and Zygote
   - unsupported/fail: ForwardDiff (`true`), Tracker (ForwardDiff/Tracker scalar ambiguity), default Enzyme (runtime-activity error), and the initially unqualified Mooncake name
   - successful Lux gradient relative errors were zero or approximately `1e-16`
   SimpleChains result: failed before its candidate loop because default TRBDF2 uses ForwardDiff dual state arrays, while SimpleChains dense kernels only accepted hardware scalar types.
   Overall test result: expected diagnostic failure, `1` pass and `1` error in `4m31.8s`.

## Targeted SimpleChains matrix

Configuration: `N=8`, `TFINAL=0.02`, `WINDOW_T=0.01`, `WINDOW_N_OBS=1`, and `TRBDF2(autodiff=AutoFiniteDiff())`.

- pass: automatic selection
- pass: finite-difference parameter Jacobian (`autojacvec=false`)
- pass: ReverseDiff uncompiled
- pass: ReverseDiff compiled
- pass: Zygote
- unsupported: ForwardDiff parameter Jacobian (`autojacvec=true`) because GaussAdjoint has no implementation branch
- unsupported: Tracker because SimpleChains cannot convert `Tracker.TrackedVector` to its pointer representation
- unsupported: default Enzyme because SimpleChains heap-memory pointer code triggers an activity error
- unsupported: runtime-activity Enzyme; the same pointer-based SimpleChains code remains incompatible
- unsupported: Mooncake because SimpleChains uses an `llvmcall` intrinsic that Mooncake cannot translate

Every successful pairing returned loss `1.1844514809258901e-5` and gradient norm `8.148129432824192e-5`.

## Targeted Lux probes

Configuration: `N=8`, `TFINAL=0.02`, `WINDOW_T=0.01`, and `WINDOW_N_OBS=1`.

1. Lux plus `EnzymeVJP(mode=Enzyme.set_runtime_activity(Enzyme.Reverse))` with production TRBDF2.
   Result: pass; loss `1.1844514809258901e-5`, gradient norm `8.148129432824192e-5`.
2. Lux plus Mooncake with production TRBDF2.
   Result: failed because Mooncake values could not be converted from the solver's ForwardDiff dual state arrays.
3. Lux plus Mooncake with `TRBDF2(autodiff=AutoFiniteDiff())`.
   Result: pass; loss `1.1844514809258901e-5`, gradient norm `8.148129432824192e-5`.

## Final matrix

`julia --project=Julia/HPC_compatibility Research_Code/HPC_compatibility/accelerated_backprop/test_accelerated_backprop.jl`

Result: pass, `51/51` assertions in `10m04.9s`.

- Successful pairings: 12 of 22.
- Expected unsupported pairings: 10 of 22.
- Every successful gradient matched compiled ReverseDiff with relative error at or below approximately `2e-16`.
- Every successful gradient passed the independent directional finite-difference check with relative error approximately `2.05e-6`.
- The CSV and both serialized output files were created in the temporary test directory.

## CLI entrypoint

1. Invoked the smoke benchmark through the `juliaup` launcher.
   Result: failed before Julia started because the sandbox denied creation of a juliaup configuration lockfile. No repository code ran.
2. Re-ran the identical command with the installed Julia 1.12.6 binary directly:

   ```bash
   /Users/milesgantcher/.julia/juliaup/julia-1.12.6+0.aarch64.apple.darwin14/bin/julia \
     --project=Julia/HPC_compatibility \
     Research_Code/HPC_compatibility/accelerated_backprop/run_accelerated_backprop_hpc.jl \
     --N 6 --tfinal 0.01 --N-obs 1 \
     --window-T 0.005 --window-N-obs 1 \
     --networks lux --vjps reverse_diff_compiled --repeats 1 \
     --run-name cli_smoke --data-root /tmp/brookhaven_backprop_cli_smoke_direct
   ```

   Result: pass. The CLI reported one successful combination, zero unsupported combinations, and zero unexpected failures. Relative gradient error was `0.0`; directional error was `2.34e-7`. CSV, result JLS, and metadata JLS files were all created.

## Slurm wrapper dry run

```bash
REPO_ROOT="$PWD" DRY_RUN=true INSTANTIATE=false \
  bash Research_Code/HPC_compatibility/accelerated_backprop/hpc1_run_accelerated_backprop.slurm
```

Result: pass. The wrapper emitted all default network, VJP, window, observation, repeat, and directional-step arguments and exited before invoking Julia.

## Final static checks

- `git diff --check`: pass.
- `bash -n` on the Slurm wrapper: pass.
- `Meta.parseall` on all three Julia source/test files: pass.
- CLI smoke CSV: pass; exactly one header and one result row with all timing, allocation, status, and directional-check columns present.
