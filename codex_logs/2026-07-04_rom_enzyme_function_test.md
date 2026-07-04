# ROM Runtime-Enzyme Function Test

### ADJUSTED: Document the focused ROM compatibility test for the selected accelerated setup.

- `Research_Code/HPC_compatibility/Variable_trajectory_length/test_rom_enzyme_setup.jl`: added a small production-structured POD/DEIM ROM test using `TRBDF2(autodiff=AutoFiniteDiff())` and runtime-activity `EnzymeVJP`; checks loss, gradient, a directional finite difference, and one Adam iteration.

Initial local test: the ROM loss, runtime-Enzyme gradient, directional check,
and optimizer execution passed, but `N_iter=1` performed no Adam update and the
parameter-change assertion failed. The test was adjusted to `N_iter=2` so it
checks one actual optimizer update.

Final local test:

```bash
JULIA_NUM_THREADS=8 JULIA_BLAS_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  julia --project=Julia/HPC_compatibility \
  Research_Code/HPC_compatibility/Variable_trajectory_length/test_rom_enzyme_setup.jl
```

- Result: pass, `6/6` assertions in `1m30.6s`.
- The ROM loss and all gradient entries were finite, with a nonzero gradient norm.
- The gradient passed the independent directional finite-difference check with relative error below `1e-2`.
- The existing ROM Adam path applied a nonzero parameter update.
- Loss decreased from `1.4465678826320612e-4` to `1.3883560790179918e-4` after the update.
