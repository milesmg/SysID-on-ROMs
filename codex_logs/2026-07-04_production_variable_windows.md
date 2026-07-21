# Central HPC Variable-Window Training

## Summary

Variable-window FOM and ROM training now comes from the existing
`FOM_opt_AC_hpc.jl` and `ROM_opt_AC_hpc.jl` helpers. The only new HPC helper is
`variable_window_common_hpc.jl`, which holds shared scheduling and staged-loop
code. Standard and sweep Slurm paths no longer depend on
`Variable_trajectory_length` or `accelerated_backprop`.

Both central optimizers default to:

```julia
alg = TRBDF2(autodiff=AutoFiniteDiff())
sensalg = GaussAdjoint(
    autojacvec=EnzymeVJP(
        mode=Enzyme.set_runtime_activity(Enzyme.Reverse),
    ),
)
```

Omitting window arguments gives every learning-rate stage the full trajectory,
`N_obs` observations, a beginning start, one window per iteration, and mean
loss.

## Files Edited

- `Research_Code/helper_functions/HPC/FOM_opt_AC_hpc.jl`
  - Owns FOM window loss, the public variable-window optimizer, accelerated defaults, and saving.
- `Research_Code/helper_functions/HPC/ROM_opt_AC_hpc.jl`
  - Owns ROM projection/window loss, the public variable-window optimizer, accelerated defaults, and saving.
- `Research_Code/helper_functions/HPC/variable_window_common_hpc.jl`
  - Shares deterministic schedule generation and the model-independent staged optimization loop.
- `Research_Code/HPC_compatibility/hpc_common.jl`
  - Parses comma-separated window policy schedules.
- `Research_Code/HPC_compatibility/run_fom_hpc.jl`
  - Parses, validates, prints, runs, and saves central FOM window schedules.
- `Research_Code/HPC_compatibility/run_rom_hpc.jl`
  - Parses, validates, prints, runs, and saves central ROM window schedules.
- `Research_Code/HPC_compatibility/hpc1_run_fom.slurm`
  - Forwards optional window environment variables.
- `Research_Code/HPC_compatibility/hpc1_run_rom.slurm`
  - Forwards optional window environment variables.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`
  - Routes legacy `fom_variable` targets through the standard FOM wrapper.
- `Research_Code/HPC_compatibility/Sweeps/README_sweeps.md`
  - Documents central FOM/ROM window schedules.

Removed the temporary production files
`variable_window_FOM_opt_AC_hpc.jl` and `variable_window_ROM_opt_AC_hpc.jl`.
The pre-existing experiment directories were not edited.

## Tests

1. Earlier standalone FOM helper load: passed.
2. Earlier two-stage differentiated FOM smoke test: passed.
   - `N=8`, `h=2`, stages `beginning,random`, seed `7`.
   - Random start `0.024045471694215607`; final validation loss `8.00960218727035e-5`.
3. Earlier two-stage differentiated ROM smoke test: passed.
   - `N=8`, `r=m=2`, `h=2`, same schedule and seed.
   - Random start `0.024045471694215607`; final validation loss `8.148718302663343e-5`.
4. Earlier deterministic schedule test initially failed because the one-line Julia test used ambiguous soft-scope assignment; no production code failed.
5. Function-scoped deterministic schedule rerun: passed.
   - Same seeds matched, different seeds differed, scalar broadcast worked, and invalid lengths failed.
6. Earlier standard FOM and ROM default inspection: passed; both recorded `EnzymeVJP`.
7. Final Julia parser test: passed for the common helper, central FOM/ROM helpers, common CLI parser, and both standard runners.
8. Shell syntax test: passed for FOM/ROM wrappers, sweep worker, and sweep submitter.
9. Final central FOM differentiated smoke test: passed.
   - Two stages with `window_T=[0.04,0.06]`, `window_N_obs=[2,3]`, and policies `beginning,random`.
   - Random start `0.024045471694215607`; final validation loss `8.00960218727035e-5`.
   - First compiled iteration `102.38 s`; second stage iteration `0.20 s`.
10. Final central ROM differentiated smoke test: passed with the same schedule.
    - Random start `0.024045471694215607`; final validation loss `8.148718302663343e-5`.
    - First compiled iteration `83.35 s`; second stage iteration `0.21 s`.
11. Standard FOM CLI schedule mismatch test: passed by failing before reference construction with all six schedule lengths in the error.
12. Existing timing sweep count: passed with `30` combinations.
13. Legacy `fom_variable` sweep dry run: passed and selected `hpc1_run_fom.slurm`.
14. Sweep parser Python compilation: passed.
15. Production dependency search: passed; standard runners, wrappers, sweep routing, and central helpers contain no experiment-directory references.
16. Edited-file whitespace check: passed.
