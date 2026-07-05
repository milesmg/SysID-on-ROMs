# Remove Variable Window Schedule Helper

Removed `variable_window_schedule` and inlined scalar expansion plus complete per-stage vector validation in the shared variable-window optimization loop.

## Files edited

- `Research_Code/src/HPC/variable_window_common_hpc.jl`: removed the schedule helper, inlined schedule preparation, and documented that scalars broadcast while vectors require one entry per stage.

## Tests

- Confirmed no active references to `variable_window_schedule` remain outside the ignored `Research_Code/HPC_compatibility/Old` tree.
- Parsed `variable_window_common_hpc.jl` successfully with `Meta.parseall`.
- Confirmed a one-element vector fails schedule validation for a two-stage optimization.
- Confirmed scalar window settings broadcast successfully for a two-stage optimization.
- The first invalid-length test harness exposed Julia global soft-scope behavior; rerunning the same assertion inside a function passed. No production-code failure was involved.
