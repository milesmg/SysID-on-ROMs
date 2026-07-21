# Inline Window Values

Removed the window-policy, loss-normalization, and observation-time helper functions. String settings now remain exactly as supplied, while positive observation-count validation and observation-time construction occur directly in `make_window_spec`.

## Files edited

- `Research_Code/src/HPC/variable_window_common_hpc.jl`: removed three small helpers and inlined their required behavior.

## Tests

- Confirmed the shared helper parses and both FOM and ROM helper files load successfully.
- Confirmed three observations over `[0.2, 0.8]` remain `[0.4, 0.6, 0.8]`.
- Confirmed zero observations fail the retained one-line positive-count check.
- Confirmed no active source references to the three removed helper names remain.
- The first zero-count test harness exposed Julia global soft-scope behavior; rerunning the assertion inside a function passed. No production-code failure was involved.
