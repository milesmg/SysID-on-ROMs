### ADJUSTED: Rewrote the solution-error/nonlinearity-error proxy notebook.

Files edited:
- `Research_Code/Optimization/Figures/soln_error_as_nonlinearity_error_proxy.ipynb`
  - Replaced scratch cells with a grouped FOM/ROM run analysis workflow.
  - Added explicit `MODEL_TYPE = FOM` / `MODEL_TYPE = ROM` configuration.
  - Added functions to load final NN parameters, compute nonlinearity L2 error, rank runs, plot loss-vs-error, and inspect a selected learned function.
- `codex_logs/2026-07-09_soln_error_nonlinearity_proxy_rewrite.md`
  - Logged this notebook rewrite.
