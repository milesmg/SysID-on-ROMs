# ROM optimization workflow

Replaced the global-dependent notebook fragments in `ROM_opt_AC.jl` with a sequential prepare, set-up, run, and save workflow. The saved run metadata now includes the POD/DEIM modes, singular-value diagnostics, mode counts, reduced operators, and full-trajectory projection errors. Verified the workflow with a small Allen–Cahn solve, `ReverseDiffVJP(true)` gradient step, Fnn evaluation counting, and save/readback test.
