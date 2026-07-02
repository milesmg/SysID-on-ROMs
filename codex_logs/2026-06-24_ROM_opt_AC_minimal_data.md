# Minimal ROM optimization data flow

Reduced the ROM workflow to `ODEProblem → OptimizationProblem → output → save`. Runtime context now travels inside the ODE problem, while saving retains only parameter/evaluation histories and raw reconstruction data such as modes, DEIM indices, singular values, problem scalars, and observation times. Derived capture ratios, projection errors, grids, and reduced operators are no longer saved.
