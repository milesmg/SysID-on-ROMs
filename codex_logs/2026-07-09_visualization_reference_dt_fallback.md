### Files edited

- `Research_Code/src/Visualizations/optimization_visualizations.jl`
  - Added a stable explicit-Euler fallback timestep for old visualization replays that did not serialize `reference_dt`.
  - Updated true trajectory replay to use that stable timestep instead of coarse saved-output spacing.

- `Research_Code/src/Visualizations/optimization_visualizations_2D.jl`
  - Applied the same stable explicit-Euler fallback for the duplicate 2D visualization helper.
