## Neural PDE training scaffold

- Added a compact trainable neural PDE section under `### Codex Code` in `sensitivity_to_parameters.ipynb`.
- Generated one true 1D Allen-Cahn trajectory using `solve_forward_AC_1d` from `useful_functions.jl`, with sine initial data and `κ = 0.1`.
- Built a simple sigmoid neural time-stepper `N_θ(x_n)` that maps one state vector to the next.
- Implemented manual adjoint/backpropagation code for the trajectory `L2` objective:
  - rollout stores the intermediate neural states,
  - the trajectory adjoint moves backward through time,
  - each time step seeds a neural-network backward pass with `v_{n+1}`,
  - parameter gradients are accumulated over the window.
- Added SGD-style parameter updates with gradients computed before mutating the parameters.
- Added a curriculum:
  - one-step windows,
  - short rollout windows,
  - longer rollout windows,
  - full-trajectory training.
- Stored loss histories in `training_errors` and true/learned trajectory snapshots in `stored_trajectories`.

## Residual neural PDE update

- Updated `working_neural_PDE.ipynb` so the network predicts the time derivative `N_θ(x_n)` rather than the next state directly.
- Changed rollout to use `x_{n+1} = x_n + Δt N_θ(x_n)`.
- Adjusted the manual adjoint/backpropagation step:
  - NN parameter gradients are seeded with `Δt * v`,
  - the trajectory adjoint adds the identity contribution from the residual update.
