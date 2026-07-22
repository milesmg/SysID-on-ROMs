# Architecture

`src/` contains the simulation and training code.

- `Core/`: grids, learners, losses, reduction, saving, CLI parsing, and staged window training, as well as named structure definitions.
- `Equations/`: Allen-Cahn, Cahn-Hilliard, and reaction-diffusion adapters.
- `Tools/Slurm/`: generic cluster launcher and sweep/virtual-queue scripts.
- `Tools/Misc/`: logging and queue state.
- `Tools/Julia/`: Julia setup and environment validation helpers.
- `Tools/Tests/`: direct Julia test suite and reusable timing benchmarks; local timing-test results are saved by test name in `Untracked/Tests/timing_test_results/`.
- `Tools/Visualizations/`: reusable saved-run reconstruction, trajectory plotting/GIF, learned-function, and ROM-stability helpers used by local notebooks.
- `run.jl`: entrypoint; pass `--mode fom` or `--mode rom`.

`Julia/` is the pinned Julia environment. 

`Data/` contains run output, Slurm logs, and sweep configurations.
