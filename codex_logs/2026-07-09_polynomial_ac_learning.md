## Polynomial Allen-Cahn Learning

Added polynomial learned-nonlinearity support alongside the existing NN FOM/ROM path, then refactored the polynomial file into a small extension over the existing FOM/ROM helpers.

Files edited:
- `Research_Code/src/HPC/FOM_ROM_polynomial_learning_AC.jl`: added only polynomial evaluation, RHS, and FOM/ROM setup overrides; reused the existing loss, optimizer, ROM-builder, and save paths.
- `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`: added final polynomial coefficient metadata to the existing FOM save path.
- `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`: added polynomial model metadata to the existing ROM save path.
- `Research_Code/src/HPC/integration_AC_hpc.jl`: kept boundary-condition validation out of the RHS Laplacian hot path so polynomial adjoints do not trace string normalization.
- `Research_Code/HPC_compatibility/run_fom_hpc.jl`: added `--learner polynomial` / `--polynomial-degree` routing for FOM jobs.
- `Research_Code/HPC_compatibility/run_rom_hpc.jl`: added `--learner polynomial` / `--polynomial-degree` routing for ROM jobs.
- `Research_Code/HPC_compatibility/hpc1_run_fom.slurm`: forwarded `LEARNER` and `POLYNOMIAL_DEGREE` to the Julia FOM runner.
- `Research_Code/HPC_compatibility/hpc1_run_rom.slurm`: forwarded `LEARNER` and `POLYNOMIAL_DEGREE` to the Julia ROM runner.
- `Research_Code/HPC_compatibility/Sweeps/sweep_params.py`: added polynomial learner and degree aliases for sweep files.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`: included polynomial learner settings in sweep environment logging.
- `Research_Code/src/Visualizations/optimization_visualizations.jl`: added polynomial coefficient loading and learned-function/trajectory branches.
- `Research_Code/src/Visualizations/optimization_visualizations_2D.jl`: added the same polynomial visualization branches for 2D data.
- `Research_Code/src/Visualizations/optimization_metadata_visualizations.jl`: added polynomial model type and degree to serialized metadata summaries.

Validation:
- Loaded polynomial FOM setup and checked coefficient count.
- Ran a one-iteration polynomial FOM optimization smoke test with Mooncake adjoints.
- Loaded polynomial ROM setup with a valid tiny POD/DEIM basis.
- Loaded 1D and 2D visualization helpers and checked polynomial learned-function plotting.
- Checked sweep parser aliases for polynomial learner and degree.
