## HPC 2D Dimension Support

Added dimension-aware 1D/2D Allen-Cahn HPC setup with default 1D behavior.

Files edited:
- `Research_Code/src/HPC/integration_AC_hpc.jl`: added dimension validation, 2D default initial condition, flattened 2D Laplacian, and dimension-aware sparse matrix/neural ODE setup.
- `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`: added `dimension` setup, initial-condition validation, 2D metadata, and dimension-aware loss weighting.
- `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`: added 2D metadata, dimension-aware loss weighting, and saved grid shape for visualization.
- `Research_Code/HPC_compatibility/hpc_common.jl`: added dimension-aware reference construction.
- `Research_Code/HPC_compatibility/run_fom_hpc.jl`: parsed and printed `--dimension`, and saved reference metadata.
- `Research_Code/HPC_compatibility/run_rom_hpc.jl`: parsed and printed `--dimension`, and used dimension-aware ROM diffusion matrices.
- `Research_Code/HPC_compatibility/hpc1_run_fom.slurm`: forwarded optional `DIMENSION`.
- `Research_Code/HPC_compatibility/hpc1_run_rom.slurm`: forwarded optional `DIMENSION`.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`: included `DIMENSION` in sweep logging.
- `Research_Code/HPC_compatibility/Sweeps/sweep_params.py`: added aliases for `DIMENSION`.
- `Research_Code/HPC_compatibility/README_hpc1.md`: documented 2D usage.
- `Research_Code/src/Visualizations/optimization_visualizations.jl`: added 2D trajectory and ROM mode visualization support.

Follow-up edit:
- `Research_Code/src/HPC/integration_AC_hpc.jl`: removed repeated hot-path state-length and dimension checks from Laplacian kernels; setup-time validation remains.
