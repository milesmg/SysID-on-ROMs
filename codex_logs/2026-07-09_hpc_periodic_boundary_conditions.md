Implemented optional periodic Allen-Cahn boundary conditions for the HPC FOM and ROM workflows while keeping homogeneous Dirichlet as the default.

Files edited:
- `Research_Code/src/HPC/integration_AC_hpc.jl`: added boundary-condition normalization, periodic 1D/2D in-place Laplacians, periodic sparse diffusion matrices, and boundary-aware neural/reference ODE setup.
- `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`: added `boundary_condition` to FOM preparation, run metadata, and neural ODE construction.
- `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`: added `boundary_condition` to ROM preparation and saved ROM metadata.
- `Research_Code/HPC_compatibility/hpc_common.jl`: added `boundary_condition` to reference construction and reference metadata.
- `Research_Code/HPC_compatibility/run_fom_hpc.jl`: parsed and forwarded `--boundary-condition`.
- `Research_Code/HPC_compatibility/run_rom_hpc.jl`: parsed and forwarded `--boundary-condition` into the reference, ROM matrix, and ROM setup.
- `Research_Code/HPC_compatibility/hpc1_run_fom.slurm`: forwarded `BOUNDARY_CONDITION` to Julia.
- `Research_Code/HPC_compatibility/hpc1_run_rom.slurm`: forwarded `BOUNDARY_CONDITION` to Julia.
- `Research_Code/HPC_compatibility/Sweeps/sweep_params.py`: added boundary-condition sweep aliases.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`: included `BOUNDARY_CONDITION` in sweep environment logging.
