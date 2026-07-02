### ADJUSTED: Document the default HPC threading update.
# HPC J8/B1 Defaults

Updated the hpc1 FOM/ROM defaults to use the benchmarked `JULIA_NUM_THREADS=8` and `JULIA_BLAS_THREADS=1` setup.

Files edited:

- `Research_Code/HPC_compatibility/hpc1_run_rom.slurm`
  - Changed the default Slurm CPU request to 8 cores.
  - Changed default ROM Julia/BLAS threading to J8/B1 while preserving environment overrides.
- `Research_Code/HPC_compatibility/hpc1_run_fom.slurm`
  - Changed the default Slurm CPU request to 8 cores.
  - Changed default FOM Julia/BLAS threading to J8/B1 while preserving environment overrides.
- `Research_Code/HPC_compatibility/hpc_common.jl`
  - Changed the shared BLAS fallback to 1 thread unless `JULIA_BLAS_THREADS` is explicitly set.
- `Research_Code/HPC_compatibility/README_hpc1.md`
  - Updated the speed note to document the J8/B1 default and the need to override CPU request and environment variables together when testing other thread counts.
