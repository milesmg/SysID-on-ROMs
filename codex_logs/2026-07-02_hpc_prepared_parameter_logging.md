### ADJUSTED: Documented FOM/ROM prepared-parameter logging updates.
# HPC prepared parameter logging

Added parameter blocks to the FOM and ROM HPC entrypoints so Slurm logs show the resolved run settings after optimization setup and before training begins.

Files edited:

- `Research_Code/HPC_compatibility/run_fom_hpc.jl`
  - Added a prepared FOM parameter log block after `set_up_optimization`.
- `Research_Code/HPC_compatibility/run_rom_hpc.jl`
  - Added a prepared ROM parameter log block after `set_up_ROM_optimization`.
- `codex_logs/2026-07-02_hpc_prepared_parameter_logging.md`
  - Added this edit log.
