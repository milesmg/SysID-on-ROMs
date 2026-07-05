# Rename Windows Per Iter

Renamed the trajectory-window batching setting to `batch_size` throughout the active FOM, ROM, CLI, Slurm, and sweep paths. Historical logs and generated job output remain unchanged.

## Files edited

- `Research_Code/src/HPC/variable_window_common_hpc.jl`: renamed optimizer arguments, schedules, logging, and saved settings.
- `Research_Code/src/HPC/FOM_opt_AC_hpc.jl`: renamed the FOM keyword argument.
- `Research_Code/src/HPC/ROM_opt_AC_hpc.jl`: renamed the ROM keyword argument.
- `Research_Code/HPC_compatibility/run_fom_hpc.jl`: renamed parsing, validation, printing, and forwarding to `batch_size`/`--batch-size`.
- `Research_Code/HPC_compatibility/run_rom_hpc.jl`: renamed parsing, validation, printing, and forwarding to `batch_size`/`--batch-size`.
- `Research_Code/HPC_compatibility/hpc1_run_fom.slurm`: renamed the environment variable and CLI flag to `BATCH_SIZE`/`--batch-size`.
- `Research_Code/HPC_compatibility/hpc1_run_rom.slurm`: renamed the environment variable and CLI flag to `BATCH_SIZE`/`--batch-size`.
- `Research_Code/HPC_compatibility/Sweeps/sweep_params.py`: renamed the grouped sweep key to `BATCH_SIZE`.
- `Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm`: renamed parameter logging to `BATCH_SIZE`.
- `Research_Code/HPC_compatibility/Sweeps/README_sweeps.md`: documented the renamed sweep setting.

## Tests

- Confirmed Julia parsing for the shared helper, FOM/ROM helpers, and both entrypoints.
- Confirmed Python compilation for `sweep_params.py` and shell parsing for the three edited Slurm scripts.
- Confirmed no old-name references remain in active source, sweep configuration, or documentation; historical logs were intentionally preserved.
- Confirmed a two-stage window schedule builds batches of sizes two and three with the expected flattened history length.
- Confirmed the ROM helper loads with the renamed optimization interface.
- Confirmed sweep expansion emits `BATCH_SIZE`, including grouped schedules and the `batch_size` run label.
- Confirmed FOM and ROM Slurm wrappers forward `BATCH_SIZE` as `--batch-size`.
- The first wrapper harness consumed its own `%s` placeholder while creating a fake Julia executable; the corrected harness passed. No production-code failure was involved.
