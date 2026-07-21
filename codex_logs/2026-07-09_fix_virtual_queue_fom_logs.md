### ADJUSTED: Fixed missing FOM out/err logs from virtual-queued sweeps.

Bug found:
- `hpc1_run_sweep.slurm` only relied on the Slurm array-worker `%x-%A_%a.out/.err` files after the reorg.
- It no longer created per-run logs keyed by `RUN_NAME`, so single-case FOM sweep logs were not appearing in the expected run/log locations.
- `_Logs` was also created inside the Slurm script body, which is too late for Slurm's own `#SBATCH -o/-e` open step.

Files edited:
- `Research_Code/src/HPC/Tools/Sweeps/hpc1_run_sweep.slurm`
  - Captures every sweep combination into `Research_Code/Optimization/Data/_Logs/<RUN_NAME>.out/.err`.
  - Moves successful run logs into `Research_Code/Optimization/Data/<RUN_NAME>/<RUN_NAME>.out/.err`.
  - Leaves failed run logs in `_Logs`.
  - Includes `LEARNER` and `POLYNOMIAL_DEGREE` in sweep environment logging.
- `Research_Code/src/HPC/Tools/Sweeps/submit_sweep.sh`
  - Creates `Research_Code/Optimization/Data/_Logs` before calling `sbatch`.
- `Research_Code/src/HPC/Tools/Sweeps/virtual_sweep_queue.sh`
  - Creates `Research_Code/Optimization/Data/_Logs` before calling `sbatch`.
- `Research_Code/src/HPC/Tools/hpc1_run_fom.slurm`
  - Forwards `LEARNER` and `POLYNOMIAL_DEGREE`.
- `Research_Code/src/HPC/Tools/hpc1_run_rom.slurm`
  - Forwards `LEARNER` and `POLYNOMIAL_DEGREE`.
- `Research_Code/src/HPC/Tools/run_fom_hpc.jl`
  - Restored `--learner polynomial` / `--polynomial-degree` routing after the path reorg.
- `Research_Code/src/HPC/Tools/run_rom_hpc.jl`
  - Restored `--learner polynomial` / `--polynomial-degree` routing after the path reorg.

Validation run:
- `bash -n` on moved FOM, ROM, sweep, submit, and virtual-queue scripts.
- Dry-ran the polynomial FOM sweep worker and verified `FOM_polynomial_best_hpo_0_best_fom_hpo_initial.out/.err` are created under `_Logs`.
- Checked polynomial FOM setup returns `learner == "polynomial"` and degree 3.
- Checked polynomial ROM setup returns `learner == "polynomial"` and degree 3.
- Corrected the successful-run log transfer from `cp` to `mv` so run-named logs are not duplicated in `_Logs`.
