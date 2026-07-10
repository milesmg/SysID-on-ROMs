# ROM Backend Test Sweeps

### ADJUSTED: Document the Mooncake ROM backend smoke suite and hpc1 sync/submit commands.

Small hpc1 ROM sweeps for checking the current SciMLSensitivity backend
(Mooncake in the HPC ROM path) without launching a large HPO run.

## Sweep

- `00_mooncake_rom_backend_smoke.txt`: 8 explicit cases covering 1D full
  trajectory, random short windows, batched windows, a staged schedule, seed
  repeats, and tiny 2D full/random-window jobs. Every optimizer stage runs at
  least 100 iterations.

## Sync to hpc1

From the repository root:

```bash
rsync -av \
  --filter='dir-merge .rsync-filter' \
  Research_Code/Optimization/Sweeps/ROM_backend_tests/ \
  hpc1:/home/mgantcher/Brookhaven/Research_Code/Optimization/Sweeps/ROM_backend_tests/

rsync -av \
  --filter='dir-merge .rsync-filter' \
  codex_logs/2026-07-07_rom_backend_test_sweep.md \
  hpc1:/home/mgantcher/Brookhaven/codex_logs/2026-07-07_rom_backend_test_sweep.md
```

## Submit on hpc1

From the repository root:

```bash
bash Research_Code/src/HPC/Tools/Sweeps/submit_sweep.sh \
  Research_Code/Optimization/Sweeps/ROM_backend_tests/00_mooncake_rom_backend_smoke.txt
```

Use fewer concurrent workers if hpc1 is busy:

```bash
MAX_CONCURRENT=4 bash Research_Code/src/HPC/Tools/Sweeps/submit_sweep.sh \
  Research_Code/Optimization/Sweeps/ROM_backend_tests/00_mooncake_rom_backend_smoke.txt
```

## Inspect locally or on hpc1

```bash
python3 Research_Code/src/HPC/Tools/Sweeps/sweep_params.py count \
  Research_Code/Optimization/Sweeps/ROM_backend_tests/00_mooncake_rom_backend_smoke.txt

python3 Research_Code/src/HPC/Tools/Sweeps/sweep_params.py list \
  Research_Code/Optimization/Sweeps/ROM_backend_tests/00_mooncake_rom_backend_smoke.txt
```

Dry-run one Slurm array worker without training:

```bash
REPO_ROOT="$PWD" \
SWEEP_FILE=Research_Code/Optimization/Sweeps/ROM_backend_tests/00_mooncake_rom_backend_smoke.txt \
SWEEP_TOTAL_COUNT=8 \
SWEEP_WORKER_COUNT=8 \
SLURM_ARRAY_TASK_ID=0 \
DRY_RUN=true \
bash Research_Code/src/HPC/Tools/Sweeps/hpc1_run_sweep.slurm
```
