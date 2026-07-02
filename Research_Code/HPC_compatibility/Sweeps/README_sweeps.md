# FOM/ROM parameter sweeps

This directory runs Cartesian-product sweeps over the existing hpc1 FOM and ROM
training wrappers.

## Files

- `sweep_params.py`: parses a sweep text file and expands parameter combinations.
- `hpc1_run_sweep.slurm`: Slurm array wrapper; one array task runs one parameter combination.
- `submit_sweep.sh`: computes the array size and submits `hpc1_run_sweep.slurm`.
- `example_rom_sweep.txt`: ROM sweep example.
- `example_fom_sweep.txt`: FOM sweep example.

## Sweep file format

Use one `KEY = value` entry per line. Keys are converted to the environment
variables consumed by `hpc1_run_fom.slurm` and `hpc1_run_rom.slurm`.

```text
target = rom
RUN_NAME = ROM_sweep_example
ITERS = [[50,40,30],[30,20,10],[20,10,5]]
ETAS = [[1e-3,1e-4,1e-5],[2e-3,2e-4,2e-5]]
M = 3,4
```

This produces every possible combination of:

- 3 `ITERS` schedules
- 2 `ETAS` schedules
- 2 `M` values

for `3 * 2 * 2 = 12` Slurm array tasks.

Grouped optimizer schedules stay grouped. For example,
`ITERS = [[50,40,30],[30,20,10]]` gives two choices:

```text
ITERS=50,40,30
ITERS=30,20,10
```

It does not sweep `50`, `40`, and `30` independently.

## Submit

From the repository root on hpc1:

```bash
bash Research_Code/HPC_compatibility/Sweeps/submit_sweep.sh \
    Research_Code/HPC_compatibility/Sweeps/example_rom_sweep.txt
```

Optionally limit how many sweep tasks run at once:

```bash
MAX_CONCURRENT=4 bash Research_Code/HPC_compatibility/Sweeps/submit_sweep.sh \
    Research_Code/HPC_compatibility/Sweeps/example_rom_sweep.txt
```

Dry-run the computed `sbatch` command without submitting:

```bash
DRY_RUN=true bash Research_Code/HPC_compatibility/Sweeps/submit_sweep.sh \
    Research_Code/HPC_compatibility/Sweeps/example_rom_sweep.txt
```

## Inspect without submitting

Count jobs:

```bash
python3 Research_Code/HPC_compatibility/Sweeps/sweep_params.py count \
    Research_Code/HPC_compatibility/Sweeps/example_rom_sweep.txt
```

List all combinations:

```bash
python3 Research_Code/HPC_compatibility/Sweeps/sweep_params.py list \
    Research_Code/HPC_compatibility/Sweeps/example_rom_sweep.txt
```

Show the environment exports for one array task:

```bash
python3 Research_Code/HPC_compatibility/Sweeps/sweep_params.py env \
    Research_Code/HPC_compatibility/Sweeps/example_rom_sweep.txt 0
```

Dry-run one concrete array task without running training:

```bash
REPO_ROOT="$PWD" \
SWEEP_FILE=Research_Code/HPC_compatibility/Sweeps/example_rom_sweep.txt \
SLURM_ARRAY_TASK_ID=0 \
DRY_RUN=true \
bash Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm
```

Each sweep task appends the array index and parameter label to `RUN_NAME`, so
saved output directories do not collide.
