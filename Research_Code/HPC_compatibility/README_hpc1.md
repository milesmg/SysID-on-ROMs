# hpc1 batch training for Allen--Cahn FOM/ROM

This directory contains batch entrypoints for running the notebook workflows on hpc1 without VS Code/Jupyter.

## Files

### ADJUSTED: Point to the repo-local Julia environment after removing the duplicate project.
- `Julia/HPC_compatibility/Project.toml`: Julia environment for the batch scripts.
- `setup_hpc_project.jl`: instantiates/precompiles the Julia environment.
- `hpc_common.jl`: shared CLI parsing, reference-solve setup, and staged learning-rate helpers.
- `run_fom_hpc.jl`: Julia entrypoint for full-order-model NN training.
- `run_rom_hpc.jl`: Julia entrypoint for ROM NN training.
- `hpc1_run_fom.slurm`: one FOM training job.
- `hpc1_run_rom.slurm`: one ROM training job.
- `hpc1_gpu_template.slurm`: GPU allocation template only; see GPU note below.

## Recommended first run

### ADJUSTED: Document the repo-local Julia setup used by hpc1 Slurm wrappers.
After syncing the repo to hpc1, build the repo-local Julia environment once:

```bash
bash Julia/setup_hpc_julia.sh
```

This uses `Julia/HPC_compatibility` as the Julia project, `Julia/depot` as the
writable package depot, and `Julia/julia-1.12.6/bin/julia` as the Julia runtime.
The Julia runtime and depot are kept inside this repo.

Submit from the repository root on hpc1:

```bash
sbatch Research_Code/HPC_compatibility/hpc1_run_rom.slurm
```

or:

```bash
sbatch Research_Code/HPC_compatibility/hpc1_run_fom.slurm
```

The scripts default to staged Adam:

```text
η = 5e-3 for 500 iterations
η = 1e-3 for 1000 iterations
η = 1e-4 for 900 iterations
```

Outputs are saved by the existing save functions under:

```text
Research_Code/Optimization/Data/<run_name>
```

Logs go to:

```text
Research_Code/HPC_compatibility/logs
```

## Override parameters

Use environment variables before `sbatch`. The Slurm wrapper scripts translate
these environment variables into Julia command-line flags.

Example ROM run:

```bash
R=20 \
M=20 \
ETAS="5e-3,1e-3,1e-4" \
ITERS="500,1000,900" \
RUN_NAME="ROM_r20_m20_staged" \
sbatch Research_Code/HPC_compatibility/hpc1_run_rom.slurm
```

Example FOM run:

```bash
ETAS="5e-3,1e-3,1e-4" \
ITERS="500,1000,900" \
RUN_NAME="FOM_staged_01" \
sbatch Research_Code/HPC_compatibility/hpc1_run_fom.slurm
```

Useful variables:

```text
N=256
L=1.0
DIMENSION=1        # set DIMENSION=2 for flattened N x N 2D runs
EPS2=1e-2
K=1.0
TFINAL=2.0
N_OBS=10
H=8
SEED=1
R=10              # ROM only
M=10              # ROM only
ETAS=5e-3,1e-3,1e-4
ITERS=500,1000,900
BETA=0.9,0.99
SAVE_FREQUENCY=10
PRINT_FREQUENCY=50
REFERENCE_DT_FACTOR=0.5
```

### ADJUSTED: Document the 2D Allen-Cahn HPC flag.
2D FOM and ROM jobs use the same wrappers and Julia entrypoints. Set
`DIMENSION=2`; `N` is then the number of interior grid points per axis, and the
ODE state is a flattened `N x N` grid. If no initial condition is supplied, the
code uses a default 2D disk profile. Any supplied initial condition must match
the requested dimension: length `N` for 1D, or length `N^2` / shape `N x N` for
2D.

Example 2D FOM run:

```bash
DIMENSION=2 \
N=64 \
RUN_NAME="FOM_2D_N64" \
sbatch Research_Code/HPC_compatibility/hpc1_run_fom.slurm
```

The Slurm wrappers intentionally use `Julia/julia-1.12.6/bin/julia`,
`Julia/HPC_compatibility`, and `Julia/depot` from this repo. Do not pass
`JULIA_EXE` or `JULIA_DEPOT_PATH` for normal hpc1 runs.

### ADJUSTED: Direct Julia calls must use the same repo-local depot as Slurm.
If you call the Julia entrypoints directly instead of going through the Slurm
wrappers, set the same repo-local depot and project used by the wrappers.
Arguments must be named flags. Both `--key value` and `--key=value` forms are
accepted; positional arguments are ignored by the parser.

Do not use a bare `Julia/julia-1.12.6/bin/julia --project=Julia/HPC_compatibility`
command for hpc1 diagnostics or direct runs: without `JULIA_DEPOT_PATH`, Julia
will use the interactive user depot instead of `Brookhaven/Julia/depot`.

Example direct Julia call:

```bash
cd /home/mgantcher/Brookhaven

JULIA_DEPOT_PATH="$PWD/Julia/depot:" \
JULIA_PKG_DEVDIR="$PWD/Julia/depot/dev" \
JULIA_PKG_USE_CLI_GIT=true \
./Julia/julia-1.12.6/bin/julia --project="$PWD/Julia/HPC_compatibility" \
    Research_Code/HPC_compatibility/run_rom_hpc.jl \
    --N 256 \
    --r 20 \
    --m 20 \
    --etas 5e-3,1e-3,1e-4 \
    --iters 500,1000,900 \
    --run-name ROM_r20_m20_staged
```

## CPU/GPU note

The current FOM/ROM code is CPU-oriented:

- the ODE states and Lux parameters stay on ordinary Julia CPU arrays;
- the RHS uses scalar loops;
- the sensitivity stack uses `GaussAdjoint(autojacvec=ReverseDiffVJP(true))`;
- nothing is moved to `CuArray`.

So requesting a P100 GPU will not speed up the current scripts. The best current use of hpc1 is CPU nodes with enough cores and memory.

The `hpc1_gpu_template.slurm` file is included only as a starting point for a future CUDA port. A real GPU port would need the RHS, neural-network call path, and adjoint path tested with CUDA arrays.

## Notes on speed

For a single training run, these scripts use:

```bash
### ADJUSTED: Default to the benchmarked J8/B1 threading configuration.
JULIA_NUM_THREADS=8
JULIA_BLAS_THREADS=1
OPENBLAS_NUM_THREADS=$JULIA_BLAS_THREADS
MKL_NUM_THREADS=$JULIA_BLAS_THREADS
```

This keeps Julia task/thread parallelism available while avoiding BLAS oversubscription for the current small ROM/FOM linear algebra. Override both the Slurm CPU request and the environment variables together when testing other thread counts.
