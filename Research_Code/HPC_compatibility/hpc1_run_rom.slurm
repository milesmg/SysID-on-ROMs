#!/bin/bash
#SBATCH -J ac_rom # slurm job name
#SBATCH -p cpu #where to submit the job 
#SBATCH -N 1 # 1 node
### ADJUSTED: Default ROM jobs to the measured J8/B1 threading setup.
#SBATCH --cpus-per-task=8 # cores per julia process.
#SBATCH --mem=128G #gb of ram on the node. There is a lot more than this available. s
#SBATCH -t 35:59:00 # run for 36 hours max (rules)
#SBATCH -o Research_Code/HPC_compatibility/logs/%x-%j.out # output
#SBATCH -e Research_Code/HPC_compatibility/logs/%x-%j.err # err

set -euo pipefail # fail immediately 

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}" 
HPC_DIR="$REPO_ROOT/Research_Code/HPC_compatibility"
### ADJUSTED: Use the repo-local Julia runtime/project defaults for hpc1 jobs.
JULIA_ROOT="$REPO_ROOT/Julia"
HPC_JULIA_PROJECT="$JULIA_ROOT/HPC_compatibility"
JULIA_EXE="$JULIA_ROOT/julia-1.12.6/bin/julia"

cd "$REPO_ROOT"
mkdir -p "$HPC_DIR/logs" "$REPO_ROOT/Research_Code/Optimization/Data"

### ADJUSTED: Fail fast if the repo-local Julia runtime has not been built.
if [ ! -x "$JULIA_EXE" ]; then
    echo "Missing repo-local Julia runtime: $JULIA_EXE"
    echo "Run: bash Julia/setup_hpc_julia.sh"
    exit 1
fi

### ADJUSTED: Keep Julia's bundled depots available while using the repo-local HPC depot.
JULIA_USER_DEPOT="$JULIA_ROOT/depot"
export JULIA_DEPOT_PATH="$JULIA_USER_DEPOT:"
export JULIA_PKG_DEVDIR="$JULIA_USER_DEPOT/dev"
export JULIA_PKG_USE_CLI_GIT=true
### ADJUSTED: Default ROM training to 8 Julia threads and 1 BLAS thread.
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export JULIA_BLAS_THREADS="${JULIA_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$JULIA_BLAS_THREADS"
export MKL_NUM_THREADS="$JULIA_BLAS_THREADS"

mkdir -p "$JULIA_USER_DEPOT" "$JULIA_PKG_DEVDIR"

if [ "${INSTANTIATE:-true}" = "true" ]; then
    ### ADJUSTED: Print Julia runtime paths before package setup for HPC debugging.
    echo "JULIA_EXE=$JULIA_EXE"
    echo "HPC_JULIA_PROJECT=$HPC_JULIA_PROJECT"
    echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
    "$JULIA_EXE" -e 'println("DEPOT_PATH=", DEPOT_PATH); println("LOAD_PATH=", LOAD_PATH)'
    "$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$HPC_DIR/setup_hpc_project.jl"
fi

### ADJUSTED: Forward only explicitly supplied ROM settings and keep defaults in Julia.
ARGS=()
if [ -n "${N:-}" ]; then ARGS+=(--N "$N"); fi
if [ -n "${L:-}" ]; then ARGS+=(--L "$L"); fi
if [ -n "${EPS2:-}" ]; then ARGS+=(--eps2 "$EPS2"); fi
if [ -n "${K:-}" ]; then ARGS+=(--k "$K"); fi
if [ -n "${TFINAL:-}" ]; then ARGS+=(--tfinal "$TFINAL"); fi
if [ -n "${N_OBS:-}" ]; then ARGS+=(--N-obs "$N_OBS"); fi
if [ -n "${R:-}" ]; then ARGS+=(--r "$R"); fi
if [ -n "${M:-}" ]; then ARGS+=(--m "$M"); fi
if [ -n "${H:-}" ]; then ARGS+=(--h "$H"); fi
if [ -n "${SEED:-}" ]; then ARGS+=(--seed "$SEED"); fi
if [ -n "${REFERENCE_DT_FACTOR:-}" ]; then ARGS+=(--reference-dt-factor "$REFERENCE_DT_FACTOR"); fi
if [ -n "${ETAS:-}" ]; then ARGS+=(--etas "$ETAS"); fi
if [ -n "${ITERS:-}" ]; then ARGS+=(--iters "$ITERS"); fi
if [ -n "${BETA:-}" ]; then ARGS+=(--beta "$BETA"); fi
if [ -n "${SAVE_FREQUENCY:-}" ]; then ARGS+=(--save-frequency "$SAVE_FREQUENCY"); fi
if [ -n "${PRINT_FREQUENCY:-}" ]; then ARGS+=(--print-frequency "$PRINT_FREQUENCY"); fi
if [ -n "${WARMUP:-}" ]; then ARGS+=(--warmup "$WARMUP"); fi
if [ -n "${WINDOW_T:-}" ]; then ARGS+=(--window-T "$WINDOW_T"); fi
if [ -n "${WINDOW_N_OBS:-}" ]; then ARGS+=(--window-N-obs "$WINDOW_N_OBS"); fi
if [ -n "${WINDOW_START_POLICY:-}" ]; then ARGS+=(--window-start-policy "$WINDOW_START_POLICY"); fi
# ### ADJUSTED: Forward trajectory-window batch size under the renamed interface.
if [ -n "${BATCH_SIZE:-}" ]; then ARGS+=(--batch-size "$BATCH_SIZE"); fi
if [ -n "${LOSS_NORMALIZATION:-}" ]; then ARGS+=(--loss-normalization "$LOSS_NORMALIZATION"); fi
if [ -n "${WINDOW_SEED:-}" ]; then ARGS+=(--window-seed "$WINDOW_SEED"); fi
if [ -n "${RUN_NAME:-}" ]; then ARGS+=(--run-name "$RUN_NAME"); fi

echo "[$(date)] Starting ROM training on $HOSTNAME"
"$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$HPC_DIR/run_rom_hpc.jl" "${ARGS[@]}"
echo "[$(date)] Finished ROM training"
