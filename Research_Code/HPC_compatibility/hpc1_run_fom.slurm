#!/bin/bash
# see hpc1_run_rom.slurm for comments 
#SBATCH -J ac_fom
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 35:59:00
#SBATCH -o Research_Code/HPC_compatibility/logs/%x-%j.out
#SBATCH -e Research_Code/HPC_compatibility/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
HPC_DIR="$REPO_ROOT/Research_Code/HPC_compatibility"
JULIA_ROOT="$REPO_ROOT/Julia"
HPC_JULIA_PROJECT="$JULIA_ROOT/HPC_compatibility"
JULIA_EXE="$JULIA_ROOT/julia-1.12.6/bin/julia"

cd "$REPO_ROOT"
mkdir -p "$HPC_DIR/logs" "$REPO_ROOT/Research_Code/Optimization/Data"

if [ ! -x "$JULIA_EXE" ]; then
    echo "Missing repo-local Julia runtime: $JULIA_EXE"
    echo "Run: bash Julia/setup_hpc_julia.sh"
    exit 1
fi

JULIA_USER_DEPOT="$JULIA_ROOT/depot"
export JULIA_DEPOT_PATH="$JULIA_USER_DEPOT:"
export JULIA_PKG_DEVDIR="$JULIA_USER_DEPOT/dev"
export JULIA_PKG_USE_CLI_GIT=true
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export JULIA_BLAS_THREADS="${JULIA_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$JULIA_BLAS_THREADS"
export MKL_NUM_THREADS="$JULIA_BLAS_THREADS"

mkdir -p "$JULIA_USER_DEPOT" "$JULIA_PKG_DEVDIR"

if [ "${INSTANTIATE:-true}" = "true" ]; then
    echo "JULIA_EXE=$JULIA_EXE"
    echo "HPC_JULIA_PROJECT=$HPC_JULIA_PROJECT"
    echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
    "$JULIA_EXE" -e 'println("DEPOT_PATH=", DEPOT_PATH); println("LOAD_PATH=", LOAD_PATH)'
    "$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$HPC_DIR/setup_hpc_project.jl"
fi

### ADJUSTED: Forward only explicitly supplied FOM settings and keep defaults in Julia.
ARGS=()
if [ -n "${N:-}" ]; then ARGS+=(--N "$N"); fi
if [ -n "${L:-}" ]; then ARGS+=(--L "$L"); fi
if [ -n "${EPS2:-}" ]; then ARGS+=(--eps2 "$EPS2"); fi
if [ -n "${K:-}" ]; then ARGS+=(--k "$K"); fi
if [ -n "${TFINAL:-}" ]; then ARGS+=(--tfinal "$TFINAL"); fi
if [ -n "${N_OBS:-}" ]; then ARGS+=(--N-obs "$N_OBS"); fi
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

echo "[$(date)] Starting FOM training on $HOSTNAME"
"$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$HPC_DIR/run_fom_hpc.jl" "${ARGS[@]}"
echo "[$(date)] Finished FOM training"
