#!/bin/bash
#SBATCH -J ac_fom_var
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 35:59:00
#SBATCH -o Research_Code/HPC_compatibility/logs/%x-%j.out
#SBATCH -e Research_Code/HPC_compatibility/logs/%x-%j.err

set -euo pipefail

### ADJUSTED: Allow manual wrapper runs outside Slurm by falling back to the current directory.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
HPC_DIR="$REPO_ROOT/Research_Code/HPC_compatibility"
VARIABLE_DIR="$HPC_DIR/Variable_trajectory_length"
JULIA_ROOT="$REPO_ROOT/Julia"
HPC_JULIA_PROJECT="$JULIA_ROOT/HPC_compatibility"
JULIA_EXE="$JULIA_ROOT/julia-1.12.6/bin/julia"

cd "$REPO_ROOT"
mkdir -p "$HPC_DIR/logs" "$REPO_ROOT/Research_Code/Optimization/Data"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export JULIA_BLAS_THREADS="${JULIA_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$JULIA_BLAS_THREADS"
export MKL_NUM_THREADS="$JULIA_BLAS_THREADS"

N="${N:-256}"
L="${L:-1.0}"
EPS2="${EPS2:-1e-2}"
K="${K:-1.0}"
TFINAL="${TFINAL:-2.0}"
N_OBS="${N_OBS:-10}"
H="${H:-8}"
SEED="${SEED:-1}"
REFERENCE_DT_FACTOR="${REFERENCE_DT_FACTOR:-0.5}"
ETAS="${ETAS:-5e-3,1e-3,1e-4}"
ITERS="${ITERS:-500,800,1100}"
BETA="${BETA:-0.9,0.99}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-10}"
PRINT_FREQUENCY="${PRINT_FREQUENCY:-10}"
WARMUP="${WARMUP:-true}"
WINDOW_T="${WINDOW_T:-$TFINAL}"
WINDOW_N_OBS="${WINDOW_N_OBS:-$N_OBS}"
WINDOW_START_POLICY="${WINDOW_START_POLICY:-beginning}"
WINDOWS_PER_ITER="${WINDOWS_PER_ITER:-1}"
LOSS_NORMALIZATION="${LOSS_NORMALIZATION:-mean}"
WINDOW_SEED="${WINDOW_SEED:-$SEED}"

ARGS=(
    --N "$N"
    --L "$L"
    --eps2 "$EPS2"
    --k "$K"
    --tfinal "$TFINAL"
    --N-obs "$N_OBS"
    --h "$H"
    --seed "$SEED"
    --reference-dt-factor "$REFERENCE_DT_FACTOR"
    --etas "$ETAS"
    --iters "$ITERS"
    --beta "$BETA"
    --save-frequency "$SAVE_FREQUENCY"
    --print-frequency "$PRINT_FREQUENCY"
    --warmup "$WARMUP"
    --window-T "$WINDOW_T"
    --window-N-obs "$WINDOW_N_OBS"
    --window-start-policy "$WINDOW_START_POLICY"
    --windows-per-iter "$WINDOWS_PER_ITER"
    --loss-normalization "$LOSS_NORMALIZATION"
    --window-seed "$WINDOW_SEED"
)

if [ -n "${RUN_NAME:-}" ]; then
    ARGS+=(--run-name "$RUN_NAME")
fi

echo "[$(date)] Starting variable-window FOM training on ${HOSTNAME:-unknown}"
echo "Variable-window args: ${ARGS[*]}"

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "DRY_RUN=true, not running Julia"
    exit 0
fi

if [ ! -x "$JULIA_EXE" ]; then
    echo "Missing repo-local Julia runtime: $JULIA_EXE"
    echo "Run: bash Julia/setup_hpc_julia.sh"
    exit 1
fi

JULIA_USER_DEPOT="$JULIA_ROOT/depot"
export JULIA_DEPOT_PATH="$JULIA_USER_DEPOT:"
export JULIA_PKG_DEVDIR="$JULIA_USER_DEPOT/dev"
export JULIA_PKG_USE_CLI_GIT=true
mkdir -p "$JULIA_USER_DEPOT" "$JULIA_PKG_DEVDIR"

if [ "${INSTANTIATE:-true}" = "true" ]; then
    echo "JULIA_EXE=$JULIA_EXE"
    echo "HPC_JULIA_PROJECT=$HPC_JULIA_PROJECT"
    echo "JULIA_DEPOT_PATH=$JULIA_DEPOT_PATH"
    "$JULIA_EXE" -e 'println("DEPOT_PATH=", DEPOT_PATH); println("LOAD_PATH=", LOAD_PATH)'
    "$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$HPC_DIR/setup_hpc_project.jl"
fi

"$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$VARIABLE_DIR/run_variable_fom_hpc.jl" "${ARGS[@]}"
echo "[$(date)] Finished variable-window FOM training"
