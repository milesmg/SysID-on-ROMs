#!/bin/bash
#SBATCH -J ac_sweep
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 35:59:00
#SBATCH -o Research_Code/HPC_compatibility/logs/%x-%A_%a.out
#SBATCH -e Research_Code/HPC_compatibility/logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
SWEEP_DIR="$REPO_ROOT/Research_Code/HPC_compatibility/Sweeps"
HPC_DIR="$REPO_ROOT/Research_Code/HPC_compatibility"

cd "$REPO_ROOT"
mkdir -p "$HPC_DIR/logs"

if [ -z "${SWEEP_FILE:-}" ]; then
    echo "Missing SWEEP_FILE. Submit with: SWEEP_FILE=Research_Code/HPC_compatibility/Sweeps/<file>.txt sbatch --array=0-N Research_Code/HPC_compatibility/Sweeps/hpc1_run_sweep.slurm"
    exit 1
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "This script must be run as a Slurm array job."
    exit 1
fi

if [[ "$SWEEP_FILE" != /* ]]; then
    SWEEP_FILE="$REPO_ROOT/$SWEEP_FILE"
fi

if [ ! -f "$SWEEP_FILE" ]; then
    echo "Sweep file not found: $SWEEP_FILE"
    exit 1
fi

PYTHON="${PYTHON:-python3}"

SWEEP_TARGET="$("$PYTHON" "$SWEEP_DIR/sweep_params.py" target "$SWEEP_FILE")"
case "$SWEEP_TARGET" in
    fom|FOM)
        TARGET_SCRIPT="$HPC_DIR/hpc1_run_fom.slurm"
        TARGET_LABEL="FOM"
        ;;
    fom_variable|FOM_VARIABLE|variable_fom|VARIABLE_FOM)
        ### ADJUSTED: Route legacy variable-FOM sweeps through the standard central FOM wrapper.
        TARGET_SCRIPT="$HPC_DIR/hpc1_run_fom.slurm"
        TARGET_LABEL="FOM_VARIABLE"
        ;;
    rom|ROM)
        TARGET_SCRIPT="$HPC_DIR/hpc1_run_rom.slurm"
        TARGET_LABEL="ROM"
        ;;
    *)
        echo "Sweep target must be fom, fom_variable, or rom in $SWEEP_FILE."
        exit 1
        ;;
esac

SWEEP_TOTAL_COUNT="${SWEEP_TOTAL_COUNT:-$("$PYTHON" "$SWEEP_DIR/sweep_params.py" count "$SWEEP_FILE")}"
SWEEP_WORKER_COUNT="${SWEEP_WORKER_COUNT:-$SWEEP_TOTAL_COUNT}"
SWEEP_WORKER_INDEX="$SLURM_ARRAY_TASK_ID"
INITIAL_RUN_NAME="${RUN_NAME:-}"
INITIAL_SWEEP_NAME="${SWEEP_NAME:-}"
export INSTANTIATE="${INSTANTIATE:-false}"

for ((COMBO_INDEX=SWEEP_WORKER_INDEX; COMBO_INDEX<SWEEP_TOTAL_COUNT; COMBO_INDEX+=SWEEP_WORKER_COUNT)); do
    RUN_NAME="$INITIAL_RUN_NAME"
    SWEEP_NAME="$INITIAL_SWEEP_NAME"
    eval "$("$PYTHON" "$SWEEP_DIR/sweep_params.py" env "$SWEEP_FILE" "$COMBO_INDEX")"

    BASE_RUN_NAME="${RUN_NAME:-${SWEEP_NAME:-${TARGET_LABEL}_sweep}}"
    RUN_NAME="${BASE_RUN_NAME}_${COMBO_INDEX}_${SWEEP_COMBO_LABEL}"
    export RUN_NAME SWEEP_WORKER_COUNT SWEEP_WORKER_INDEX

    echo "[$(date)] Worker $SWEEP_WORKER_INDEX starting $TARGET_LABEL combination $COMBO_INDEX / $SWEEP_TOTAL_COUNT"
    echo "Sweep file: $SWEEP_FILE"
    echo "Run name: $RUN_NAME"
    echo "Combination label: $SWEEP_COMBO_LABEL"
    # ### ADJUSTED: Print BATCH_SIZE with the other forwarded sweep parameters.
    env | sort | grep -E '^(N|L|EPS2|K|TFINAL|N_OBS|R|M|H|SEED|REFERENCE_DT_FACTOR|ETAS|ITERS|BETA|SAVE_FREQUENCY|PRINT_FREQUENCY|WARMUP|WINDOW_T|WINDOW_N_OBS|WINDOW_START_POLICY|BATCH_SIZE|LOSS_NORMALIZATION|WINDOW_SEED|JULIA_NUM_THREADS|JULIA_BLAS_THREADS|RUN_NAME|SWEEP_.*)='

    if [ "${DRY_RUN:-false}" = "true" ]; then
        echo "DRY_RUN=true, not running $TARGET_SCRIPT"
        continue
    fi

    bash "$TARGET_SCRIPT"
    echo "[$(date)] Worker $SWEEP_WORKER_INDEX finished $TARGET_LABEL combination $COMBO_INDEX"
done
