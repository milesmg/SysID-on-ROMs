#!/bin/bash
#SBATCH -J sweep
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 35:59:00
#SBATCH -o Data/_Logs/%x-%A_%a.out
#SBATCH -e Data/_Logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
SLURM_DIR="$REPO_ROOT/src/Tools/Slurm"
SWEEP_DIR="$SLURM_DIR/Sweeps"
LOG_DIR="$REPO_ROOT/Data/_Logs"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

if [ -z "${SWEEP_FILE:-}" ]; then
    echo "Missing SWEEP_FILE. Submit with: SWEEP_FILE=Data/Sweeps/<file>.txt sbatch --array=0-N src/Tools/Slurm/Sweeps/run_sweep.slurm"
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
    fom|rom) TARGET_SCRIPT="$SLURM_DIR/run.slurm" ;;
    *) echo "TARGET must be fom or rom in $SWEEP_FILE."; exit 1 ;;
esac
TARGET_LABEL="${SWEEP_TARGET^^}"

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
    RUN_LOG_OUT="$LOG_DIR/${RUN_NAME}.out"
    RUN_LOG_ERR="$LOG_DIR/${RUN_NAME}.err"
    RUN_STATUS=0
    set +e
    {
        echo "[$(date)] Worker $SWEEP_WORKER_INDEX starting $TARGET_LABEL combination $COMBO_INDEX / $SWEEP_TOTAL_COUNT"
        echo "Sweep file: $SWEEP_FILE"
        echo "Run name: $RUN_NAME"
        echo "Combination label: $SWEEP_COMBO_LABEL"

        env | sort | grep -E '^(EQUATION|D1|D2|SIGMA|MEAN_C|N|L|EPS2|K|TFINAL|N_OBS|R|M|H|SEED|REFERENCE_DT_FACTOR|DIMENSION|BOUNDARY_CONDITION|INITIAL_CONDITION|ETAS|ITERS|BETA|SAVE_FREQUENCY|PRINT_FREQUENCY|WARMUP|WINDOW_T|WINDOW_N_OBS|WINDOW_START_POLICY|LOSS_NORMALIZATION|LOSS_SPACE|WINDOW_SEED|FORCED_DEIM_SPLIT|LEARNED_FUNCTION_ERROR|LEARNED_FUNCTION_ERROR_BOUNDS|LEARNER|POLYNOMIAL_DEGREE|JULIA_NUM_THREADS|JULIA_BLAS_THREADS|RUN_NAME|SWEEP_.*)='

        if [ "${DRY_RUN:-false}" = "true" ]; then
            echo "DRY_RUN=true, not running $TARGET_SCRIPT"
        else
            MODE="$SWEEP_TARGET" bash "$TARGET_SCRIPT"
            RUN_STATUS=$?
        fi

        if [ "$RUN_STATUS" -eq 0 ]; then
            echo "[$(date)] Worker $SWEEP_WORKER_INDEX finished $TARGET_LABEL combination $COMBO_INDEX"
        else
            echo "[$(date)] Worker $SWEEP_WORKER_INDEX failed $TARGET_LABEL combination $COMBO_INDEX with status $RUN_STATUS"
        fi
    } > "$RUN_LOG_OUT" 2> "$RUN_LOG_ERR"
    set -e

    if [ "$RUN_STATUS" -eq 0 ] && [ "${DRY_RUN:-false}" != "true" ]; then
        RUN_DIR="$REPO_ROOT/Data/$RUN_NAME"
        if [ -d "$RUN_DIR" ]; then
            mv "$RUN_LOG_OUT" "$RUN_DIR/log.out"
            mv "$RUN_LOG_ERR" "$RUN_DIR/log.err"
        else
            echo "[$(date)] Expected run directory not found, leaving logs in $LOG_DIR: $RUN_DIR"
        fi
    elif [ "$RUN_STATUS" -ne 0 ]; then
        echo "[$(date)] Sweep combination failed; logs left in $LOG_DIR:"
        echo "  $RUN_LOG_OUT"
        echo "  $RUN_LOG_ERR"
        exit "$RUN_STATUS"
    fi
done
