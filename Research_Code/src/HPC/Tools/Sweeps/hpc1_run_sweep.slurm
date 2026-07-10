#!/bin/bash
#SBATCH -J ac_sweep
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 35:59:00
#SBATCH -o Research_Code/Optimization/Data/_Logs/%x-%A_%a.out
#SBATCH -e Research_Code/Optimization/Data/_Logs/%x-%A_%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
### ADJUSTED: Use the moved HPC sweep/tooling directories.
HPC_DIR="$REPO_ROOT/Research_Code/src/HPC/Tools"
SWEEP_DIR="$HPC_DIR/Sweeps"
LOG_DIR="$REPO_ROOT/Research_Code/Optimization/Data/_Logs"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR"

if [ -z "${SWEEP_FILE:-}" ]; then
    ### ADJUSTED: Point missing-file guidance at the moved Optimization/Sweeps directory.
    echo "Missing SWEEP_FILE. Submit with: SWEEP_FILE=Research_Code/Optimization/Sweeps/<file>.txt sbatch --array=0-N Research_Code/src/HPC/Tools/Sweeps/hpc1_run_sweep.slurm"
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
    ### ADJUSTED: Capture one explicit out/err pair per sweep combination so FOM/ROM logs are not hidden under the array-worker log.
    RUN_LOG_OUT="$LOG_DIR/${RUN_NAME}.out"
    RUN_LOG_ERR="$LOG_DIR/${RUN_NAME}.err"
    RUN_STATUS=0
    set +e
    {
        echo "[$(date)] Worker $SWEEP_WORKER_INDEX starting $TARGET_LABEL combination $COMBO_INDEX / $SWEEP_TOTAL_COUNT"
        echo "Sweep file: $SWEEP_FILE"
        echo "Run name: $RUN_NAME"
        echo "Combination label: $SWEEP_COMBO_LABEL"
        # ### ADJUSTED: Print BATCH_SIZE with the other forwarded sweep parameters.
        ### ADJUSTED: Include named initial conditions in per-combination sweep logs.
        env | sort | grep -E '^(N|L|EPS2|K|TFINAL|N_OBS|R|M|H|SEED|REFERENCE_DT_FACTOR|DIMENSION|BOUNDARY_CONDITION|INITIAL_CONDITION|ETAS|ITERS|BETA|SAVE_FREQUENCY|PRINT_FREQUENCY|WARMUP|WINDOW_T|WINDOW_N_OBS|WINDOW_START_POLICY|BATCH_SIZE|LOSS_NORMALIZATION|WINDOW_SEED|LEARNER|POLYNOMIAL_DEGREE|JULIA_NUM_THREADS|JULIA_BLAS_THREADS|RUN_NAME|SWEEP_.*)='

        if [ "${DRY_RUN:-false}" = "true" ]; then
            echo "DRY_RUN=true, not running $TARGET_SCRIPT"
        else
            bash "$TARGET_SCRIPT"
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
        RUN_DIR="$REPO_ROOT/Research_Code/Optimization/Data/$RUN_NAME"
        if [ -d "$RUN_DIR" ]; then
            ### ADJUSTED: Move successful run-named logs out of _Logs instead of leaving duplicate copies.
            mv "$RUN_LOG_OUT" "$RUN_DIR/${RUN_NAME}.out"
            mv "$RUN_LOG_ERR" "$RUN_DIR/${RUN_NAME}.err"
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
