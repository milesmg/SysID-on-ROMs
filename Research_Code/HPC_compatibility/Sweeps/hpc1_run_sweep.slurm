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

eval "$("$PYTHON" "$SWEEP_DIR/sweep_params.py" env "$SWEEP_FILE" "$SLURM_ARRAY_TASK_ID")"

SWEEP_TARGET="${SWEEP_TARGET:-}"
case "$SWEEP_TARGET" in
    fom|FOM)
        TARGET_SCRIPT="$HPC_DIR/hpc1_run_fom.slurm"
        TARGET_LABEL="FOM"
        ;;
    fom_variable|FOM_VARIABLE|variable_fom|VARIABLE_FOM)
        TARGET_SCRIPT="$HPC_DIR/Variable_trajectory_length/hpc1_run_variable_fom.slurm"
        TARGET_LABEL="FOM_VARIABLE"
        ;;
    rom|ROM)
        TARGET_SCRIPT="$HPC_DIR/hpc1_run_rom.slurm"
        TARGET_LABEL="ROM"
        ;;
    *)
        echo "Sweep target must be fom or rom. Set target=rom or target=fom in $SWEEP_FILE."
        exit 1
        ;;
esac

BASE_RUN_NAME="${RUN_NAME:-${SWEEP_NAME:-${TARGET_LABEL}_sweep}}"
RUN_NAME="${BASE_RUN_NAME}_${SLURM_ARRAY_TASK_ID}_${SWEEP_COMBO_LABEL}"
export RUN_NAME

export INSTANTIATE="${INSTANTIATE:-false}"

echo "[$(date)] Starting $TARGET_LABEL sweep task $SLURM_ARRAY_TASK_ID / $SWEEP_COMBO_COUNT"
echo "Sweep file: $SWEEP_FILE"
echo "Run name: $RUN_NAME"
echo "Combination label: $SWEEP_COMBO_LABEL"
env | sort | grep -E '^(N|L|EPS2|K|TFINAL|N_OBS|R|M|H|SEED|REFERENCE_DT_FACTOR|ETAS|ITERS|BETA|SAVE_FREQUENCY|PRINT_FREQUENCY|WARMUP|WINDOW_T|WINDOW_N_OBS|WINDOW_START_POLICY|WINDOWS_PER_ITER|LOSS_NORMALIZATION|WINDOW_SEED|JULIA_NUM_THREADS|JULIA_BLAS_THREADS|RUN_NAME|SWEEP_.*)='

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "DRY_RUN=true, not running $TARGET_SCRIPT"
    exit 0
fi

bash "$TARGET_SCRIPT"

echo "[$(date)] Finished $TARGET_LABEL sweep task $SLURM_ARRAY_TASK_ID"
