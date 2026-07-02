#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash Research_Code/HPC_compatibility/Sweeps/submit_sweep.sh <sweep_file.txt>"
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SWEEP_DIR="$REPO_ROOT/Research_Code/HPC_compatibility/Sweeps"
SWEEP_FILE="$1"
PYTHON="${PYTHON:-python3}"

if [[ "$SWEEP_FILE" != /* ]]; then
    SWEEP_FILE="$REPO_ROOT/$SWEEP_FILE"
fi

if [ ! -f "$SWEEP_FILE" ]; then
    echo "Sweep file not found: $SWEEP_FILE"
    exit 1
fi

COUNT="$("$PYTHON" "$SWEEP_DIR/sweep_params.py" count "$SWEEP_FILE")"
if [ "$COUNT" -lt 1 ]; then
    echo "Sweep file produced no jobs: $SWEEP_FILE"
    exit 1
fi

ARRAY_END=$((COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}"

if [ -n "${MAX_CONCURRENT:-}" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

echo "Submitting $COUNT sweep jobs from $SWEEP_FILE"
echo "Array spec: $ARRAY_SPEC"

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "DRY_RUN=true, not submitting."
    echo "SWEEP_FILE=\"$SWEEP_FILE\" sbatch --array=\"$ARRAY_SPEC\" \"$SWEEP_DIR/hpc1_run_sweep.slurm\""
    exit 0
fi

SWEEP_FILE="$SWEEP_FILE" sbatch --array="$ARRAY_SPEC" "$SWEEP_DIR/hpc1_run_sweep.slurm"
