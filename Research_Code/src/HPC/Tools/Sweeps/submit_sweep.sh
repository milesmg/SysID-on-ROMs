#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash Research_Code/src/HPC/Tools/Sweeps/submit_sweep.sh <sweep_file.txt>"
    exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
### ADJUSTED: Use the moved sweep tooling directory.
SWEEP_DIR="$REPO_ROOT/Research_Code/src/HPC/Tools/Sweeps"
### ADJUSTED: Create the Slurm output directory before sbatch opens %x-%A_%a logs.
SLURM_LOG_DIR="$REPO_ROOT/Research_Code/Optimization/Data/_Logs"
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

### ADJUSTED: Fit sweep workers within the hpc1 submit limit and run excess combinations sequentially.
MAX_SUBMITTED="${MAX_SUBMITTED:-20}"
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
CURRENT_SUBMITTED="${CURRENT_SUBMITTED:-$(squeue -r -h -u "$USER" | wc -l | tr -d '[:space:]')}"
AVAILABLE_SUBMISSIONS=$((MAX_SUBMITTED - CURRENT_SUBMITTED))

if [ "$AVAILABLE_SUBMISSIONS" -lt 1 ]; then
    echo "No submission slots available: $CURRENT_SUBMITTED of $MAX_SUBMITTED are already in use."
    exit 1
fi

WORKER_COUNT="$COUNT"
if [ "$WORKER_COUNT" -gt "$AVAILABLE_SUBMISSIONS" ]; then
    WORKER_COUNT="$AVAILABLE_SUBMISSIONS"
fi

CONCURRENT_WORKERS="$MAX_CONCURRENT"
if [ "$CONCURRENT_WORKERS" -gt "$WORKER_COUNT" ]; then
    CONCURRENT_WORKERS="$WORKER_COUNT"
fi

ARRAY_END=$((WORKER_COUNT - 1))
ARRAY_SPEC="0-${ARRAY_END}%${CONCURRENT_WORKERS}"

echo "Submitting $COUNT sweep combinations with $WORKER_COUNT Slurm workers from $SWEEP_FILE"
echo "Submitted jobs already in queue: $CURRENT_SUBMITTED / $MAX_SUBMITTED"
echo "Array spec: $ARRAY_SPEC"

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "DRY_RUN=true, not submitting."
    echo "SWEEP_FILE=\"$SWEEP_FILE\" SWEEP_TOTAL_COUNT=\"$COUNT\" SWEEP_WORKER_COUNT=\"$WORKER_COUNT\" sbatch --array=\"$ARRAY_SPEC\" \"$SWEEP_DIR/hpc1_run_sweep.slurm\""
    exit 0
fi

mkdir -p "$SLURM_LOG_DIR"
SWEEP_FILE="$SWEEP_FILE" SWEEP_TOTAL_COUNT="$COUNT" SWEEP_WORKER_COUNT="$WORKER_COUNT" sbatch --array="$ARRAY_SPEC" "$SWEEP_DIR/hpc1_run_sweep.slurm"
