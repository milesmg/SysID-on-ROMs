#!/bin/bash
#SBATCH -J brookhaven
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 35:59:00
#SBATCH -o Data/_Logs/%x-%j.out
#SBATCH -e Data/_Logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
JULIA_ROOT="$REPO_ROOT/Julia"
JULIA_EXE="$JULIA_ROOT/julia-1.12.6/bin/julia"
MODE="${MODE:?MODE must be fom or rom}"

case "$MODE" in fom|rom) ;; *) echo "MODE must be fom or rom"; exit 1 ;; esac
cd "$REPO_ROOT"
mkdir -p Data/_Logs "$JULIA_ROOT/depot/dev"

export JULIA_DEPOT_PATH="$JULIA_ROOT/depot:"
export JULIA_PKG_DEVDIR="$JULIA_ROOT/depot/dev"
export JULIA_PKG_USE_CLI_GIT=true
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export JULIA_BLAS_THREADS="${JULIA_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$JULIA_BLAS_THREADS"
export MKL_NUM_THREADS="$JULIA_BLAS_THREADS"

if [ "${INSTANTIATE:-true}" = "true" ]; then
    "$JULIA_EXE" --project="$JULIA_ROOT" "$REPO_ROOT/src/Tools/Julia/setup_project.jl"
fi

### ADJUSTED: Forward the selected equation to the generic Julia entrypoint.
ARGS=(--mode "$MODE")
for pair in \
    EQUATION:equation N:N L:L EPS2:eps2 K:k TFINAL:tfinal N_OBS:N-obs R:r M:m H:h SEED:seed \
    REFERENCE_DT_FACTOR:reference-dt-factor DIMENSION:dimension BOUNDARY_CONDITION:boundary-condition \
    INITIAL_CONDITION:initial-condition ETAS:etas ITERS:iters BETA:beta SAVE_FREQUENCY:save-frequency \
    PRINT_FREQUENCY:print-frequency WARMUP:warmup WINDOW_T:window-T WINDOW_N_OBS:window-N-obs \
    WINDOW_START_POLICY:window-start-policy LOSS_NORMALIZATION:loss-normalization WINDOW_SEED:window-seed \
    LEARNER:learner POLYNOMIAL_DEGREE:polynomial-degree SIGMA:sigma MEAN_C:mean-c D1:d1 D2:d2 RUN_NAME:run-name
do
    environment="${pair%%:*}"
    option="${pair#*:}"
    value="${!environment:-}"
    [ -n "$value" ] && ARGS+=("--$option" "$value")
done

"$JULIA_EXE" --project="$JULIA_ROOT" "$REPO_ROOT/src/run.jl" "${ARGS[@]}"
