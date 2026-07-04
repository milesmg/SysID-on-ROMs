#!/bin/bash
#SBATCH -J ac_backprop
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 35:59:00
#SBATCH -o Research_Code/HPC_compatibility/logs/%x-%j.out
#SBATCH -e Research_Code/HPC_compatibility/logs/%x-%j.err

set -euo pipefail

### ADJUSTED: Mirror the variable-window wrapper with only benchmark-specific arguments added.
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
HPC_DIR="$REPO_ROOT/Research_Code/HPC_compatibility"
BENCHMARK_DIR="$HPC_DIR/accelerated_backprop"
JULIA_ROOT="$REPO_ROOT/Julia"
HPC_JULIA_PROJECT="$JULIA_ROOT/HPC_compatibility"
JULIA_EXE="$JULIA_ROOT/julia-1.12.6/bin/julia"

cd "$REPO_ROOT"
mkdir -p "$HPC_DIR/logs" "$REPO_ROOT/Research_Code/Optimization/Data/BackpropBenchmarks"

export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-8}"
export JULIA_BLAS_THREADS="${JULIA_BLAS_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$JULIA_BLAS_THREADS"
export MKL_NUM_THREADS="$JULIA_BLAS_THREADS"

N="${N:-256}"
L="${L:-1.0}"
EPS2="${EPS2:-1e-2}"
K="${K:-1.0}"
TFINAL="${TFINAL:-2.0}"
N_OBS="${N_OBS:-100}"
H="${H:-8}"
SEED="${SEED:-1}"
REFERENCE_DT_FACTOR="${REFERENCE_DT_FACTOR:-0.5}"
NETWORKS="${NETWORKS:-lux,simplechains}"
### ADJUSTED: Include runtime Enzyme and the Reactant compatibility probe in the default matrix.
VJPS="${VJPS:-auto,finite_diff,forward_diff,reverse_diff,reverse_diff_compiled,zygote,tracker,enzyme,enzyme_runtime,mooncake,reactant}"
WINDOW_T="${WINDOW_T:-0.1,2.0}"
WINDOW_N_OBS="${WINDOW_N_OBS:-10,50}"
REPEATS="${REPEATS:-3}"
DIRECTIONAL_STEP="${DIRECTIONAL_STEP:-1e-4}"
### ADJUSTED: Select the original or finite-difference TRBDF2 state-Jacobian mode.
SOLVER_AUTODIFF="${SOLVER_AUTODIFF:-finite_diff}"

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
    --networks "$NETWORKS"
    --vjps "$VJPS"
    --window-T "$WINDOW_T"
    --window-N-obs "$WINDOW_N_OBS"
    --repeats "$REPEATS"
    --directional-step "$DIRECTIONAL_STEP"
    --solver-autodiff "$SOLVER_AUTODIFF"
)

if [ -n "${RUN_NAME:-}" ]; then
    ARGS+=(--run-name "$RUN_NAME")
fi

echo "[$(date)] Starting accelerated-backprop benchmark on ${HOSTNAME:-unknown}"
echo "Benchmark args: ${ARGS[*]}"

if [ "${DRY_RUN:-false}" = "true" ]; then
    echo "DRY_RUN=true, not running Julia"
    exit 0
fi

JULIA_USER_DEPOT="$JULIA_ROOT/depot"
export JULIA_DEPOT_PATH="$JULIA_USER_DEPOT:"
export JULIA_PKG_DEVDIR="$JULIA_USER_DEPOT/dev"
export JULIA_PKG_USE_CLI_GIT=true

if [ "${INSTANTIATE:-true}" = "true" ]; then
    "$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$HPC_DIR/setup_hpc_project.jl"
fi

"$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$BENCHMARK_DIR/run_accelerated_backprop_hpc.jl" "${ARGS[@]}"
echo "[$(date)] Finished accelerated-backprop benchmark"
