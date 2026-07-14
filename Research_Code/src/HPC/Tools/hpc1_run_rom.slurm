#!/bin/bash
#SBATCH -J hpc_rom # slurm job name
#SBATCH -p cpu #where to submit the job 
#SBATCH -N 1 # 1 node
#SBATCH --cpus-per-task=8 # cores per julia process.
#SBATCH --mem=128G #gb of ram on the node. There is a lot more than this available. s
#SBATCH -t 35:59:00 # run for 36 hours max (rules)
#SBATCH -o Research_Code/Optimization/Data/_Logs/%x-%j.out # output
#SBATCH -e Research_Code/Optimization/Data/_Logs/%x-%j.err # err

set -euo pipefail # fail immediately 

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}" 
HPC_DIR="$REPO_ROOT/Research_Code/src/HPC/Tools"
LOG_DIR="$REPO_ROOT/Research_Code/Optimization/Data/_Logs"
JULIA_ROOT="$REPO_ROOT/Julia"
HPC_JULIA_PROJECT="$JULIA_ROOT/HPC_compatibility"
JULIA_EXE="$JULIA_ROOT/julia-1.12.6/bin/julia"

cd "$REPO_ROOT"
mkdir -p "$LOG_DIR" "$REPO_ROOT/Research_Code/Optimization/Data"

EQUATION_NAME="$(printf '%s' "${EQUATION:-ac}" | tr '[:upper:]' '[:lower:]')"
case "$EQUATION_NAME" in
    ac|allen_cahn|allen-cahn)
        RUNNER="run_rom_hpc.jl"
        EQUATION_NAME="ac"
        ;;
    ch|cahn_hilliard|cahn-hilliard)
        RUNNER="run_rom_ch_hpc.jl"
        EQUATION_NAME="ch"
        ;;
    rd|reaction_diffusion|reaction-diffusion)
        ### ADJUSTED: Route reaction-diffusion ROM jobs to the POD-DEIM runner.
        RUNNER="run_rom_rd_hpc.jl"
        EQUATION_NAME="rd"
        ;;
    *)
        echo "EQUATION must be ac, ch, or rd; got $EQUATION_NAME"
        exit 1
        ;;
esac

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

ARGS=()
if [ -n "${N:-}" ]; then ARGS+=(--N "$N"); fi
if [ -n "${L:-}" ]; then ARGS+=(--L "$L"); fi
if [ -n "${EPS2:-}" ]; then ARGS+=(--eps2 "$EPS2"); fi
if [ -n "${K:-}" ]; then ARGS+=(--k "$K"); fi
if [ -n "${TFINAL:-}" ]; then ARGS+=(--tfinal "$TFINAL"); fi
if [ -n "${N_OBS:-}" ]; then ARGS+=(--N-obs "$N_OBS"); fi
if [ -n "${R:-}" ]; then ARGS+=(--r "$R"); fi
if [ -n "${M:-}" ]; then ARGS+=(--m "$M"); fi
if [ -n "${H:-}" ]; then ARGS+=(--h "$H"); fi
if [ -n "${SEED:-}" ]; then ARGS+=(--seed "$SEED"); fi
if [ -n "${REFERENCE_DT_FACTOR:-}" ]; then ARGS+=(--reference-dt-factor "$REFERENCE_DT_FACTOR"); fi
if [ -n "${DIMENSION:-}" ]; then ARGS+=(--dimension "$DIMENSION"); fi
if [ -n "${BOUNDARY_CONDITION:-}" ]; then ARGS+=(--boundary-condition "$BOUNDARY_CONDITION"); fi
if [ -n "${INITIAL_CONDITION:-}" ]; then ARGS+=(--initial-condition "$INITIAL_CONDITION"); fi
if [ -n "${ETAS:-}" ]; then ARGS+=(--etas "$ETAS"); fi
if [ -n "${ITERS:-}" ]; then ARGS+=(--iters "$ITERS"); fi
if [ -n "${BETA:-}" ]; then ARGS+=(--beta "$BETA"); fi
if [ -n "${SAVE_FREQUENCY:-}" ]; then ARGS+=(--save-frequency "$SAVE_FREQUENCY"); fi
if [ -n "${PRINT_FREQUENCY:-}" ]; then ARGS+=(--print-frequency "$PRINT_FREQUENCY"); fi
if [ -n "${WARMUP:-}" ]; then ARGS+=(--warmup "$WARMUP"); fi
if [ -n "${WINDOW_T:-}" ]; then ARGS+=(--window-T "$WINDOW_T"); fi
if [ -n "${WINDOW_N_OBS:-}" ]; then ARGS+=(--window-N-obs "$WINDOW_N_OBS"); fi
if [ -n "${WINDOW_START_POLICY:-}" ]; then ARGS+=(--window-start-policy "$WINDOW_START_POLICY"); fi
if [ -n "${BATCH_SIZE:-}" ]; then ARGS+=(--batch-size "$BATCH_SIZE"); fi
if [ -n "${LOSS_NORMALIZATION:-}" ]; then ARGS+=(--loss-normalization "$LOSS_NORMALIZATION"); fi
if [ -n "${WINDOW_SEED:-}" ]; then ARGS+=(--window-seed "$WINDOW_SEED"); fi
if [ -n "${LEARNER:-}" ]; then ARGS+=(--learner "$LEARNER"); fi
if [ -n "${POLYNOMIAL_DEGREE:-}" ]; then ARGS+=(--polynomial-degree "$POLYNOMIAL_DEGREE"); fi
if [ "$EQUATION_NAME" = "ch" ]; then
    if [ -n "${SIGMA:-}" ]; then ARGS+=(--sigma "$SIGMA"); fi
    if [ -n "${MEAN_C:-}" ]; then ARGS+=(--mean-c "$MEAN_C"); fi
fi
if [ "$EQUATION_NAME" = "rd" ]; then
    ### ADJUSTED: Forward fixed RD diffusion coefficients through the shared ROM wrapper.
    if [ -n "${D1:-}" ]; then ARGS+=(--d1 "$D1"); fi
    if [ -n "${D2:-}" ]; then ARGS+=(--d2 "$D2"); fi
fi
if [ -n "${RUN_NAME:-}" ]; then ARGS+=(--run-name "$RUN_NAME"); fi

echo "[$(date)] Starting $EQUATION_NAME ROM training on $HOSTNAME"
"$JULIA_EXE" --project="$HPC_JULIA_PROJECT" "$HPC_DIR/$RUNNER" "${ARGS[@]}"
echo "[$(date)] Finished ROM training"
