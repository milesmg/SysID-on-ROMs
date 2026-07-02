#!/bin/bash
#SBATCH -J texas2k_prof
#SBATCH -p long-96core-shared          # or medium-40core-shared
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH -t 47:59:00
#SBATCH -o logs/texas2k-%A_%a.out
#SBATCH -e logs/texas2k-%A_%a.err
#SBATCH --array=0-23                   # 24 tasks: 0..23

# Make sure log directory exists
mkdir -p logs

export PYTHON_JULIAPKG_EXE=/gpfs/software/julia-1.11.5/bin/julia
export PYTHON_JULIAPKG_PROJECT=/gpfs/scratch/$USER/gridfm_julia_env_base
export JULIA_DEPOT_PATH=/gpfs/scratch/$USER/julia_depot
export JULIA_PKG_DEVDIR=/gpfs/scratch/$USER/julia_depot/dev
export JULIA_PKG_USE_CLI_GIT=true

# Prevent juliapkg from modifying the pre-built project (read-only usage)
export PYTHON_JULIAPKG_OFFLINE=yes

mkdir -p "$JULIA_DEPOT_PATH" "$JULIA_PKG_DEVDIR"

# Sanity checks
echo "python=$(which python)"
julia --version
julia --project="$PYTHON_JULIAPKG_PROJECT" -e 'using Pkg; Pkg.instantiate(); using PowerModels, Ipopt, Memento; println("Julia env OK")'

# === Per-task scenario range ===
# 81024 scenarios / 24 jobs = 3376 per job
CHUNK=3376

START_INDEX=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
END_INDEX=$(( START_INDEX + CHUNK ))

# === Launch multiple parallel workers within this job ===
# Each worker gets its own Python+Julia process and a sub-range of scenarios.
# 32 cores / 4 workers = 8 cores per worker (for BLAS/Ipopt threading)
WORKERS=4
CORES_PER_WORKER=$(( SLURM_CPUS_PER_TASK / WORKERS ))
SUB_CHUNK=$(( (CHUNK + WORKERS - 1) / WORKERS ))

echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}: scenarios [${START_INDEX}, ${END_INDEX})"
echo "  Launching ${WORKERS} workers, ${CORES_PER_WORKER} cores each, ~${SUB_CHUNK} scenarios each"

PIDS=()
for w in $(seq 0 $((WORKERS-1))); do
    SUB_START=$(( START_INDEX + w * SUB_CHUNK ))
    SUB_END=$(( SUB_START + SUB_CHUNK ))
    if [ $SUB_END -gt $END_INDEX ]; then SUB_END=$END_INDEX; fi
    if [ $SUB_START -ge $END_INDEX ]; then break; fi

    echo "  Worker ${w}: scenarios [${SUB_START}, ${SUB_END})"

    OPENBLAS_NUM_THREADS=$CORES_PER_WORKER \
    MKL_NUM_THREADS=$CORES_PER_WORKER \
    JULIA_NUM_THREADS=$CORES_PER_WORKER \
    python -u /gpfs/home/srburra/GridSearch/Texas2k-Prof/Powermodels-case_ACTIVSg2000.py \
        --start-index "${SUB_START}" \
        --end-index "${SUB_END}" \
        --save-every 100 \
        > logs/texas2k-${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_w${w}.out 2>&1 &

    PIDS+=($!)
done

echo "  Waiting for ${#PIDS[@]} workers (PIDs: ${PIDS[*]})..."
FAIL=0
for pid in "${PIDS[@]}"; do
    wait $pid || FAIL=$((FAIL+1))
done

if [ $FAIL -gt 0 ]; then
    echo "[$(date)] ${FAIL} worker(s) failed"
    exit 1
else
    echo "[$(date)] All workers completed successfully"
fi
