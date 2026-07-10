#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
### ADJUSTED: Resolve paths from the moved Research_Code/src/HPC/Tools/Sweeps directory.
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../../../../" && pwd)}"
HPC_DIR="$REPO_ROOT/Research_Code/src/HPC/Tools"
SWEEP_DIR="$HPC_DIR/Sweeps"
### ADJUSTED: Keep virtual-queue logs in the moved sweep tooling directory.
LOG_DIR="$SWEEP_DIR"
### ADJUSTED: Keep virtual-queue runtime state beside the queue files in the moved sweep tooling directory.
STATE_DIR="$SWEEP_DIR/.virtual_queue_state"

PYTHON="${PYTHON:-python3}"
QUEUE_LOG="$LOG_DIR/virtual_sweep_queue_queue.tsv"
ACTIVITY_LOG="$LOG_DIR/virtual_sweep_queue.log"
PID_FILE="$STATE_DIR/daemon.pid"
LOCK_DIR="$STATE_DIR/lock"
STOP_FILE="$STATE_DIR/stop"

VQ_JOB_NAME="${VQ_JOB_NAME:-ac_vqueue}"
VQ_MAX_CONCURRENT="${VQ_MAX_CONCURRENT:-8}"
VQ_CLUSTER_MAX_SUBMITTED="${VQ_CLUSTER_MAX_SUBMITTED:-20}"
VQ_SLEEP_SECONDS="${VQ_SLEEP_SECONDS:-60}"
VQ_IDLE_EXIT_CHECKS="${VQ_IDLE_EXIT_CHECKS:-60}"

mkdir -p "$LOG_DIR" "$STATE_DIR"
touch "$QUEUE_LOG" "$ACTIVITY_LOG"

usage() {
    cat <<EOF
Usage:
  bash Research_Code/src/HPC/Tools/Sweeps/virtual_sweep_queue.sh <sweep_file.txt> [more_sweeps...]
  bash Research_Code/src/HPC/Tools/Sweeps/virtual_sweep_queue.sh status
  bash Research_Code/src/HPC/Tools/Sweeps/virtual_sweep_queue.sh stop

Environment:
  VQ_MAX_CONCURRENT=8          Maximum active virtual-queue Slurm jobs.
  VQ_CLUSTER_MAX_SUBMITTED=20  User-wide Slurm queue limit to stay under.
  VQ_SLEEP_SECONDS=60          Seconds between queue checks.
  VQ_JOB_NAME=ac_vqueue        Slurm job name used for queued jobs.

Logs:
  $ACTIVITY_LOG
  $QUEUE_LOG
EOF
}

timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

log_activity() {
    echo "[$(timestamp)] $*" >> "$ACTIVITY_LOG"
}

acquire_lock() {
    local waited=0
    while ! mkdir "$LOCK_DIR" 2>/dev/null; do
        sleep 1
        waited=$((waited + 1))
        if [ "$waited" -ge 300 ]; then
            echo "Timed out waiting for virtual queue lock: $LOCK_DIR" >&2
            exit 1
        fi
    done
}

release_lock() {
    rmdir "$LOCK_DIR" 2>/dev/null || true
}

with_lock_cleanup() {
    release_lock
}

pending_count_unlocked() {
    awk 'NF > 0 { count += 1 } END { print count + 0 }' "$QUEUE_LOG"
}

active_virtual_jobs() {
    squeue -r -h -u "$USER" -n "$VQ_JOB_NAME" 2>/dev/null | wc -l | tr -d '[:space:]'
}

active_user_jobs() {
    squeue -r -h -u "$USER" 2>/dev/null | wc -l | tr -d '[:space:]'
}

make_absolute_sweep_path() {
    local sweep_file="$1"
    if [[ "$sweep_file" == /* ]]; then
        printf '%s\n' "$sweep_file"
    else
        printf '%s\n' "$REPO_ROOT/$sweep_file"
    fi
}

enqueue_sweep() {
    local sweep_file
    sweep_file="$(make_absolute_sweep_path "$1")"

    if [ ! -f "$sweep_file" ]; then
        echo "Sweep file not found: $sweep_file" >&2
        exit 1
    fi

    local count
    count="$("$PYTHON" "$SWEEP_DIR/sweep_params.py" count "$sweep_file")"
    if [ "$count" -lt 1 ]; then
        echo "Sweep file produced no jobs: $sweep_file" >&2
        exit 1
    fi

    mapfile -t combo_lines < <("$PYTHON" "$SWEEP_DIR/sweep_params.py" list "$sweep_file")

    acquire_lock
    trap with_lock_cleanup EXIT
    local now summary line_index
    now="$(timestamp)"
    for ((combo_index=0; combo_index<count; combo_index+=1)); do
        line_index=$((combo_index + 2))
        summary="${combo_lines[$line_index]#*: }"
        printf '%s\t%s\t%s\t%s\t%s\n' "$now" "$sweep_file" "$combo_index" "$count" "$summary" >> "$QUEUE_LOG"
    done
    log_activity "enqueued $count jobs from $sweep_file"
    trap - EXIT
    release_lock

    echo "Enqueued $count jobs from $sweep_file"
    echo "Queue log: $QUEUE_LOG"
    echo "Activity log: $ACTIVITY_LOG"
}

daemon_running() {
    if [ ! -s "$PID_FILE" ]; then
        return 1
    fi
    local pid
    pid="$(cat "$PID_FILE")"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

start_daemon_if_needed() {
    if daemon_running; then
        echo "Virtual sweep queue daemon already running with PID $(cat "$PID_FILE")"
        log_activity "daemon already running with PID $(cat "$PID_FILE")"
        return
    fi

    if [ -f "$STOP_FILE" ]; then
        mv "$STOP_FILE" "$STOP_FILE.stale" 2>/dev/null || true
    fi
    nohup bash "$0" --daemon >> "$ACTIVITY_LOG" 2>&1 &
    local pid="$!"
    echo "$pid" > "$PID_FILE"
    log_activity "started daemon PID $pid"
    echo "Started virtual sweep queue daemon with PID $pid"
}

pop_next_job() {
    local first_line tmp_file
    first_line=""
    acquire_lock
    trap with_lock_cleanup EXIT
    if [ -s "$QUEUE_LOG" ]; then
        IFS= read -r first_line < "$QUEUE_LOG" || true
        tmp_file="$STATE_DIR/queue.tmp"
        sed -n '2,$p' "$QUEUE_LOG" > "$tmp_file"
        mv "$tmp_file" "$QUEUE_LOG"
    fi
    trap - EXIT
    release_lock
    printf '%s\n' "$first_line"
}

requeue_front() {
    local line="$1"
    local tmp_file
    acquire_lock
    trap with_lock_cleanup EXIT
    tmp_file="$STATE_DIR/queue.tmp"
    {
        printf '%s\n' "$line"
        cat "$QUEUE_LOG"
    } > "$tmp_file"
    mv "$tmp_file" "$QUEUE_LOG"
    trap - EXIT
    release_lock
}

submit_queue_line() {
    local line="$1"
    local enqueued_at sweep_file combo_index combo_count summary
    IFS=$'\t' read -r enqueued_at sweep_file combo_index combo_count summary <<< "$line"

    if [ -z "${sweep_file:-}" ] || [ -z "${combo_index:-}" ] || [ -z "${combo_count:-}" ]; then
        log_activity "discarded malformed queue line: $line"
        return 0
    fi

    if [ "${DRY_RUN:-false}" = "true" ]; then
        log_activity "DRY_RUN would submit sweep=$sweep_file combo=$combo_index/$combo_count summary=[$summary]"
        return 0
    fi

    ### ADJUSTED: Create the Slurm output directory before sbatch opens ac_vqueue out/err files.
    mkdir -p "$REPO_ROOT/Research_Code/Optimization/Data/_Logs"

    local sbatch_output
    if sbatch_output="$(
        SWEEP_FILE="$sweep_file" \
        SWEEP_TOTAL_COUNT="$combo_count" \
        SWEEP_WORKER_COUNT="$combo_count" \
        sbatch --parsable --job-name "$VQ_JOB_NAME" --array="$combo_index-$combo_index" "$SWEEP_DIR/hpc1_run_sweep.slurm"
    )"; then
        log_activity "submitted job_id=$sbatch_output sweep=$sweep_file combo=$combo_index/$combo_count summary=[$summary]"
        return 0
    fi

    log_activity "submit failed; requeueing sweep=$sweep_file combo=$combo_index/$combo_count summary=[$summary]"
    requeue_front "$line"
    return 1
}

daemon_loop() {
    if ! command -v squeue >/dev/null 2>&1 || ! command -v sbatch >/dev/null 2>&1; then
        log_activity "daemon cannot run because squeue/sbatch is not available"
        echo "virtual_sweep_queue.sh must run on a Slurm login node with squeue and sbatch." >&2
        exit 1
    fi

    log_activity "daemon loop started: job_name=$VQ_JOB_NAME max_concurrent=$VQ_MAX_CONCURRENT cluster_max_submitted=$VQ_CLUSTER_MAX_SUBMITTED sleep_seconds=$VQ_SLEEP_SECONDS"
    local idle_checks=0

    while true; do
        if [ -f "$STOP_FILE" ]; then
            log_activity "stop requested; daemon exiting"
            break
        fi

        local pending total_jobs virtual_jobs
        pending="$(pending_count_unlocked)"
        total_jobs="$(active_user_jobs)"
        virtual_jobs="$(active_virtual_jobs)"

        log_activity "check pending=$pending virtual_active=$virtual_jobs user_active=$total_jobs"

        if [ "$pending" -lt 1 ]; then
            idle_checks=$((idle_checks + 1))
            if [ "$VQ_IDLE_EXIT_CHECKS" -gt 0 ] && [ "$idle_checks" -ge "$VQ_IDLE_EXIT_CHECKS" ]; then
                log_activity "queue empty for $idle_checks checks; daemon exiting"
                break
            fi
            sleep "$VQ_SLEEP_SECONDS"
            continue
        fi
        idle_checks=0

        if [ "$total_jobs" -ge "$VQ_CLUSTER_MAX_SUBMITTED" ]; then
            log_activity "waiting: user_active=$total_jobs is at or above cluster limit $VQ_CLUSTER_MAX_SUBMITTED"
            sleep "$VQ_SLEEP_SECONDS"
            continue
        fi

        if [ "$virtual_jobs" -ge "$VQ_MAX_CONCURRENT" ]; then
            log_activity "waiting: virtual_active=$virtual_jobs is at or above virtual limit $VQ_MAX_CONCURRENT"
            sleep "$VQ_SLEEP_SECONDS"
            continue
        fi

        local line
        line="$(pop_next_job)"
        if [ -z "$line" ]; then
            sleep "$VQ_SLEEP_SECONDS"
            continue
        fi

        submit_queue_line "$line" || true
        sleep "$VQ_SLEEP_SECONDS"
    done

    log_activity "daemon loop stopped"
}

show_status() {
    local pending
    pending="$(pending_count_unlocked)"
    echo "Queue log: $QUEUE_LOG"
    echo "Activity log: $ACTIVITY_LOG"
    echo "Pending jobs: $pending"
    if daemon_running; then
        echo "Daemon: running with PID $(cat "$PID_FILE")"
    else
        echo "Daemon: not running"
    fi
    if command -v squeue >/dev/null 2>&1; then
        echo "Virtual active jobs: $(active_virtual_jobs)"
        echo "User active jobs: $(active_user_jobs)"
    fi
}

stop_daemon() {
    touch "$STOP_FILE"
    log_activity "stop requested by user"
    echo "Stop requested. The daemon will exit after its current check/sleep."
}

main() {
    if [ "$#" -lt 1 ]; then
        usage
        exit 1
    fi

    case "$1" in
        --daemon)
            daemon_loop
            ;;
        status)
            show_status
            ;;
        stop)
            stop_daemon
            ;;
        -h|--help)
            usage
            ;;
        *)
            local sweep_file
            for sweep_file in "$@"; do
                enqueue_sweep "$sweep_file"
            done
            start_daemon_if_needed
            ;;
    esac
}

main "$@"
