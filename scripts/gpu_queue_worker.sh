#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

QUEUE_DIR="${STWM_GPU_QUEUE_DIR:-$STWM_ROOT/outputs/queue/stwm_gpu}"
IDLE_SLEEP="${STWM_GPU_QUEUE_IDLE_SLEEP:-20}"
STOP_WHEN_EMPTY=0

DEFAULT_PREFER_GPUS=8
DEFAULT_MIN_GPUS=1
DEFAULT_POLL_SECONDS=30
DEFAULT_MAX_MEM_USED_MIB=2000
DEFAULT_MAX_UTIL_PERCENT=20
DEFAULT_CANDIDATE_GPUS=""
DEFAULT_TIMEOUT_SECONDS=0

usage() {
  cat <<'USAGE'
Usage:
  gpu_queue_worker.sh [options]

Description:
  Consume FIFO queue jobs from pending/, claim GPUs automatically, and run each
  job sequentially. Designed to run inside tmux for persistence.

Options:
  --queue-dir PATH          Queue root directory (default: outputs/queue/stwm_gpu)
  --idle-sleep N            Sleep seconds when queue is empty (default: 20)
  --stop-when-empty         Exit once pending queue is empty

  --prefer-gpus N           Default prefer-gpus when job file omits it (default: 8)
  --min-gpus N              Default min-gpus when job file omits it (default: 1)
  --poll-seconds N          Default poll interval (default: 30)
  --max-mem-used-mib N      Default idle memory threshold (default: 2000)
  --max-utilization N       Default idle utilization threshold (default: 20)
  --candidate-gpus CSV      Default candidate GPU set
  --timeout-seconds N       Default timeout (0 = wait forever)
  -h, --help                Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --queue-dir" >&2; exit 1; }
      QUEUE_DIR="$2"
      shift 2
      ;;
    --idle-sleep)
      [[ $# -ge 2 ]] || { echo "Missing value for --idle-sleep" >&2; exit 1; }
      IDLE_SLEEP="$2"
      shift 2
      ;;
    --stop-when-empty)
      STOP_WHEN_EMPTY=1
      shift
      ;;
    --prefer-gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for --prefer-gpus" >&2; exit 1; }
      DEFAULT_PREFER_GPUS="$2"
      shift 2
      ;;
    --min-gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for --min-gpus" >&2; exit 1; }
      DEFAULT_MIN_GPUS="$2"
      shift 2
      ;;
    --poll-seconds)
      [[ $# -ge 2 ]] || { echo "Missing value for --poll-seconds" >&2; exit 1; }
      DEFAULT_POLL_SECONDS="$2"
      shift 2
      ;;
    --max-mem-used-mib)
      [[ $# -ge 2 ]] || { echo "Missing value for --max-mem-used-mib" >&2; exit 1; }
      DEFAULT_MAX_MEM_USED_MIB="$2"
      shift 2
      ;;
    --max-utilization)
      [[ $# -ge 2 ]] || { echo "Missing value for --max-utilization" >&2; exit 1; }
      DEFAULT_MAX_UTIL_PERCENT="$2"
      shift 2
      ;;
    --candidate-gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for --candidate-gpus" >&2; exit 1; }
      DEFAULT_CANDIDATE_GPUS="$2"
      shift 2
      ;;
    --timeout-seconds)
      [[ $# -ge 2 ]] || { echo "Missing value for --timeout-seconds" >&2; exit 1; }
      DEFAULT_TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

ensure_dir "$QUEUE_DIR/pending" "$QUEUE_DIR/running" "$QUEUE_DIR/done" "$QUEUE_DIR/failed" "$QUEUE_DIR/logs"

LOCK_FILE="$QUEUE_DIR/.worker.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[gpu-queue-worker] another worker already holds lock: $LOCK_FILE" >&2
  exit 3
fi

EVENT_LOG="$QUEUE_DIR/queue_events.log"

event() {
  local level="$1"
  shift
  printf '%s\t%s\t%s\n' "$(timestamp)" "$level" "$*" | tee -a "$EVENT_LOG"
}

pick_next_job() {
  find "$QUEUE_DIR/pending" -maxdepth 1 -type f -name '*.job' | sort | head -n 1
}

run_one_job() {
  local job_path="$1"
  local base
  base="$(basename "$job_path")"
  local running_path="$QUEUE_DIR/running/$base"
  mv "$job_path" "$running_path"

  local job_id job_name submit_ts workdir
  local prefer_gpus min_gpus poll_seconds max_mem_used_mib max_util_percent candidate_gpus timeout_seconds
  local command_escaped command_pretty

  # shellcheck disable=SC1090
  source "$running_path"

  workdir="${workdir:-$STWM_ROOT}"
  prefer_gpus="${prefer_gpus:-$DEFAULT_PREFER_GPUS}"
  min_gpus="${min_gpus:-$DEFAULT_MIN_GPUS}"
  poll_seconds="${poll_seconds:-$DEFAULT_POLL_SECONDS}"
  max_mem_used_mib="${max_mem_used_mib:-$DEFAULT_MAX_MEM_USED_MIB}"
  max_util_percent="${max_util_percent:-$DEFAULT_MAX_UTIL_PERCENT}"
  candidate_gpus="${candidate_gpus:-$DEFAULT_CANDIDATE_GPUS}"
  timeout_seconds="${timeout_seconds:-$DEFAULT_TIMEOUT_SECONDS}"

  local job_log="$QUEUE_DIR/logs/${base%.job}.log"

  event INFO "job_start id=${job_id:-unknown} name=${job_name:-unknown} submit_ts=${submit_ts:-unknown}"
  event INFO "job_cmd id=${job_id:-unknown} cmd=${command_pretty:-unknown}"

  local claim_cmd=(
    "$SCRIPT_DIR/gpu_auto_claim_run.sh"
    --prefer-gpus "$prefer_gpus"
    --min-gpus "$min_gpus"
    --poll-seconds "$poll_seconds"
    --max-mem-used-mib "$max_mem_used_mib"
    --max-utilization "$max_util_percent"
    --timeout-seconds "$timeout_seconds"
  )
  if [[ -n "$candidate_gpus" ]]; then
    claim_cmd+=(--candidate-gpus "$candidate_gpus")
  fi
  claim_cmd+=(-- bash -lc "$command_escaped")

  local rc
  set +e
  (
    cd "$workdir"
    "${claim_cmd[@]}"
  ) 2>&1 | tee -a "$job_log"
  rc=${PIPESTATUS[0]}
  set -e

  {
    printf 'finish_ts=%q\n' "$(timestamp)"
    printf 'exit_code=%q\n' "$rc"
    printf 'worker_log=%q\n' "$job_log"
  } >> "$running_path"

  if (( rc == 0 )); then
    mv "$running_path" "$QUEUE_DIR/done/$base"
    event INFO "job_done id=${job_id:-unknown} exit=${rc} log=$job_log"
  else
    mv "$running_path" "$QUEUE_DIR/failed/$base"
    event ERROR "job_failed id=${job_id:-unknown} exit=${rc} log=$job_log"
  fi
}

event INFO "worker_started queue_dir=$QUEUE_DIR idle_sleep=${IDLE_SLEEP}s stop_when_empty=$STOP_WHEN_EMPTY"

while true; do
  next_job="$(pick_next_job)"
  if [[ -n "$next_job" ]]; then
    run_one_job "$next_job"
    continue
  fi

  if (( STOP_WHEN_EMPTY == 1 )); then
    event INFO "worker_exit queue_empty=1"
    exit 0
  fi

  sleep "$IDLE_SLEEP"
done
