#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

SESSION_NAME="stwm_queue_$(slug_now)"
WORKDIR="$STWM_ROOT"
QUEUE_DIR="${STWM_GPU_QUEUE_DIR:-$STWM_ROOT/outputs/queue/stwm_gpu}"
LOG_FILE=""
ATTACH_AFTER_START=0

IDLE_SLEEP=20
STOP_WHEN_EMPTY=0

PREFER_GPUS=8
MIN_GPUS=1
POLL_SECONDS=30
MAX_MEM_USED_MIB=2000
MAX_UTIL_PERCENT=20
CANDIDATE_GPUS=""
TIMEOUT_SECONDS=0

usage() {
  cat <<'USAGE'
Usage:
  start_gpu_queue_tmux.sh [options]

Description:
  Start FIFO GPU queue worker in detached tmux. Worker survives disconnect.

Options:
  --session NAME            tmux session name (default: stwm_queue_<timestamp>)
  --workdir PATH            Working directory in tmux (default: STWM root)
  --queue-dir PATH          Queue root directory
  --log-file PATH           Worker log file (default: logs/<session>.log)
  --attach                  Attach immediately after start

  --idle-sleep N            Empty-queue sleep seconds (default: 20)
  --stop-when-empty         Exit worker once queue is empty

  --prefer-gpus N           Default prefer-gpus for queued jobs (default: 8)
  --min-gpus N              Default min-gpus for queued jobs (default: 1)
  --poll-seconds N          Default poll interval (default: 30)
  --max-mem-used-mib N      Default idle memory threshold (default: 2000)
  --max-utilization N       Default idle utilization threshold (default: 20)
  --candidate-gpus CSV      Default candidate GPUs
  --timeout-seconds N       Default timeout for GPU claim (default: 0)
  -h, --help                Show this help message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      [[ $# -ge 2 ]] || { echo "Missing value for --session" >&2; exit 1; }
      SESSION_NAME="$2"
      shift 2
      ;;
    --workdir)
      [[ $# -ge 2 ]] || { echo "Missing value for --workdir" >&2; exit 1; }
      WORKDIR="$2"
      shift 2
      ;;
    --queue-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --queue-dir" >&2; exit 1; }
      QUEUE_DIR="$2"
      shift 2
      ;;
    --log-file)
      [[ $# -ge 2 ]] || { echo "Missing value for --log-file" >&2; exit 1; }
      LOG_FILE="$2"
      shift 2
      ;;
    --attach)
      ATTACH_AFTER_START=1
      shift
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
      PREFER_GPUS="$2"
      shift 2
      ;;
    --min-gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for --min-gpus" >&2; exit 1; }
      MIN_GPUS="$2"
      shift 2
      ;;
    --poll-seconds)
      [[ $# -ge 2 ]] || { echo "Missing value for --poll-seconds" >&2; exit 1; }
      POLL_SECONDS="$2"
      shift 2
      ;;
    --max-mem-used-mib)
      [[ $# -ge 2 ]] || { echo "Missing value for --max-mem-used-mib" >&2; exit 1; }
      MAX_MEM_USED_MIB="$2"
      shift 2
      ;;
    --max-utilization)
      [[ $# -ge 2 ]] || { echo "Missing value for --max-utilization" >&2; exit 1; }
      MAX_UTIL_PERCENT="$2"
      shift 2
      ;;
    --candidate-gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for --candidate-gpus" >&2; exit 1; }
      CANDIDATE_GPUS="$2"
      shift 2
      ;;
    --timeout-seconds)
      [[ $# -ge 2 ]] || { echo "Missing value for --timeout-seconds" >&2; exit 1; }
      TIMEOUT_SECONDS="$2"
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

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed" >&2
  exit 2
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$STWM_ROOT/logs/${SESSION_NAME}.log"
fi

ensure_dir "$(dirname "$LOG_FILE")" "$QUEUE_DIR"

WORKER_CMD=(
  "$SCRIPT_DIR/gpu_queue_worker.sh"
  --queue-dir "$QUEUE_DIR"
  --idle-sleep "$IDLE_SLEEP"
  --prefer-gpus "$PREFER_GPUS"
  --min-gpus "$MIN_GPUS"
  --poll-seconds "$POLL_SECONDS"
  --max-mem-used-mib "$MAX_MEM_USED_MIB"
  --max-utilization "$MAX_UTIL_PERCENT"
  --timeout-seconds "$TIMEOUT_SECONDS"
)
if [[ -n "$CANDIDATE_GPUS" ]]; then
  WORKER_CMD+=(--candidate-gpus "$CANDIDATE_GPUS")
fi
if (( STOP_WHEN_EMPTY == 1 )); then
  WORKER_CMD+=(--stop-when-empty)
fi

printf -v WORKER_CMD_Q '%q ' "${WORKER_CMD[@]}"
printf -v WORKDIR_Q '%q' "$WORKDIR"
printf -v LOG_Q '%q' "$LOG_FILE"

TMUX_SHELL_CMD="cd ${WORKDIR_Q} && ${WORKER_CMD_Q} 2>&1 | tee -a ${LOG_Q}"
tmux new-session -d -s "$SESSION_NAME" "$TMUX_SHELL_CMD"

echo "started tmux session: $SESSION_NAME"
echo "queue dir: $QUEUE_DIR"
echo "log file: $LOG_FILE"
echo "attach: tmux attach -t $SESSION_NAME"
echo "watch log: tail -f $LOG_FILE"

if (( ATTACH_AFTER_START == 1 )); then
  exec tmux attach -t "$SESSION_NAME"
fi
