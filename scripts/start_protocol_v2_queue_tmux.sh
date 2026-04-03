#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

SESSION_NAME=""
WORKDIR="$STWM_ROOT"
QUEUE_DIR="${STWM_PROTOCOL_V2_QUEUE_DIR:-$STWM_ROOT/outputs/queue/stwm_protocol_v2/d0_eval}"
CLASS_TYPE="A"
IDLE_SLEEP="15"
LOG_FILE=""
STOP_WHEN_EMPTY=0
ATTACH_AFTER_START=0

usage() {
  cat <<'USAGE'
Usage:
  start_protocol_v2_queue_tmux.sh [options]

Options:
  --session NAME            tmux session name
  --workdir PATH            Working directory
  --queue-dir PATH          Queue directory
  --class-type {A|B|C}      Worker class type
  --idle-sleep N            Worker idle sleep seconds
  --log-file PATH           Worker log file
  --stop-when-empty         Exit when queue empty
  --attach                  Attach tmux after start
  -h, --help                Show help
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
    --class-type)
      [[ $# -ge 2 ]] || { echo "Missing value for --class-type" >&2; exit 1; }
      CLASS_TYPE="$2"
      shift 2
      ;;
    --idle-sleep)
      [[ $# -ge 2 ]] || { echo "Missing value for --idle-sleep" >&2; exit 1; }
      IDLE_SLEEP="$2"
      shift 2
      ;;
    --log-file)
      [[ $# -ge 2 ]] || { echo "Missing value for --log-file" >&2; exit 1; }
      LOG_FILE="$2"
      shift 2
      ;;
    --stop-when-empty)
      STOP_WHEN_EMPTY=1
      shift
      ;;
    --attach)
      ATTACH_AFTER_START=1
      shift
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
  echo "tmux is required" >&2
  exit 2
fi

CLASS_TYPE="$(echo "$CLASS_TYPE" | tr '[:lower:]' '[:upper:]')"
if [[ "$CLASS_TYPE" != "A" && "$CLASS_TYPE" != "B" && "$CLASS_TYPE" != "C" ]]; then
  echo "class-type must be A/B/C" >&2
  exit 1
fi

if [[ -z "$SESSION_NAME" ]]; then
  queue_name="$(basename "$QUEUE_DIR")"
  SESSION_NAME="stwm_protocol_v2_${queue_name}_worker"
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
  "$SCRIPT_DIR/protocol_v2_queue_worker.sh"
  --queue-dir "$QUEUE_DIR"
  --class-type "$CLASS_TYPE"
  --idle-sleep "$IDLE_SLEEP"
)
if (( STOP_WHEN_EMPTY == 1 )); then
  WORKER_CMD+=(--stop-when-empty)
fi

printf -v WORKER_CMD_Q '%q ' "${WORKER_CMD[@]}"
printf -v WORKDIR_Q '%q' "$WORKDIR"
printf -v LOG_Q '%q' "$LOG_FILE"

TMUX_CMD="cd ${WORKDIR_Q} && ${WORKER_CMD_Q} 2>&1 | tee -a ${LOG_Q}"
tmux new-session -d -s "$SESSION_NAME" "$TMUX_CMD"

pane_pid="$(tmux list-panes -t "$SESSION_NAME" -F '#{pane_pid}' | head -n 1)"

echo "started tmux session: $SESSION_NAME"
echo "pane pid: $pane_pid"
echo "queue dir: $QUEUE_DIR"
echo "class type: $CLASS_TYPE"
echo "log file: $LOG_FILE"
echo "attach: tmux attach -t $SESSION_NAME"

if (( ATTACH_AFTER_START == 1 )); then
  exec tmux attach -t "$SESSION_NAME"
fi
