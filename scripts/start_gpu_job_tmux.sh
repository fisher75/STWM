#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

SESSION_NAME="stwm_gpu_$(slug_now)"
WORKDIR="$STWM_ROOT"
LOG_FILE=""
ATTACH_AFTER_START=0

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
  start_gpu_job_tmux.sh [options] -- <command> [args...]

Description:
  Starts a detached tmux session that waits for idle GPU(s), auto-claims the best
  available set, and then runs your command. Session survives terminal disconnect.

Options:
  --session NAME            tmux session name (default: stwm_gpu_<timestamp>)
  --workdir PATH            Working directory inside tmux (default: STWM root)
  --log-file PATH           Log file path (default: logs/<session>.log)
  --attach                  Attach immediately after creating session

  --prefer-gpus N           Prefer up to N GPUs when available (default: 8)
  --min-gpus N              Minimum GPUs required to start (default: 1)
  --poll-seconds N          Poll interval in seconds (default: 30)
  --max-mem-used-mib N      Idle threshold for used memory (default: 2000)
  --max-utilization N       Idle threshold for GPU utilization percent (default: 20)
  --candidate-gpus CSV      Restrict claims to listed IDs (e.g. 0,1,2,3)
  --timeout-seconds N       Stop waiting after N seconds (0 means wait forever)
  -h, --help                Show this help message

Examples:
  start_gpu_job_tmux.sh --session stwm_1b --prefer-gpus 8 --min-gpus 4 -- \
    bash scripts/run_week2_minival_v2_3_multiseed.sh

  start_gpu_job_tmux.sh --session stwm_seed42 --candidate-gpus 4,5,6,7 -- \
    bash scripts/run_stwm_v4_2_minival_seed42.sh
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
    --log-file)
      [[ $# -ge 2 ]] || { echo "Missing value for --log-file" >&2; exit 1; }
      LOG_FILE="$2"
      shift 2
      ;;
    --attach)
      ATTACH_AFTER_START=1
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
    --)
      shift
      break
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "You must provide a command after --" >&2
  usage
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required but not installed" >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required but not found" >&2
  exit 2
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME" >&2
  exit 1
fi

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="$STWM_ROOT/logs/${SESSION_NAME}.log"
fi

ensure_dir "$(dirname "$LOG_FILE")"

CLAIM_CMD=(
  "$SCRIPT_DIR/gpu_auto_claim_run.sh"
  --prefer-gpus "$PREFER_GPUS"
  --min-gpus "$MIN_GPUS"
  --poll-seconds "$POLL_SECONDS"
  --max-mem-used-mib "$MAX_MEM_USED_MIB"
  --max-utilization "$MAX_UTIL_PERCENT"
  --timeout-seconds "$TIMEOUT_SECONDS"
)
if [[ -n "$CANDIDATE_GPUS" ]]; then
  CLAIM_CMD+=(--candidate-gpus "$CANDIDATE_GPUS")
fi
CLAIM_CMD+=(--)
CLAIM_CMD+=("$@")

printf -v CLAIM_CMD_Q '%q ' "${CLAIM_CMD[@]}"
printf -v WORKDIR_Q '%q' "$WORKDIR"
printf -v LOG_Q '%q' "$LOG_FILE"

TMUX_SHELL_CMD="cd ${WORKDIR_Q} && ${CLAIM_CMD_Q} 2>&1 | tee -a ${LOG_Q}"
tmux new-session -d -s "$SESSION_NAME" "$TMUX_SHELL_CMD"

echo "started tmux session: $SESSION_NAME"
echo "log file: $LOG_FILE"
echo "attach: tmux attach -t $SESSION_NAME"
echo "watch log: tail -f $LOG_FILE"

if (( ATTACH_AFTER_START == 1 )); then
  exec tmux attach -t "$SESSION_NAME"
fi
