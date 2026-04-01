#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="$STWM_ROOT/outputs/training/stwm_v4_2_1b_real_confirmation"
QUEUE_ROOT="$STWM_ROOT/outputs/queue/stwm_1b_real"

SEEDS_CSV="42,123"
INCLUDE_OPTIONAL_456=0

SESSION_PREFIX="stwm_1b_real"

LANE0_GPUS="0,1"
LANE1_GPUS="3,7"

POLL_SECONDS=15
MAX_MEM_USED_MIB=90000
MAX_UTIL_PERCENT=98

usage() {
  cat <<'USAGE'
Usage:
  submit_stwm_v4_2_real_1b_queue.sh [options]

Description:
  Start two persistent queue workers (single-GPU claim each) and submit real 1B
  confirmation jobs by seed.

Defaults:
  mandatory seeds: 42,123
  optional seed:   456 (submit only when requested)

Options:
  --out-root PATH             Training output root
  --queue-root PATH           Queue root (will use lane0/lane1 subdirs)
  --seeds CSV                 Mandatory seeds (default: 42,123)
  --include-seed-456          Append optional seed 456 after mandatory jobs
  --session-prefix NAME       tmux session prefix (default: stwm_1b_real)
  --lane0-gpus CSV            Candidate GPUs for lane0 worker (default: 0,1)
  --lane1-gpus CSV            Candidate GPUs for lane1 worker (default: 3,7)
  --poll-seconds N            GPU claim poll seconds (default: 15)
  --max-mem-used-mib N        GPU idle memory threshold (default: 90000)
  --max-utilization N         GPU idle utilization threshold (default: 98)
  -h, --help                  Show this message
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --out-root" >&2; exit 1; }
      OUT_ROOT="$2"
      shift 2
      ;;
    --queue-root)
      [[ $# -ge 2 ]] || { echo "Missing value for --queue-root" >&2; exit 1; }
      QUEUE_ROOT="$2"
      shift 2
      ;;
    --seeds)
      [[ $# -ge 2 ]] || { echo "Missing value for --seeds" >&2; exit 1; }
      SEEDS_CSV="$2"
      shift 2
      ;;
    --include-seed-456)
      INCLUDE_OPTIONAL_456=1
      shift
      ;;
    --session-prefix)
      [[ $# -ge 2 ]] || { echo "Missing value for --session-prefix" >&2; exit 1; }
      SESSION_PREFIX="$2"
      shift 2
      ;;
    --lane0-gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for --lane0-gpus" >&2; exit 1; }
      LANE0_GPUS="$2"
      shift 2
      ;;
    --lane1-gpus)
      [[ $# -ge 2 ]] || { echo "Missing value for --lane1-gpus" >&2; exit 1; }
      LANE1_GPUS="$2"
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

lane0_queue="$QUEUE_ROOT/lane0"
lane1_queue="$QUEUE_ROOT/lane1"
lane0_session="${SESSION_PREFIX}_lane0"
lane1_session="${SESSION_PREFIX}_lane1"

ensure_dir "$lane0_queue" "$lane1_queue"

start_lane() {
  local session_name="$1"
  local queue_dir="$2"
  local gpu_csv="$3"
  local log_file="$STWM_ROOT/logs/${session_name}.log"

  if tmux has-session -t "$session_name" 2>/dev/null; then
    echo "[1b-real-queue] tmux session exists, keep running: $session_name"
    return
  fi

  bash "$SCRIPT_DIR/start_gpu_queue_tmux.sh" \
    --session "$session_name" \
    --queue-dir "$queue_dir" \
    --log-file "$log_file" \
    --idle-sleep 20 \
    --prefer-gpus 1 \
    --min-gpus 1 \
    --poll-seconds "$POLL_SECONDS" \
    --max-mem-used-mib "$MAX_MEM_USED_MIB" \
    --max-utilization "$MAX_UTIL_PERCENT" \
    --candidate-gpus "$gpu_csv" \
    --timeout-seconds 0
}

echo "[1b-real-queue] start/reuse lane workers"
start_lane "$lane0_session" "$lane0_queue" "$LANE0_GPUS"
start_lane "$lane1_session" "$lane1_queue" "$LANE1_GPUS"

IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
if (( INCLUDE_OPTIONAL_456 == 1 )); then
  seeds+=(456)
fi

for i in "${!seeds[@]}"; do
  seed="${seeds[$i]}"
  if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "invalid seed in list: $seed" >&2
    exit 1
  fi

  if (( i % 2 == 0 )); then
    queue_dir="$lane0_queue"
    lane_name="lane0"
    lane_gpus="$LANE0_GPUS"
  else
    queue_dir="$lane1_queue"
    lane_name="lane1"
    lane_gpus="$LANE1_GPUS"
  fi

  job_name="stwm_1b_real_seed${seed}"
  echo "[1b-real-queue] submit ${job_name} -> ${lane_name} (candidates=${lane_gpus})"

  bash "$SCRIPT_DIR/gpu_queue_submit.sh" \
    --queue-dir "$queue_dir" \
    --job-name "$job_name" \
    --prefer-gpus 1 \
    --min-gpus 1 \
    --poll-seconds "$POLL_SECONDS" \
    --max-mem-used-mib "$MAX_MEM_USED_MIB" \
    --max-utilization "$MAX_UTIL_PERCENT" \
    --candidate-gpus "$lane_gpus" \
    --timeout-seconds 0 \
    -- bash "$SCRIPT_DIR/run_stwm_v4_2_1b_real_confirmation_seed.sh" "$seed" "$OUT_ROOT"
done

echo "[1b-real-queue] submitted"
echo "  out_root:          $OUT_ROOT"
echo "  lane0_queue:       $lane0_queue"
echo "  lane1_queue:       $lane1_queue"
echo "  lane0_session:     $lane0_session"
echo "  lane1_session:     $lane1_session"
echo "  next_after_done:   bash scripts/finalize_stwm_v4_2_1b_real_confirmation.sh $OUT_ROOT"
