#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

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
  gpu_auto_claim_run.sh [options] -- <command> [args...]

Description:
  Waits for idle GPU(s), claims the largest available set between --prefer-gpus and
  --min-gpus, exports CUDA_VISIBLE_DEVICES, then executes your command.

Options:
  --prefer-gpus N           Prefer up to N GPUs when available (default: 8)
  --min-gpus N              Minimum GPUs required to start (default: 1)
  --poll-seconds N          Poll interval in seconds (default: 30)
  --max-mem-used-mib N      Idle threshold for used memory (default: 2000)
  --max-utilization N       Idle threshold for GPU utilization percent (default: 20)
  --candidate-gpus CSV      Restrict claims to listed IDs (e.g. 0,1,2,3)
  --timeout-seconds N       Stop waiting after N seconds (0 means wait forever)
  -h, --help                Show this help message

Example:
  gpu_auto_claim_run.sh --prefer-gpus 8 --min-gpus 4 -- \
    bash scripts/run_week2_minival_v2_3_multiseed.sh
USAGE
}

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "$s"
}

is_pos_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && (( "$1" > 0 ))
}

count_csv_items() {
  local csv="$1"
  local token
  local n=0
  if [[ -z "$csv" ]]; then
    echo 0
    return
  fi
  IFS=',' read -r -a _arr <<< "$csv"
  for token in "${_arr[@]}"; do
    token="$(trim "$token")"
    [[ -n "$token" ]] && (( n += 1 ))
  done
  echo "$n"
}

candidate_total() {
  if [[ -n "$CANDIDATE_GPUS" ]]; then
    count_csv_items "$CANDIDATE_GPUS"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
  else
    echo 0
  fi
}

pick_best_once() {
  local g ids
  local pick_cmd
  for (( g=PREFER_GPUS; g>=MIN_GPUS; g-- )); do
    pick_cmd=(
      "$SCRIPT_DIR/gpu_pick_idle.sh"
      --num-gpus "$g"
      --poll-seconds "$POLL_SECONDS"
      --max-mem-used-mib "$MAX_MEM_USED_MIB"
      --max-utilization "$MAX_UTIL_PERCENT"
      --timeout-seconds 1
      --once
    )
    if [[ -n "$CANDIDATE_GPUS" ]]; then
      pick_cmd+=(--candidate-gpus "$CANDIDATE_GPUS")
    fi
    if ids="$("${pick_cmd[@]}" 2>/dev/null)"; then
      printf '%s\n' "$ids"
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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

if ! is_pos_int "$PREFER_GPUS"; then
  echo "--prefer-gpus must be a positive integer" >&2
  exit 1
fi

if ! is_pos_int "$MIN_GPUS"; then
  echo "--min-gpus must be a positive integer" >&2
  exit 1
fi

if ! is_pos_int "$POLL_SECONDS"; then
  echo "--poll-seconds must be a positive integer" >&2
  exit 1
fi

if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "--timeout-seconds must be >= 0" >&2
  exit 1
fi

if (( MIN_GPUS > PREFER_GPUS )); then
  echo "--min-gpus cannot be greater than --prefer-gpus" >&2
  exit 1
fi

TOTAL_CANDIDATES="$(candidate_total)"
if ! [[ "$TOTAL_CANDIDATES" =~ ^[0-9]+$ ]] || (( TOTAL_CANDIDATES <= 0 )); then
  echo "[gpu-auto] no visible GPUs found" >&2
  exit 2
fi

if (( PREFER_GPUS > TOTAL_CANDIDATES )); then
  PREFER_GPUS="$TOTAL_CANDIDATES"
fi

if (( MIN_GPUS > TOTAL_CANDIDATES )); then
  echo "[gpu-auto] min-gpus (${MIN_GPUS}) > visible candidates (${TOTAL_CANDIDATES})" >&2
  exit 2
fi

JOB_CMD=("$@")
start_ts="$(date +%s)"

echo "[gpu-auto] waiting for GPUs: prefer=${PREFER_GPUS}, min=${MIN_GPUS}, poll=${POLL_SECONDS}s"
echo "[gpu-auto] thresholds: max_mem_used_mib=${MAX_MEM_USED_MIB}, max_utilization=${MAX_UTIL_PERCENT}%"
[[ -n "$CANDIDATE_GPUS" ]] && echo "[gpu-auto] candidate-gpus=${CANDIDATE_GPUS}"

while true; do
  if ids="$(pick_best_once)"; then
    gpu_count="$(count_csv_items "$ids")"
    export CUDA_VISIBLE_DEVICES="$ids"
    export STWM_ASSIGNED_GPUS="$ids"
    export STWM_ASSIGNED_GPU_COUNT="$gpu_count"

    echo "[gpu-auto] claimed ${gpu_count} GPU(s): ${ids}"
    echo "[gpu-auto] launching command: ${JOB_CMD[*]}"
    exec "${JOB_CMD[@]}"
  fi

  now_ts="$(date +%s)"
  elapsed=$(( now_ts - start_ts ))
  if (( TIMEOUT_SECONDS > 0 && elapsed >= TIMEOUT_SECONDS )); then
    echo "[gpu-auto] timeout after ${elapsed}s" >&2
    exit 3
  fi

  echo "[gpu-auto] no suitable GPU set yet; elapsed=${elapsed}s"
  sleep "$POLL_SECONDS"
done
