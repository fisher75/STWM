#!/usr/bin/env bash
set -euo pipefail

NUM_GPUS=1
POLL_SECONDS=30
MAX_MEM_USED_MIB=2000
MAX_UTIL_PERCENT=20
CANDIDATE_GPUS=""
TIMEOUT_SECONDS=0
ONCE=0

usage() {
  cat <<'USAGE'
Usage:
  gpu_pick_idle.sh [options]

Options:
  --num-gpus N              Number of GPUs to claim (default: 1)
  --poll-seconds N          Poll interval in seconds (default: 30)
  --max-mem-used-mib N      Max used memory per GPU in MiB (default: 2000)
  --max-utilization N       Max gpu utilization percent (default: 20)
  --candidate-gpus CSV      Candidate GPU indices, e.g. 0,1,2,3
  --timeout-seconds N       Timeout in seconds (0 means wait forever)
  --once                    Do one scan only, return non-zero if unavailable
  -h, --help                Show this help message

Output:
  Prints claimed GPU indices as CSV, e.g. 0,3
USAGE
}

trim() {
  local s="$1"
  s="${s#${s%%[![:space:]]*}}"
  s="${s%${s##*[![:space:]]}}"
  printf '%s' "$s"
}

is_candidate() {
  local idx="$1"
  if [[ -z "$CANDIDATE_GPUS" ]]; then
    return 0
  fi
  local token
  IFS=',' read -r -a _cand <<< "$CANDIDATE_GPUS"
  for token in "${_cand[@]}"; do
    token="$(trim "$token")"
    if [[ "$token" == "$idx" ]]; then
      return 0
    fi
  done
  return 1
}

scan_once() {
  local raw
  if ! raw="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)"; then
    echo "[gpu-pick] nvidia-smi query failed" >&2
    return 3
  fi

  local selected=()
  local line idx mem util
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    IFS=',' read -r idx mem util <<< "$line"
    idx="$(trim "$idx")"
    mem="$(trim "$mem")"
    util="$(trim "$util")"

    is_candidate "$idx" || continue

    if [[ "$mem" =~ ^[0-9]+$ ]] && [[ "$util" =~ ^[0-9]+$ ]]; then
      if (( mem <= MAX_MEM_USED_MIB && util <= MAX_UTIL_PERCENT )); then
        selected+=("$idx")
      fi
    fi
  done <<< "$raw"

  if (( ${#selected[@]} >= NUM_GPUS )); then
    local out=()
    local i
    for (( i=0; i<NUM_GPUS; i++ )); do
      out+=("${selected[$i]}")
    done
    local csv
    csv="$(IFS=','; echo "${out[*]}")"
    printf '%s\n' "$csv"
    return 0
  fi

  return 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --max-mem-used-mib)
      MAX_MEM_USED_MIB="$2"
      shift 2
      ;;
    --max-utilization)
      MAX_UTIL_PERCENT="$2"
      shift 2
      ;;
    --candidate-gpus)
      CANDIDATE_GPUS="$2"
      shift 2
      ;;
    --timeout-seconds)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --once)
      ONCE=1
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

if ! [[ "$NUM_GPUS" =~ ^[0-9]+$ ]] || (( NUM_GPUS <= 0 )); then
  echo "--num-gpus must be a positive integer" >&2
  exit 1
fi

if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || (( POLL_SECONDS <= 0 )); then
  echo "--poll-seconds must be a positive integer" >&2
  exit 1
fi

if ! [[ "$TIMEOUT_SECONDS" =~ ^[0-9]+$ ]] || (( TIMEOUT_SECONDS < 0 )); then
  echo "--timeout-seconds must be >= 0" >&2
  exit 1
fi

start_ts="$(date +%s)"

while true; do
  if ids="$(scan_once)"; then
    printf '%s\n' "$ids"
    exit 0
  fi

  rc="$?"
  if (( rc == 3 )); then
    exit 3
  fi

  if (( ONCE == 1 )); then
    exit 2
  fi

  now_ts="$(date +%s)"
  elapsed=$(( now_ts - start_ts ))
  if (( TIMEOUT_SECONDS > 0 && elapsed >= TIMEOUT_SECONDS )); then
    echo "[gpu-pick] timeout after ${elapsed}s" >&2
    exit 4
  fi

  echo "[gpu-pick] waiting for ${NUM_GPUS} idle GPU(s); elapsed=${elapsed}s" >&2
  sleep "$POLL_SECONDS"
done
