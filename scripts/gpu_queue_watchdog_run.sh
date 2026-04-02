#!/usr/bin/env bash
set -euo pipefail

RETRY_SLEEP_SECONDS=60
MAX_RETRIES=0
APPEND_RESUME_FLAG=1
OOM_REGEX='CUDA out of memory|cuda out of memory|OutOfMemoryError|out of memory|oom-kill|Killed process|Cannot allocate memory'

usage() {
  cat <<'USAGE'
Usage:
  gpu_queue_watchdog_run.sh [options] -- <command> [args...]

Description:
  Wrap a queue job command with retry-on-failure policy for OOM/SIGKILL cases.
  On retryable failures, waits 60s and retries indefinitely by default.

Options:
  --sleep-seconds N        Sleep before retry (default: 60)
  --max-retries N          Max retry attempts (0 = infinite, default: 0)
  --no-append-resume       Do not append --resume on retry
  --append-resume          Force append --resume on retry (default: on)
  --oom-regex REGEX        Regex used to detect OOM signatures in attempt output
  -h, --help               Show this help message
USAGE
}

is_non_negative_int() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

append_resume_if_needed() {
  local -n _cmd_ref=$1

  local joined
  printf -v joined '%q ' "${_cmd_ref[@]}"
  if [[ "$joined" != *"run_stwm_v4_2_real_train_seed.sh"* ]]; then
    return 0
  fi

  if (( ${#_cmd_ref[@]} >= 3 )) && [[ "${_cmd_ref[0]}" == "bash" ]] && [[ "${_cmd_ref[1]}" == "-lc" ]]; then
    local inner="${_cmd_ref[2]}"
    if [[ "$inner" != *" --resume"* ]] && [[ "$inner" != *"--resume="* ]]; then
      _cmd_ref[2]="${inner} --resume"
    fi
    return 0
  fi

  local arg
  for arg in "${_cmd_ref[@]}"; do
    if [[ "$arg" == "--resume" ]] || [[ "$arg" == --resume=* ]]; then
      return 0
    fi
  done

  _cmd_ref+=(--resume)
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sleep-seconds)
      [[ $# -ge 2 ]] || { echo "Missing value for --sleep-seconds" >&2; exit 1; }
      RETRY_SLEEP_SECONDS="$2"
      shift 2
      ;;
    --max-retries)
      [[ $# -ge 2 ]] || { echo "Missing value for --max-retries" >&2; exit 1; }
      MAX_RETRIES="$2"
      shift 2
      ;;
    --no-append-resume)
      APPEND_RESUME_FLAG=0
      shift
      ;;
    --append-resume)
      APPEND_RESUME_FLAG=1
      shift
      ;;
    --oom-regex)
      [[ $# -ge 2 ]] || { echo "Missing value for --oom-regex" >&2; exit 1; }
      OOM_REGEX="$2"
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

if ! is_non_negative_int "$RETRY_SLEEP_SECONDS"; then
  echo "--sleep-seconds must be a non-negative integer" >&2
  exit 1
fi

if ! is_non_negative_int "$MAX_RETRIES"; then
  echo "--max-retries must be a non-negative integer" >&2
  exit 1
fi

CMD=("$@")
attempt=1

while true; do
  printf -v cmd_pretty '%q ' "${CMD[@]}"
  cmd_pretty="${cmd_pretty% }"
  echo "[gpu-watchdog] attempt=${attempt} cmd=${cmd_pretty}"

  attempt_log="$(mktemp /tmp/gpu_watchdog_attempt_${attempt}_XXXX.log)"
  rc=0

  set +e
  "${CMD[@]}" 2>&1 | tee -a "$attempt_log"
  rc=${PIPESTATUS[0]}
  set -e

  if (( rc == 0 )); then
    echo "[gpu-watchdog] success attempt=${attempt}"
    exit 0
  fi

  retryable=0
  reason=""

  if (( rc == 137 )); then
    retryable=1
    reason="exit_code_137"
  elif grep -Eqi "$OOM_REGEX" "$attempt_log"; then
    retryable=1
    reason="oom_signature"
  fi

  if (( retryable == 0 )); then
    echo "[gpu-watchdog] non-retryable failure rc=${rc}; stop" >&2
    exit "$rc"
  fi

  if (( APPEND_RESUME_FLAG == 1 )); then
    append_resume_if_needed CMD
  fi

  if (( MAX_RETRIES > 0 && attempt >= MAX_RETRIES )); then
    echo "[gpu-watchdog] retry limit reached (${MAX_RETRIES}), last_reason=${reason}, rc=${rc}" >&2
    exit "$rc"
  fi

  echo "[gpu-watchdog] retryable failure reason=${reason} rc=${rc}; sleep ${RETRY_SLEEP_SECONDS}s then retry"
  sleep "$RETRY_SLEEP_SECONDS"
  attempt=$((attempt + 1))
done