#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

QUEUE_DIR="${STWM_GPU_QUEUE_DIR:-$STWM_ROOT/outputs/queue/stwm_gpu}"
JOB_NAME=""
WORKDIR="$STWM_ROOT"

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
  gpu_queue_submit.sh [options] -- <command> [args...]

Description:
  Submit one job into FIFO GPU queue. Jobs are consumed by gpu_queue_worker.sh.

Options:
  --queue-dir PATH          Queue root directory (default: outputs/queue/stwm_gpu)
  --job-name NAME           Human-readable job name
  --workdir PATH            Working directory for the command (default: STWM root)

  --prefer-gpus N           Prefer up to N GPUs when available (default: 8)
  --min-gpus N              Minimum GPUs required to start (default: 1)
  --poll-seconds N          Poll interval while claiming GPUs (default: 30)
  --max-mem-used-mib N      Idle threshold for memory used (default: 2000)
  --max-utilization N       Idle threshold for utilization % (default: 20)
  --candidate-gpus CSV      Restrict candidate GPU ids (e.g. 0,1,2,3)
  --timeout-seconds N       Timeout while waiting for GPUs (0 = wait forever)
  -h, --help                Show this message

Examples:
  gpu_queue_submit.sh --job-name 1b_smoke -- -- bash scripts/run_stwm_v4_2_1b_smoke.sh
  gpu_queue_submit.sh --job-name 1b_confirm --prefer-gpus 8 --min-gpus 4 -- \
    bash scripts/run_stwm_v4_2_1b_confirmation_round.sh
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --queue-dir" >&2; exit 1; }
      QUEUE_DIR="$2"
      shift 2
      ;;
    --job-name)
      [[ $# -ge 2 ]] || { echo "Missing value for --job-name" >&2; exit 1; }
      JOB_NAME="$2"
      shift 2
      ;;
    --workdir)
      [[ $# -ge 2 ]] || { echo "Missing value for --workdir" >&2; exit 1; }
      WORKDIR="$2"
      shift 2
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

ensure_dir "$QUEUE_DIR/pending" "$QUEUE_DIR/running" "$QUEUE_DIR/done" "$QUEUE_DIR/failed" "$QUEUE_DIR/logs"

if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="$(basename "$1")"
fi

safe_name="$(echo "$JOB_NAME" | tr -cs '[:alnum:]_-' '_' | sed 's/^_\+//;s/_\+$//')"
if [[ -z "$safe_name" ]]; then
  safe_name="job"
fi

stamp="$(date +%Y%m%d_%H%M%S)"
ms="$(date +%s%3N 2>/dev/null || date +%s000)"
job_id="${stamp}_${RANDOM}"
job_file="$QUEUE_DIR/pending/${ms}_${safe_name}.job"

printf -v cmd_q '%q ' "$@"
cmd_pretty="$*"

{
  printf 'job_id=%q\n' "$job_id"
  printf 'job_name=%q\n' "$JOB_NAME"
  printf 'submit_ts=%q\n' "$(timestamp)"
  printf 'workdir=%q\n' "$WORKDIR"
  printf 'prefer_gpus=%q\n' "$PREFER_GPUS"
  printf 'min_gpus=%q\n' "$MIN_GPUS"
  printf 'poll_seconds=%q\n' "$POLL_SECONDS"
  printf 'max_mem_used_mib=%q\n' "$MAX_MEM_USED_MIB"
  printf 'max_util_percent=%q\n' "$MAX_UTIL_PERCENT"
  printf 'candidate_gpus=%q\n' "$CANDIDATE_GPUS"
  printf 'timeout_seconds=%q\n' "$TIMEOUT_SECONDS"
  printf 'command_escaped=%q\n' "$cmd_q"
  printf 'command_pretty=%q\n' "$cmd_pretty"
} > "$job_file"

pending_count="$(find "$QUEUE_DIR/pending" -maxdepth 1 -type f -name '*.job' | wc -l | tr -d ' ')"

echo "[gpu-queue-submit] queued"
echo "  job_id:      $job_id"
echo "  job_name:    $JOB_NAME"
echo "  job_file:    $job_file"
echo "  position:    $pending_count"
echo "  worker_cmd:  bash scripts/gpu_queue_worker.sh --queue-dir $QUEUE_DIR"
