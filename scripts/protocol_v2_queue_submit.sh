#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

QUEUE_DIR="${STWM_PROTOCOL_V2_QUEUE_DIR:-$STWM_ROOT/outputs/queue/stwm_protocol_v2/d0_eval}"
JOB_NAME=""
CLASS_TYPE="A"
WORKDIR="$STWM_ROOT"
NOTES=""
RESUME_HINT=""

usage() {
  cat <<'USAGE'
Usage:
  protocol_v2_queue_submit.sh [options] -- <command> [args...]

Options:
  --queue-dir PATH          Queue directory (d0_eval or d1_train)
  --job-name NAME           Human-readable job name
  --class-type {A|B|C}      Scheduling class type
  --workdir PATH            Working directory
  --notes TEXT              Notes for status metadata
  --resume-hint TEXT        Resume hint text for status metadata
  -h, --help                Show help
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
    --class-type)
      [[ $# -ge 2 ]] || { echo "Missing value for --class-type" >&2; exit 1; }
      CLASS_TYPE="$2"
      shift 2
      ;;
    --workdir)
      [[ $# -ge 2 ]] || { echo "Missing value for --workdir" >&2; exit 1; }
      WORKDIR="$2"
      shift 2
      ;;
    --notes)
      [[ $# -ge 2 ]] || { echo "Missing value for --notes" >&2; exit 1; }
      NOTES="$2"
      shift 2
      ;;
    --resume-hint)
      [[ $# -ge 2 ]] || { echo "Missing value for --resume-hint" >&2; exit 1; }
      RESUME_HINT="$2"
      shift 2
      ;;
    --)
      shift
      break
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

if [[ $# -eq 0 ]]; then
  echo "You must provide command after --" >&2
  usage
  exit 1
fi

CLASS_TYPE="$(echo "$CLASS_TYPE" | tr '[:lower:]' '[:upper:]')"
if [[ "$CLASS_TYPE" != "A" && "$CLASS_TYPE" != "B" && "$CLASS_TYPE" != "C" ]]; then
  echo "class-type must be A/B/C" >&2
  exit 1
fi

ensure_dir "$QUEUE_DIR/pending" "$QUEUE_DIR/running" "$QUEUE_DIR/done" "$QUEUE_DIR/failed" "$QUEUE_DIR/logs" "$QUEUE_DIR/status" "$QUEUE_DIR/pids"

if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="$(basename "$1")"
fi

safe_name="$(echo "$JOB_NAME" | tr -cs '[:alnum:]_-' '_' | sed 's/^_\+//;s/_\+$//')"
[[ -n "$safe_name" ]] || safe_name="job"

stamp="$(date +%Y%m%d_%H%M%S)"
ms="$(date +%s%3N 2>/dev/null || date +%s000)"
job_id="${stamp}_${RANDOM}"
job_file="$QUEUE_DIR/pending/${ms}_${safe_name}.job"
status_file="$QUEUE_DIR/status/${ms}_${safe_name}.status.json"
pid_file="$QUEUE_DIR/pids/${ms}_${safe_name}.pid"
main_log="$QUEUE_DIR/logs/${ms}_${safe_name}.log"

CMD=("$@")
printf -v cmd_q '%q ' "${CMD[@]}"
cmd_q="${cmd_q% }"
printf -v cmd_pretty '%q ' "${CMD[@]}"
cmd_pretty="${cmd_pretty% }"

{
  printf 'job_id=%q\n' "$job_id"
  printf 'job_name=%q\n' "$JOB_NAME"
  printf 'class_type=%q\n' "$CLASS_TYPE"
  printf 'submit_ts=%q\n' "$(timestamp)"
  printf 'workdir=%q\n' "$WORKDIR"
  printf 'command_escaped=%q\n' "$cmd_q"
  printf 'command_pretty=%q\n' "$cmd_pretty"
  printf 'status_file=%q\n' "$status_file"
  printf 'pid_file=%q\n' "$pid_file"
  printf 'main_log=%q\n' "$main_log"
  printf 'notes=%q\n' "$NOTES"
  printf 'resume_hint=%q\n' "$RESUME_HINT"
} > "$job_file"

python - "$status_file" "$job_id" "$JOB_NAME" "$CLASS_TYPE" "$job_file" "$main_log" "$pid_file" "$NOTES" "$RESUME_HINT" <<'PY'
from pathlib import Path
import json
import sys

status_path = Path(sys.argv[1])
payload = {
    "state": "queued",
    "job_id": sys.argv[2],
    "job_name": sys.argv[3],
    "class_type": sys.argv[4],
    "job_file": sys.argv[5],
    "main_log": sys.argv[6],
    "pid_file": sys.argv[7],
    "notes": sys.argv[8],
    "resume_hint": sys.argv[9],
}
status_path.write_text(json.dumps(payload, indent=2))
PY

pending_count="$(find "$QUEUE_DIR/pending" -maxdepth 1 -type f -name '*.job' | wc -l | tr -d ' ')"

echo "[protocol-v2-queue-submit] queued"
echo "  job_id:      $job_id"
echo "  class_type:  $CLASS_TYPE"
echo "  job_file:    $job_file"
echo "  status_file: $status_file"
echo "  pid_file:    $pid_file"
echo "  main_log:    $main_log"
echo "  position:    $pending_count"
