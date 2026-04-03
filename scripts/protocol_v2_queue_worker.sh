#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

QUEUE_DIR="${STWM_PROTOCOL_V2_QUEUE_DIR:-$STWM_ROOT/outputs/queue/stwm_protocol_v2/d0_eval}"
CLASS_TYPE="A"
IDLE_SLEEP="15"
STOP_WHEN_EMPTY=0

usage() {
  cat <<'USAGE'
Usage:
  protocol_v2_queue_worker.sh [options]

Options:
  --queue-dir PATH          Queue directory (d0_eval or d1_train)
  --class-type {A|B|C}      Scheduler class type for this worker
  --idle-sleep N            Idle sleep seconds when queue empty or no GPU
  --stop-when-empty         Exit when pending queue is empty
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
    --stop-when-empty)
      STOP_WHEN_EMPTY=1
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

CLASS_TYPE="$(echo "$CLASS_TYPE" | tr '[:lower:]' '[:upper:]')"
if [[ "$CLASS_TYPE" != "A" && "$CLASS_TYPE" != "B" && "$CLASS_TYPE" != "C" ]]; then
  echo "class-type must be A/B/C" >&2
  exit 1
fi

ensure_dir "$QUEUE_DIR/pending" "$QUEUE_DIR/running" "$QUEUE_DIR/done" "$QUEUE_DIR/failed" "$QUEUE_DIR/logs" "$QUEUE_DIR/status" "$QUEUE_DIR/pids"
QUEUE_ROOT="$(dirname "$QUEUE_DIR")"
ensure_dir "$QUEUE_ROOT/leases"

WORKER_LOCK="$QUEUE_DIR/.worker.lock"
exec 9>"$WORKER_LOCK"
if ! flock -n 9; then
  echo "[protocol-v2-worker] another worker holds lock for queue: $QUEUE_DIR" >&2
  exit 3
fi

EVENT_LOG="$QUEUE_DIR/queue_events.log"
GPU_SELECT_LOCK="$QUEUE_ROOT/.gpu_select.lock"

event() {
  local level="$1"
  shift
  printf '%s\t%s\t%s\n' "$(timestamp)" "$level" "$*" | tee -a "$EVENT_LOG"
}

pick_next_job() {
  find "$QUEUE_DIR/pending" -maxdepth 1 -type f -name '*.job' | sort | head -n 1
}

update_status() {
  local status_path="$1"
  local state="$2"
  local job_id="$3"
  local job_name="$4"
  local class_type="$5"
  local job_file="$6"
  local main_log="$7"
  local pid_file="$8"
  local gpu_index="$9"
  local worker_session="${10}"
  local message="${11}"
  local exit_code="${12}"

  python - "$status_path" "$state" "$job_id" "$job_name" "$class_type" "$job_file" "$main_log" "$pid_file" "$gpu_index" "$worker_session" "$message" "$exit_code" <<'PY'
from pathlib import Path
import json
import time
import sys

status_path = Path(sys.argv[1])
state = sys.argv[2]
job_id = sys.argv[3]
job_name = sys.argv[4]
class_type = sys.argv[5]
job_file = sys.argv[6]
main_log = sys.argv[7]
pid_file = sys.argv[8]
gpu_index = sys.argv[9]
worker_session = sys.argv[10]
message = sys.argv[11]
exit_code = sys.argv[12]

payload = {}
if status_path.exists():
    try:
        payload = json.loads(status_path.read_text())
    except Exception:
        payload = {}

payload.update(
    {
        "state": state,
        "job_id": job_id,
        "job_name": job_name,
        "class_type": class_type,
        "job_file": job_file,
        "main_log": main_log,
        "pid_file": pid_file,
        "gpu_index": gpu_index,
        "worker_session": worker_session,
        "message": message,
        "update_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
)
if exit_code != "":
    payload["exit_code"] = int(float(exit_code))

status_path.write_text(json.dumps(payload, indent=2))
PY
}

cleanup_stale_leases() {
  local lease
  for lease in "$QUEUE_ROOT"/leases/*.json; do
    [[ -e "$lease" ]] || continue
    local pid
    pid="$(python - "$lease" <<'PY'
from pathlib import Path
import json
import sys

p = Path(sys.argv[1])
try:
    obj = json.loads(p.read_text())
    print(obj.get("pid", ""))
except Exception:
    print("")
PY
)"
    if [[ -n "$pid" && "$pid" =~ ^[0-9]+$ ]]; then
      if [[ ! -d "/proc/$pid" ]]; then
        rm -f "$lease"
      fi
    fi
  done
}

event INFO "worker_started queue_dir=$QUEUE_DIR class_type=$CLASS_TYPE idle_sleep=${IDLE_SLEEP}s stop_when_empty=$STOP_WHEN_EMPTY"

while true; do
  next_job="$(pick_next_job)"
  if [[ -z "$next_job" ]]; then
    if (( STOP_WHEN_EMPTY == 1 )); then
      event INFO "worker_exit queue_empty=1"
      exit 0
    fi
    sleep "$IDLE_SLEEP"
    continue
  fi

  base="$(basename "$next_job")"
  running_path="$QUEUE_DIR/running/$base"
  mv "$next_job" "$running_path"

  # shellcheck disable=SC1090
  source "$running_path"

  job_id="${job_id:-unknown}"
  job_name="${job_name:-unknown}"
  job_class="${class_type:-$CLASS_TYPE}"
  workdir="${workdir:-$STWM_ROOT}"
  command_escaped="${command_escaped:-}"
  command_pretty="${command_pretty:-unknown}"
  status_file="${status_file:-$QUEUE_DIR/status/${base%.job}.status.json}"
  pid_file="${pid_file:-$QUEUE_DIR/pids/${base%.job}.pid}"
  main_log="${main_log:-$QUEUE_DIR/logs/${base%.job}.log}"
  notes="${notes:-}"
  resume_hint="${resume_hint:-}"

  worker_session="${TMUX_SESSION:-$(tmux display-message -p '#S' 2>/dev/null || echo unknown)}"

  event INFO "job_start id=$job_id name=$job_name class=$job_class file=$running_path"
  event INFO "job_cmd id=$job_id cmd=$command_pretty"
  if [[ -n "$notes" ]]; then
    event INFO "job_notes id=$job_id notes=$notes"
  fi
  if [[ -n "$resume_hint" ]]; then
    event INFO "job_resume_hint id=$job_id hint=$resume_hint"
  fi

  update_status "$status_file" "waiting_for_gpu" "$job_id" "$job_name" "$job_class" "$running_path" "$main_log" "$pid_file" "" "$worker_session" "waiting_for_gpu" ""

  chosen_gpu=""
  chosen_reason=""
  select_json_file="$QUEUE_DIR/logs/${base%.job}.gpu_select.json"

  while true; do
    exec 8>"$GPU_SELECT_LOCK"
    flock 8

    cleanup_stale_leases

    select_json="$(python "$SCRIPT_DIR/select_protocol_v2_gpu.py" --class-type "$job_class" --queue-root "$QUEUE_ROOT")"
    printf '%s\n' "$select_json" > "$select_json_file"

    status="$(python - "$select_json_file" <<'PY'
from pathlib import Path
import json
import sys
obj = json.loads(Path(sys.argv[1]).read_text())
print(obj.get("status", ""))
PY
)"

    snapshot_compact="$(python - "$select_json_file" <<'PY'
from pathlib import Path
import json
import sys
obj = json.loads(Path(sys.argv[1]).read_text())
print(json.dumps(obj.get("snapshot", []), ensure_ascii=True, separators=(",", ":")))
PY
)"
    event INFO "gpu_snapshot id=$job_id snapshot=$snapshot_compact"

    if [[ "$status" == "ok" ]]; then
      chosen_gpu="$(python - "$select_json_file" <<'PY'
from pathlib import Path
import json
import sys
obj = json.loads(Path(sys.argv[1]).read_text())
chosen = obj.get("chosen") or {}
print(chosen.get("index", ""))
PY
)"
      chosen_reason="$(python - "$select_json_file" <<'PY'
from pathlib import Path
import json
import sys
obj = json.loads(Path(sys.argv[1]).read_text())
chosen = obj.get("chosen") or {}
print(chosen.get("eligible_reason", "eligible"))
PY
)"

      lease_file="$QUEUE_ROOT/leases/${job_id}.json"
      python - "$lease_file" "$job_id" "$job_class" "$chosen_gpu" <<'PY'
from pathlib import Path
import json
import os
import time
import sys

lease_path = Path(sys.argv[1])
obj = {
    "job_id": sys.argv[2],
    "class_type": sys.argv[3],
    "gpu_index": int(float(sys.argv[4])),
    "pid": int(os.getpid()),
    "lease_ts": time.strftime("%Y-%m-%d %H:%M:%S"),
}
lease_path.write_text(json.dumps(obj, indent=2))
PY

      event INFO "gpu_chosen id=$job_id gpu=$chosen_gpu reason=$chosen_reason"
      update_status "$status_file" "running" "$job_id" "$job_name" "$job_class" "$running_path" "$main_log" "$pid_file" "$chosen_gpu" "$worker_session" "gpu_assigned:$chosen_reason" ""
      flock -u 8
      break
    fi

    event INFO "gpu_wait id=$job_id reason=no_eligible_gpu"
    flock -u 8
    sleep "$IDLE_SLEEP"
  done

  rc=0
  set +e
  (
    cd "$workdir"
    export CUDA_VISIBLE_DEVICES="$chosen_gpu"
    export STWM_ASSIGNED_GPUS="$chosen_gpu"
    export STWM_ASSIGNED_GPU_COUNT="1"
    bash -lc "$command_escaped"
  ) >> "$main_log" 2>&1 &
  child_pid=$!
  echo "$child_pid" > "$pid_file"
  wait "$child_pid"
  rc=$?
  set -e

  rm -f "$QUEUE_ROOT/leases/${job_id}.json"

  {
    printf 'finish_ts=%q\n' "$(timestamp)"
    printf 'exit_code=%q\n' "$rc"
    printf 'assigned_gpu=%q\n' "$chosen_gpu"
    printf 'main_log=%q\n' "$main_log"
    printf 'status_file=%q\n' "$status_file"
    printf 'pid_file=%q\n' "$pid_file"
  } >> "$running_path"

  if (( rc == 0 )); then
    mv "$running_path" "$QUEUE_DIR/done/$base"
    update_status "$status_file" "done" "$job_id" "$job_name" "$job_class" "$QUEUE_DIR/done/$base" "$main_log" "$pid_file" "$chosen_gpu" "$worker_session" "completed" "$rc"
    event INFO "job_done id=$job_id gpu=$chosen_gpu exit=$rc log=$main_log status=$status_file"
  else
    mv "$running_path" "$QUEUE_DIR/failed/$base"
    update_status "$status_file" "failed" "$job_id" "$job_name" "$job_class" "$QUEUE_DIR/failed/$base" "$main_log" "$pid_file" "$chosen_gpu" "$worker_session" "failed" "$rc"
    event ERROR "job_failed id=$job_id gpu=$chosen_gpu exit=$rc log=$main_log status=$status_file"
  fi
done
