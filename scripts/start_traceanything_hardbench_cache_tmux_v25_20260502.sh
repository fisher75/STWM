#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
RUN_SCRIPT="$ROOT/scripts/run_traceanything_hardbench_cache_v25_20260502.sh"
LOG_DIR="$ROOT/logs/traceanything_hardbench_v25_20260502"
REPORT_DIR="$ROOT/reports/stwm_traceanything_hardbench_v25_shards"
DOC_DIR="$ROOT/docs/stwm_traceanything_hardbench_v25_shards"
OUT_ROOT="$ROOT/outputs/cache/stwm_traceanything_hardbench_v25"
MANIFEST="$ROOT/reports/stwm_traceanything_hardbench_launch_manifest_v25_20260502.json"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$DOC_DIR" "$OUT_ROOT"

INITIAL_CONCURRENCY="${INITIAL_CONCURRENCY:-4}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
SLEEP_AFTER_INITIAL="${SLEEP_AFTER_INITIAL:-30}"
MIN_FREE_MEM_GB="${MIN_FREE_MEM_GB:-25}"
PREFERRED_FREE_MEM_GB="${PREFERRED_FREE_MEM_GB:-40}"
OOM_MAX_SIDE="${OOM_MAX_SIDE:-384}"
PRIMARY_MAX_SIDE="${PRIMARY_MAX_SIDE:-512}"

pick_gpus() {
  local count="$1"
  python - "$count" "$MIN_FREE_MEM_GB" "$PREFERRED_FREE_MEM_GB" <<'PY'
import csv, subprocess, sys
count = int(sys.argv[1])
min_free = float(sys.argv[2]) * 1024.0
preferred = float(sys.argv[3]) * 1024.0
raw = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ],
    text=True,
)
rows = []
for line in raw.strip().splitlines():
    idx, used, free, total, util = [x.strip() for x in line.split(",")]
    idx_i = int(idx)
    used_i = int(used)
    free_i = int(free)
    util_i = int(util)
    if free_i < min_free:
        continue
    if used_i > 170 * 1024:
        continue
    rows.append((0 if free_i >= preferred else 1, -free_i, util_i, idx_i, free_i, used_i))
rows.sort()
picked = [str(r[3]) for r in rows[:count]]
print(" ".join(picked))
PY
}

launch_one() {
  local horizon="$1"
  local shard="$2"
  local gpu="$3"
  local session="stwm_ta_v25_h${horizon}_s${shard}"
  local log="$LOG_DIR/${session}.log"
  local report="$REPORT_DIR/${session}.json"
  local doc="$DOC_DIR/${session}.md"
  local cmd="cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$gpu' && export STWM_PROC_TITLE=python && export STWM_PROC_TITLE_MODE=generic && '$RUN_SCRIPT' --horizon '$horizon' --selection-horizon '$horizon' --max-clips 300 --num-shards 4 --shard-index '$shard' --max-side '$PRIMARY_MAX_SIDE' --fallback-max-side '$OOM_MAX_SIDE' --out-root '$OUT_ROOT' --resume --report-path '$report' --doc-path '$doc' > '$log' 2>&1"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "$cmd"
  tmux set-environment -t "$session" CUDA_VISIBLE_DEVICES "$gpu"
  local free_mem
  free_mem="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, -v g="$gpu" '$1==g {gsub(/ /,"",$2); print $2}')"
  python - "$MANIFEST" "$session" "$horizon" "$shard" "$gpu" "$free_mem" "$log" "$report" "$cmd" <<'PY'
import json, sys
from pathlib import Path
manifest_path = Path(sys.argv[1])
session, horizon, shard, gpu, free_mem, log, report, cmd = sys.argv[2:]
payload = {"launch_manifest": []}
if manifest_path.exists():
    payload = json.loads(manifest_path.read_text())
rows = [r for r in payload.get("launch_manifest", []) if not (r.get("session_name") == session)]
rows.append(
    {
        "session_name": session,
        "horizon": int(horizon),
        "shard_index": int(shard),
        "gpu_id": int(gpu),
        "free_memory_before_launch_mib": int(float(free_mem or 0)),
        "log_path": str(Path(log).relative_to('/raid/chen034/workspace/stwm')),
        "report_path": str(Path(report).relative_to('/raid/chen034/workspace/stwm')),
        "command": cmd,
        "min_free_mem_gb": 25,
        "preferred_free_mem_gb": 40,
        "retry_policy": "retry_on_oom_with_max_side_384",
        "status": "launched",
    }
)
manifest_path.write_text(json.dumps({"launch_manifest": rows}, indent=2) + "\n")
PY
  echo "$session gpu=$gpu log=$log report=$report"
}

gpu_list=($(pick_gpus "$MAX_CONCURRENCY"))
if [[ "${#gpu_list[@]}" -eq 0 ]]; then
  echo "no_eligible_gpus_found" >&2
  exit 1
fi

declare -a plan=(
  "32 0"
  "32 1"
  "32 2"
  "32 3"
  "64 0"
  "64 1"
  "64 2"
  "64 3"
)

python - "$MANIFEST" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({"launch_manifest": []}, indent=2) + "\n")
PY

launched=0
for idx in "${!plan[@]}"; do
  read -r h s <<<"${plan[$idx]}"
  gpu="${gpu_list[$((idx % ${#gpu_list[@]}))]}"
  if [[ "$launched" -lt "$INITIAL_CONCURRENCY" ]]; then
    launch_one "$h" "$s" "$gpu"
    launched=$((launched + 1))
  fi
done

if [[ "$MAX_CONCURRENCY" -gt "$INITIAL_CONCURRENCY" ]]; then
  sleep "$SLEEP_AFTER_INITIAL"
  for idx in "${!plan[@]}"; do
    if [[ "$idx" -lt "$INITIAL_CONCURRENCY" ]]; then
      continue
    fi
    if [[ "$launched" -ge "$MAX_CONCURRENCY" ]]; then
      break
    fi
    read -r h s <<<"${plan[$idx]}"
    gpu="${gpu_list[$((idx % ${#gpu_list[@]}))]}"
    launch_one "$h" "$s" "$gpu"
    launched=$((launched + 1))
  done
fi

echo "$MANIFEST"
