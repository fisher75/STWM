#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
RUN_SCRIPT="$ROOT/scripts/run_ostf_lastobs_residual_v28_20260502.sh"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
LOG_DIR="$ROOT/logs/stwm_ostf_v28_multiseed_20260507"
MANIFEST="$ROOT/reports/stwm_ostf_v28_multiseed_launch_manifest_20260507.json"
STATUS="$ROOT/reports/stwm_ostf_v28_multiseed_status_20260507.json"
mkdir -p "$LOG_DIR" "$ROOT/reports/stwm_ostf_v28_runs"

complete_report() {
  local report="$1"
  "$PY" - "$report" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists() or p.stat().st_size <= 0:
    raise SystemExit(1)
try:
    x = json.loads(p.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
required = ["experiment_name", "item_scores", "test_metrics", "best_checkpoint_path", "final_checkpoint_path", "steps"]
if any(k not in x for k in required):
    raise SystemExit(1)
if not x.get("item_scores"):
    raise SystemExit(1)
root = Path("/raid/chen034/workspace/stwm")
for key in ["best_checkpoint_path", "final_checkpoint_path"]:
    ck = root / str(x.get(key, ""))
    if not ck.exists() or ck.stat().st_size <= 0:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

pick_gpu_table() {
  "$PY" <<'PY'
import subprocess
rows = []
try:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception:
    raw = ""
for line in raw.strip().splitlines():
    parts = [x.strip() for x in line.split(",")]
    if len(parts) != 4:
        continue
    idx, used, free, util = map(int, parts)
    if free < 25 * 1024 or used > 170 * 1024:
        continue
    pref = 0 if free >= 40 * 1024 else 1
    rows.append((pref, -free, util, idx, free, used))
rows.sort()
for pref, neg_free, util, idx, free, used in rows:
    print(f"{idx}\t{free}\t{used}\t{util}")
PY
}

mapfile -t GPU_ROWS < <(pick_gpu_table)
if [[ "${#GPU_ROWS[@]}" -eq 0 ]]; then
  echo "no_eligible_gpus_found_min_free_25gb" >&2
  exit 1
fi

GPU_IDS=()
GPU_FREE=()
for row in "${GPU_ROWS[@]}"; do
  IFS=$'\t' read -r idx free used util <<< "$row"
  GPU_IDS+=("$idx")
  GPU_FREE+=("$free")
done

gpu_at() {
  local i="$1"
  echo "${GPU_IDS[$(( i % ${#GPU_IDS[@]} ))]}"
}

gpu_free_at() {
  local i="$1"
  echo "${GPU_FREE[$(( i % ${#GPU_FREE[@]} ))]}"
}

json_quote() {
  "$PY" -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1"
}

declare -a ROWS
job_index=0

add_row() {
  local status="$1" session="$2" exp="$3" kind="$4" horizon="$5" seed="$6" gpu="$7" free_mem="$8" log="$9" report="${10}" command="${11}" reason="${12}"
  ROWS+=("{\"status\":\"$status\",\"session_name\":\"$session\",\"experiment_name\":\"$exp\",\"model_kind\":\"$kind\",\"horizon\":$horizon,\"seed\":$seed,\"gpu_id\":\"$gpu\",\"free_memory_mb_before_launch\":\"$free_mem\",\"log_path\":$(json_quote "$log"),\"report_path\":$(json_quote "$report"),\"command\":$(json_quote "$command"),\"reason\":$(json_quote "$reason")}")
}

launch_job() {
  local exp="$1" kind="$2" horizon="$3" seed="$4" steps="$5" batch="$6" eval_every="$7"
  local report="$ROOT/reports/stwm_ostf_v28_runs/${exp}.json"
  local session="stwm_v28_ms_${exp}"
  session="${session//[^A-Za-z0-9_]/_}"
  local log="$LOG_DIR/${session}.log"
  if complete_report "$report"; then
    add_row "skipped_completed" "$session" "$exp" "$kind" "$horizon" "$seed" "" "" "$log" "$report" "" "complete_report_and_checkpoints_exist"
    return 0
  fi
  if tmux has-session -t "$session" >/dev/null 2>&1; then
    add_row "already_running" "$session" "$exp" "$kind" "$horizon" "$seed" "" "" "$log" "$report" "" "tmux_session_exists"
    return 0
  fi
  local gpu free_mem cmd
  gpu="$(gpu_at "$job_index")"
  free_mem="$(gpu_free_at "$job_index")"
  job_index=$((job_index + 1))
  cmd="train --experiment-name $exp --model-kind $kind --horizon $horizon --seed $seed --steps $steps --batch-size $batch --device cuda --eval-every $eval_every"
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$ROOT/code:\${PYTHONPATH:-}' STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic STWM_PYTHON='$PY' && bash '$RUN_SCRIPT' $cmd > '$log' 2>&1"
  add_row "launched" "$session" "$exp" "$kind" "$horizon" "$seed" "$gpu" "$free_mem" "$log" "$report" "$cmd" "launched_in_tmux"
  echo "[launch] $session gpu=$gpu free_mb=$free_mem exp=$exp"
}

for seed in 42 123 456 789 2026; do
  launch_job "v28_lastobs_m128_h64_seed${seed}" "v28_lastobs_m128_h64" 64 "$seed" 30000 1 1000
  launch_job "v28_lastobs_m128_h32_seed${seed}" "v28_lastobs_m128_h32" 32 "$seed" 30000 2 1500
done

for seed in 42 123 456; do
  launch_job "v28_lastobs_m128_h64_wo_dense_points_seed${seed}" "v28_lastobs_m128_h64_wo_dense_points" 64 "$seed" 20000 1 1000
  launch_job "v28_lastobs_m128_h64_wo_semantic_memory_seed${seed}" "v28_lastobs_m128_h64_wo_semantic_memory" 64 "$seed" 20000 1 1000
  launch_job "v28_lastobs_m128_h64_wo_residual_modes_seed${seed}" "v28_lastobs_m128_h64_wo_residual_modes" 64 "$seed" 20000 1 1000
  launch_job "v28_lastobs_m128_h64_prior_only_seed${seed}" "v28_lastobs_m128_h64_prior_only" 64 "$seed" 10000 1 1000
done

"$PY" - "$MANIFEST" "${ROWS[@]}" <<'PY'
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(x) for x in sys.argv[2:]]
counts = Counter(r["status"] for r in rows)
payload = {
    "manifest_name": "stwm_ostf_v28_multiseed_launch_manifest",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "root": "/raid/chen034/workspace/stwm",
    "gpu_policy": {
        "preferred_free_mem_gb": 40,
        "min_free_mem_gb": 25,
        "avoid_memory_used_gt_gb": 170,
    },
    "expected_run_count": len(rows),
    "status_counts": dict(counts),
    "launches": rows,
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY

"$PY" - "$MANIFEST" "$STATUS" <<'PY'
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = manifest.get("launches", [])
active = []
for row in rows:
    session = row.get("session_name")
    if not session:
        continue
    ok = subprocess.run(["tmux", "has-session", "-t", session], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if ok:
        active.append(session)
counts = Counter(r.get("status") for r in rows)
payload = {
    "status_name": "stwm_ostf_v28_multiseed_status",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "partial": True,
    "manifest_path": str(Path(sys.argv[1])),
    "status_counts": dict(counts),
    "active_tmux_session_count": len(active),
    "active_tmux_sessions": active,
    "note": "Status is written immediately after launch; run reports complete asynchronously in reports/stwm_ostf_v28_runs/.",
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(sys.argv[2])
PY
