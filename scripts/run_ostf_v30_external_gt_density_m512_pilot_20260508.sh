#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT"

LOG_DIR="$ROOT/logs/stwm_ostf_v30_extgt_density_m512_pilot_runs_20260508"
mkdir -p "$LOG_DIR" "$ROOT/reports"

gpu_list() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return 0
  fi
  nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.used --format=csv,noheader,nounits \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); gsub(/ /,"",$4); if (($3+0)>=25000 && ($4+0)<=170000) print ($2+0)" "($3+0)" "$1}' \
    | sort -k1,1n -k2,2nr \
    | awk '{print $3}'
}

mapfile -t GPUS < <(gpu_list)
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  GPUS=("")
fi
MAX_PARALLEL="${V30_DENSITY_MAX_PARALLEL:-${#GPUS[@]}}"
if [[ "$MAX_PARALLEL" -lt 1 ]]; then MAX_PARALLEL=1; fi

echo "[V30 density M512 pilot parallel] candidate_gpus=${GPUS[*]:-cpu} max_parallel=$MAX_PARALLEL"
"$PY" code/stwm/tools/audit_ostf_v30_density_scaling_readiness_20260508.py
"$PY" code/stwm/tools/audit_ostf_v30_density_code_20260508.py
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('/raid/chen034/workspace/stwm')
ready=json.loads((root/'reports/stwm_ostf_v30_density_scaling_readiness_audit_20260508.json').read_text())
code=json.loads((root/'reports/stwm_ostf_v30_density_code_audit_20260508.json').read_text())
if not ready.get('density_scaling_ready'):
    raise SystemExit(f"density scaling not ready: {ready.get('exact_blocker')}")
if code.get('fatal_issue_found'):
    raise SystemExit('density code audit failed')
PY

is_compatible() {
  local name="$1"
  "$PY" - "$name" <<'PY'
import json, sys
from pathlib import Path
root=Path('/raid/chen034/workspace/stwm')
path=root/'reports/stwm_ostf_v30_external_gt_runs'/f'{sys.argv[1]}.json'
ok=False
if path.exists() and path.stat().st_size>0:
    try:
        p=json.loads(path.read_text())
        ok=bool(
            p.get('completed') and p.get('test_item_rows') and p.get('checkpoint_path')
            and int(p.get('steps', -1)) == 4000
            and p.get('batch_size') is not None
            and p.get('eval_interval') == 1000
            and p.get('setproctitle_status', {}).get('requested_title') == 'python'
        )
    except Exception:
        ok=False
raise SystemExit(0 if ok else 1)
PY
}

launch_one() {
  local gpu="$1" name="$2" h="$3" seed="$4" batch="$5"
  local log="$LOG_DIR/${name}.log"
  if is_compatible "$name"; then
    echo "[V30 density M512 pilot parallel] skip compatible $name"
    return 0
  fi
  echo "[V30 density M512 pilot parallel] launch $name gpu=${gpu:-cpu} log=${log#$ROOT/}"
  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py \
      --experiment-name "$name" --horizon "$h" --m-points 512 --seed "$seed" \
      --steps "${V30_DENSITY_STEPS:-4000}" --batch-size "$batch" --eval-interval 1000 --amp \
      > "$log" 2>&1 &
  else
    "$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py \
      --experiment-name "$name" --horizon "$h" --m-points 512 --seed "$seed" \
      --steps "${V30_DENSITY_STEPS:-4000}" --batch-size "$batch" --eval-interval 1000 --amp \
      > "$log" 2>&1 &
  fi
  PIDS+=("$!")
  NAMES+=("$name")
}

PIDS=()
NAMES=()
gpu_idx=0
for seed in 42 123 456; do
  for h in 32 64 96; do
    batch="${V30_DENSITY_H32_BATCH:-8}"
    if [[ "$h" == "64" ]]; then batch="${V30_DENSITY_H64_BATCH:-8}"; fi
    if [[ "$h" == "96" ]]; then batch="${V30_DENSITY_H96_BATCH:-4}"; fi
    gpu="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
    gpu_idx=$((gpu_idx + 1))
    launch_one "$gpu" "v30_extgt_m512_h${h}_seed${seed}" "$h" "$seed" "$batch"
    while [[ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]]; do
      sleep 5
    done
  done
done

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  cat > "$ROOT/reports/stwm_ostf_v30_density_m512_pilot_status_20260508.json" <<JSON
{
  "session": "stwm_ostf_v30_extgt_density_m512_pilot_20260508",
  "log": "logs/stwm_ostf_v30_extgt_density_m512_pilot_20260508.log",
  "status": "failed",
  "updated_at_local": "$(date -Iseconds)",
  "run_logs_dir": "${LOG_DIR#$ROOT/}"
}
JSON
  exit 1
fi

"$PY" code/stwm/tools/aggregate_ostf_external_gt_v30_density_m512_pilot_20260508.py
"$PY" code/stwm/tools/write_ostf_v30_density_scaling_decision_20260508.py
cat > "$ROOT/reports/stwm_ostf_v30_density_m512_pilot_status_20260508.json" <<JSON
{
  "session": "stwm_ostf_v30_extgt_density_m512_pilot_20260508",
  "log": "logs/stwm_ostf_v30_extgt_density_m512_pilot_20260508.log",
  "status": "completed",
  "updated_at_local": "$(date -Iseconds)",
  "run_logs_dir": "${LOG_DIR#$ROOT/}",
  "decision_path": "reports/stwm_ostf_v30_density_m512_pilot_decision_20260508.json"
}
JSON
