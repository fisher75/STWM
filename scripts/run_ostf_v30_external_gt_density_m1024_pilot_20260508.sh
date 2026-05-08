#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT"
LOG_DIR="$ROOT/logs/stwm_ostf_v30_extgt_density_m1024_pilot_runs_20260508"
mkdir -p "$LOG_DIR" "$ROOT/reports"
gpu_list() {
  nvidia-smi --query-gpu=index,utilization.gpu,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); gsub(/ /,"",$4); if (($3+0)>=25000 && ($4+0)<=170000) print ($2+0)" "($3+0)" "$1}' \
    | sort -k1,1n -k2,2nr | awk '{print $3}'
}
mapfile -t GPUS < <(gpu_list || true)
if [[ "${#GPUS[@]}" -eq 0 ]]; then GPUS=(""); fi
echo "[V30 density M1024 pilot] candidate_gpus=${GPUS[*]:-cpu}"
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('/raid/chen034/workspace/stwm')
smoke=json.loads((root/'reports/stwm_ostf_v30_density_m1024_smoke_summary_20260508.json').read_text())
m512=json.loads((root/'reports/stwm_ostf_v30_density_m512_pilot_decision_20260508.json').read_text())
if not smoke.get('m1024_smoke_passed'):
    raise SystemExit('M1024 smoke did not pass')
if not m512.get('density_scaling_positive_preliminary'):
    raise SystemExit('M512 pilot was not positive enough for M1024 pilot')
PY
launch() {
  local gpu="$1" name="$2" h="$3" seed="$4"
  local log="$LOG_DIR/${name}.log"
  echo "[V30 density M1024 pilot] launch $name gpu=${gpu:-cpu}"
  if [[ -n "$gpu" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py \
      --experiment-name "$name" --horizon "$h" --m-points 1024 --seed "$seed" \
      --steps "${V30_M1024_PILOT_STEPS:-4000}" --batch-size "${V30_M1024_BATCH:-1}" --grad-accum-steps "${V30_M1024_GRAD_ACCUM:-8}" \
      --eval-interval 1000 --amp > "$log" 2>&1 &
  else
    "$PY" code/stwm/tools/train_ostf_external_gt_v30_20260508.py \
      --experiment-name "$name" --horizon "$h" --m-points 1024 --seed "$seed" \
      --steps "${V30_M1024_PILOT_STEPS:-4000}" --batch-size "${V30_M1024_BATCH:-1}" --grad-accum-steps "${V30_M1024_GRAD_ACCUM:-8}" \
      --eval-interval 1000 --amp > "$log" 2>&1 &
  fi
  PIDS+=("$!")
}
PIDS=()
i=0
for seed in 42 123; do
  for h in 32 64 96; do
    launch "${GPUS[$((i % ${#GPUS[@]}))]}" "v30_extgt_m1024_h${h}_seed${seed}" "$h" "$seed"
    i=$((i+1))
  done
done
failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
"$PY" code/stwm/tools/aggregate_ostf_external_gt_v30_density_m1024_pilot_20260508.py
"$PY" code/stwm/tools/write_ostf_v30_density_scaling_decision_20260508.py
cat > "$ROOT/reports/stwm_ostf_v30_density_m1024_pilot_status_20260508.json" <<JSON
{
  "session": "stwm_ostf_v30_extgt_density_m1024_pilot_20260508",
  "log": "logs/stwm_ostf_v30_extgt_density_m1024_pilot_20260508.log",
  "status": "$([[ "$failed" == "0" ]] && echo completed || echo failed)",
  "updated_at_local": "$(date -Iseconds)",
  "run_logs_dir": "${LOG_DIR#$ROOT/}",
  "decision_path": "reports/stwm_ostf_v30_density_m1024_pilot_decision_20260508.json"
}
JSON
exit "$failed"
