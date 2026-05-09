#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT"

LOG_DIR="$ROOT/logs/stwm_ostf_v31_field_ablation_burst_20260509"
RUN_DIR="$ROOT/reports/stwm_ostf_v31_field_preserving_runs"
MANIFEST="$ROOT/reports/stwm_ostf_v31_field_ablation_burst_launch_manifest_20260509.json"
STATUS="$ROOT/reports/stwm_ostf_v31_field_ablation_burst_status_20260509.json"
mkdir -p "$LOG_DIR" "$RUN_DIR" "$ROOT/reports"

batch_for() {
  local m="$1" h="$2"
  if [[ "$m" -eq 128 && "$h" -eq 32 ]]; then echo 96; return; fi
  if [[ "$m" -eq 128 && "$h" -eq 64 ]]; then echo 72; return; fi
  if [[ "$m" -eq 128 && "$h" -eq 96 ]]; then echo 56; return; fi
  if [[ "$m" -eq 512 && "$h" -eq 32 ]]; then echo 32; return; fi
  if [[ "$m" -eq 512 && "$h" -eq 64 ]]; then echo 24; return; fi
  if [[ "$m" -eq 512 && "$h" -eq 96 ]]; then echo 16; return; fi
  echo 16
}

mem_need() {
  local m="$1" h="$2"
  if [[ "$m" -eq 512 && "$h" -eq 96 ]]; then echo 36000; return; fi
  if [[ "$m" -eq 512 && "$h" -eq 64 ]]; then echo 30000; return; fi
  if [[ "$m" -eq 512 && "$h" -eq 32 ]]; then echo 24000; return; fi
  if [[ "$m" -eq 128 && "$h" -eq 96 ]]; then echo 26000; return; fi
  if [[ "$m" -eq 128 && "$h" -eq 64 ]]; then echo 22000; return; fi
  echo 18000
}

"$PY" - "$MANIFEST" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"generated_at_unix": time.time(), "kind": "field_ablation_burst", "runs": []}, indent=2) + "\n")
PY

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2+0)>=25000 && ($3+0)<=175000) print $1":"$2}' \
  | sort -t: -k2,2nr)

declare -A FREE
for row in "${GPU_ROWS[@]}"; do
  gpu="${row%%:*}"
  free="${row##*:}"
  FREE["$gpu"]="$free"
done

append_manifest() {
  local name="$1" gpu="$2" log="$3" report="$4" cmd="$5" need="$6" free_before="$7"
  "$PY" - "$MANIFEST" "$name" "$gpu" "$log" "$report" "$cmd" "$need" "$free_before" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["runs"].append({
    "experiment_name": sys.argv[2],
    "gpu": sys.argv[3],
    "log_path": sys.argv[4],
    "report_path": sys.argv[5],
    "command": sys.argv[6],
    "estimated_mem_need_mib": int(sys.argv[7]),
    "free_mem_before_reservation_mib": int(sys.argv[8]),
})
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

write_status() {
  local status="$1" completed="$2"
  "$PY" - "$STATUS" "$status" "$completed" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"generated_at_unix": time.time(), "status": sys.argv[2], "completed": sys.argv[3] == "true"}, indent=2) + "\n")
PY
}

pick_gpu_for() {
  local need="$1"
  local best_gpu=""
  local best_free="-1"
  for gpu in "${!FREE[@]}"; do
    local free="${FREE[$gpu]}"
    if [[ "$free" -ge "$need" && "$free" -gt "$best_free" ]]; then
      best_gpu="$gpu"
      best_free="$free"
    fi
  done
  if [[ -z "$best_gpu" ]]; then
    return 1
  fi
  FREE["$best_gpu"]=$(( best_free - need ))
  PICK_GPU="$best_gpu"
  PICK_FREE="$best_free"
  return 0
}

PIDS=()
for seed in 42 123; do
  for m in 128 512; do
    for h in 32 64 96; do
      name="v31_field_m${m}_h${h}_no_field_seed${seed}"
      report="$RUN_DIR/${name}.json"
      if [[ -s "$report" ]] && jq -e '.completed == true and .steps == 4000' "$report" >/dev/null 2>&1; then
        echo "[skip] $name"
        continue
      fi
      need="$(mem_need "$m" "$h")"
      PICK_GPU=""
      PICK_FREE="0"
      if ! pick_gpu_for "$need"; then
        echo "[defer] $name insufficient_free_mem_for_need_${need}MiB"
        continue
      fi
      gpu="$PICK_GPU"
      free_before="$PICK_FREE"
      batch="$(batch_for "$m" "$h")"
      log="$LOG_DIR/${name}.log"
      cmd="CUDA_VISIBLE_DEVICES=$gpu '$PY' code/stwm/tools/train_ostf_field_preserving_v31_20260508.py --experiment-name '$name' --horizon '$h' --m-points '$m' --seed '$seed' --steps 4000 --eval-interval 1000 --batch-size '$batch' --hidden-dim 192 --field-interaction-layers 2 --temporal-rollout-layers 2 --heads 6 --num-workers 4 --amp --disable-field-interaction"
      append_manifest "$name" "$gpu" "${log#$ROOT/}" "${report#$ROOT/}" "$cmd" "$need" "$free_before"
      echo "[launch][ablation-burst] $name gpu=$gpu free_before=${free_before} need=${need} batch=$batch"
      (
        set +e
        bash -lc "$cmd" > "$log" 2>&1
        rc=$?
        if [[ "$rc" -ne 0 ]]; then
          retry=$(( batch / 2 ))
          [[ "$retry" -lt 1 ]] && retry=1
          echo "[retry-half-batch] rc=$rc retry_batch=$retry" >> "$log"
          retry_cmd="CUDA_VISIBLE_DEVICES=$gpu '$PY' code/stwm/tools/train_ostf_field_preserving_v31_20260508.py --experiment-name '$name' --horizon '$h' --m-points '$m' --seed '$seed' --steps 4000 --eval-interval 1000 --batch-size '$retry' --hidden-dim 192 --field-interaction-layers 2 --temporal-rollout-layers 2 --heads 6 --num-workers 4 --amp --disable-field-interaction"
          bash -lc "$retry_cmd" >> "$log" 2>&1
        fi
      ) &
      PIDS+=("$!")
    done
  done
done

write_status "launched" "false"
failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done
if [[ "$failed" == "0" ]]; then
  write_status "completed" "true"
else
  write_status "failed_or_partial" "false"
fi
exit "$failed"
