#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT"

SESSION="stwm_ostf_v31_field_multiseed_20260508"
LOG_DIR="$ROOT/logs/stwm_ostf_v31_field_multiseed_20260508"
RUN_DIR="$ROOT/reports/stwm_ostf_v31_field_preserving_runs"
PRIMARY_MANIFEST="$ROOT/reports/stwm_ostf_v31_field_multiseed_launch_manifest_20260508.json"
PRIMARY_STATUS="$ROOT/reports/stwm_ostf_v31_field_multiseed_status_20260508.json"
ABL_MANIFEST="$ROOT/reports/stwm_ostf_v31_field_ablation_launch_manifest_20260508.json"
ABL_STATUS="$ROOT/reports/stwm_ostf_v31_field_ablation_status_20260508.json"
mkdir -p "$LOG_DIR" "$RUN_DIR" "$ROOT/reports"

gpu_list() {
  nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2+0)>=60000 && ($3+0)<=150000) print "0 "$2" "$1; else if (($2+0)>=40000 && ($3+0)<=150000) print "1 "$2" "$1}' \
    | sort -k1,1n -k2,2nr | awk '{print $3}'
}
mapfile -t GPUS < <(gpu_list || true)
if [[ "${#GPUS[@]}" -eq 0 ]]; then GPUS=("0"); fi
GPU_INDEX=0
MAX_CONCURRENT="${V31_MAX_CONCURRENT:-12}"

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

init_manifest() {
  local path="$1" kind="$2"
  "$PY" - "$path" "$kind" "$SESSION" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"generated_at_unix": time.time(), "kind": sys.argv[2], "session": sys.argv[3], "runs": []}, indent=2) + "\n")
PY
}

append_manifest() {
  local path="$1" name="$2" gpu="$3" log="$4" report="$5" cmd="$6"
  "$PY" - "$path" "$name" "$gpu" "$log" "$report" "$cmd" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["runs"].append({
    "experiment_name": sys.argv[2],
    "gpu": sys.argv[3],
    "log_path": sys.argv[4],
    "report_path": sys.argv[5],
    "command": sys.argv[6],
})
path.write_text(json.dumps(payload, indent=2) + "\n")
PY
}

write_status() {
  local path="$1" status="$2" completed="$3"
  "$PY" - "$path" "$status" "$completed" <<'PY'
import json, pathlib, sys, time
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({"generated_at_unix": time.time(), "status": sys.argv[2], "completed": sys.argv[3] == "true"}, indent=2) + "\n")
PY
}

wait_slot() {
  while [[ "$(jobs -rp | wc -l)" -ge "$MAX_CONCURRENT" ]]; do
    sleep 10
  done
}

run_one() {
  local kind="$1" name="$2" m="$3" h="$4" seed="$5" extra="$6"
  local report="$RUN_DIR/${name}.json"
  local log="$LOG_DIR/${name}.log"
  if [[ -s "$report" ]] && jq -e '.completed == true and .steps == 4000' "$report" >/dev/null 2>&1; then
    echo "[skip] $name"
    return 0
  fi
  wait_slot
  local gpu="${GPUS[$((GPU_INDEX % ${#GPUS[@]}))]}"
  GPU_INDEX=$((GPU_INDEX + 1))
  local batch
  batch="$(batch_for "$m" "$h")"
  local retry_batch=$(( batch / 2 ))
  if [[ "$retry_batch" -lt 1 ]]; then retry_batch=1; fi
  local cmd="CUDA_VISIBLE_DEVICES=$gpu '$PY' code/stwm/tools/train_ostf_field_preserving_v31_20260508.py --experiment-name '$name' --horizon '$h' --m-points '$m' --seed '$seed' --steps 4000 --eval-interval 1000 --batch-size '$batch' --hidden-dim 192 --field-interaction-layers 2 --temporal-rollout-layers 2 --heads 6 --num-workers 4 --amp $extra"
  if [[ "$kind" == "primary" ]]; then
    append_manifest "$PRIMARY_MANIFEST" "$name" "$gpu" "${log#$ROOT/}" "${report#$ROOT/}" "$cmd"
  else
    append_manifest "$ABL_MANIFEST" "$name" "$gpu" "${log#$ROOT/}" "${report#$ROOT/}" "$cmd"
  fi
  echo "[launch][$kind] $name gpu=$gpu batch=$batch"
  (
    set +e
    bash -lc "$cmd" > "$log" 2>&1
    rc=$?
    if [[ "$rc" -ne 0 ]]; then
      echo "[retry-half-batch] $name rc=$rc retry_batch=$retry_batch" >> "$log"
      local retry_cmd="CUDA_VISIBLE_DEVICES=$gpu '$PY' code/stwm/tools/train_ostf_field_preserving_v31_20260508.py --experiment-name '$name' --horizon '$h' --m-points '$m' --seed '$seed' --steps 4000 --eval-interval 1000 --batch-size '$retry_batch' --hidden-dim 192 --field-interaction-layers 2 --temporal-rollout-layers 2 --heads 6 --num-workers 4 --amp $extra"
      bash -lc "$retry_cmd" >> "$log" 2>&1
    fi
  ) &
}

init_manifest "$PRIMARY_MANIFEST" "primary_multiseed"
init_manifest "$ABL_MANIFEST" "field_interaction_ablation"
write_status "$PRIMARY_STATUS" "launched" "false"
write_status "$ABL_STATUS" "launched" "false"

for seed in 42 123 456 789 2026; do
  for m in 128 512; do
    for h in 32 64 96; do
      run_one primary "v31_field_m${m}_h${h}_seed${seed}" "$m" "$h" "$seed" ""
    done
  done
done

for seed in 42 123; do
  for m in 128 512; do
    for h in 32 64 96; do
      run_one ablation "v31_field_m${m}_h${h}_no_field_seed${seed}" "$m" "$h" "$seed" "--disable-field-interaction"
    done
  done
done

failed=0
for pid in $(jobs -rp); do
  wait "$pid" || failed=1
done

"$PY" code/stwm/tools/aggregate_ostf_v31_field_multiseed_20260508.py
if [[ "$failed" == "0" ]]; then
  write_status "$PRIMARY_STATUS" "completed" "true"
  write_status "$ABL_STATUS" "completed" "true"
else
  write_status "$PRIMARY_STATUS" "failed_or_partial" "false"
  write_status "$ABL_STATUS" "failed_or_partial" "false"
fi
exit "$failed"
