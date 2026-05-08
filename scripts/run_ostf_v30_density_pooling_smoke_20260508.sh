#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
cd "$ROOT"

LOG_DIR="$ROOT/logs/stwm_ostf_v30_density_pooling_smoke_runs_20260508"
mkdir -p "$LOG_DIR" "$ROOT/reports"

gpu_list() {
  nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); gsub(/ /,"",$3); if (($2+0)>=40000 && ($3+0)<=170000) print "0 "$2" "$1; else if (($2+0)>=25000 && ($3+0)<=170000) print "1 "$2" "$1}' \
    | sort -k1,1n -k2,2nr | awk '{print $3}'
}

mapfile -t GPUS < <(gpu_list || true)
if [[ "${#GPUS[@]}" -eq 0 ]]; then GPUS=(""); fi

run_one() {
  local gpu="$1" name="$2" m="$3" h="$4" mode="$5"
  local log="$LOG_DIR/${name}.log"
  local batch="24"
  if [[ "$m" -ge 1024 ]]; then batch="12"; fi
  local common=(code/stwm/tools/train_ostf_external_gt_v30_20260508.py
    --experiment-name "$name" --horizon "$h" --m-points "$m" --seed 42
    --steps "${V30_DENSITY_SMOKE_STEPS:-800}" --batch-size "$batch" --grad-accum-steps 1
    --eval-interval 400 --amp --density-aware-pooling "$mode"
    --density-inducing-tokens 16 --density-motion-topk 128 --num-workers 2)
  echo "[smoke] launch $name gpu=${gpu:-cpu} batch=$batch mode=$mode"
  if [[ -n "$gpu" ]]; then
    (CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${common[@]}" > "$log" 2>&1 || {
      echo "[smoke] retry $name with half batch" >> "$log"
      local retry_batch=$(( batch / 2 ))
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${common[@]}" --batch-size "$retry_batch" >> "$log" 2>&1
    }) &
  else
    ("$PY" "${common[@]}" --cpu > "$log" 2>&1) &
  fi
  PIDS+=("$!")
}

PIDS=()
i=0
for m in 512 1024; do
  for h in 32 64 96; do
    for mode in moments induced_attention hybrid_moments_attention; do
      run_one "${GPUS[$((i % ${#GPUS[@]}))]}" "v30_extgt_density_smoke_m${m}_h${h}_${mode}_seed42" "$m" "$h" "$mode"
      i=$((i+1))
    done
  done
done

failed=0
for pid in "${PIDS[@]}"; do wait "$pid" || failed=1; done
"$PY" code/stwm/tools/aggregate_ostf_v30_density_pooling_smoke_20260508.py
cat > "$ROOT/reports/stwm_ostf_v30_density_pooling_smoke_status_20260508.json" <<JSON
{"session":"stwm_ostf_v30_density_pooling_smoke_20260508","status":"$([[ "$failed" == "0" ]] && echo completed || echo failed)","updated_at_local":"$(date -Iseconds)","run_logs_dir":"${LOG_DIR#$ROOT/}","summary_path":"reports/stwm_ostf_v30_density_pooling_smoke_summary_20260508.json"}
JSON
exit "$failed"
