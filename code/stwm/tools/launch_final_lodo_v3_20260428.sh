#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"
export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS:-16}"

cd "${REPO_ROOT}"
mkdir -p outputs/logs outputs/checkpoints/stwm_final_lodo_v3_20260428 reports docs outputs/cache outputs/run_status

START_CKPT="${START_CKPT:-outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt}"
OBS_REPORT="${OBS_REPORT:-reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json}"
FUTURE_C32="${FUTURE_C32:-reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json}"
FUTURE_C64="${FUTURE_C64:-reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json}"
STEPS="${STEPS:-5000}"
LR="${LR:-3e-5}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.25}"
GPU_UTIL_MAX="${GPU_UTIL_MAX:-75}"
GPU_MEM_USED_MAX_MB="${GPU_MEM_USED_MAX_MB:-120000}"
GPU_UTIL_MAX_FOR_DOUBLE="${GPU_UTIL_MAX_FOR_DOUBLE:-65}"
GPU_MEM_USED_MAX_MB_FOR_DOUBLE="${GPU_MEM_USED_MAX_MB_FOR_DOUBLE:-100000}"
GPU_UTIL_MAX_FOR_TRIPLE="${GPU_UTIL_MAX_FOR_TRIPLE:-25}"
GPU_MEM_USED_MAX_MB_FOR_TRIPLE="${GPU_MEM_USED_MAX_MB_FOR_TRIPLE:-60000}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
POLL_SECONDS="${POLL_SECONDS:-120}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAT_TRAIN_SHARDS="${MAT_TRAIN_SHARDS:-4}"
MAT_VAL_SHARDS="${MAT_VAL_SHARDS:-2}"
MAT_TEST_SHARDS="${MAT_TEST_SHARDS:-2}"
MAT_POLL_SECONDS="${MAT_POLL_SECONDS:-20}"

split_item_count() {
  local SPLIT_REPORT="$1"
  local EVAL_SPLIT="$2"
  "${PY}" - <<PY
import json
from pathlib import Path
d = json.loads(Path("${SPLIT_REPORT}").read_text())
print(int(d.get("${EVAL_SPLIT}_item_count", 0)))
PY
}

materialize_split_sharded() {
  local DIR_NAME="$1"
  local SPLIT_REPORT="$2"
  local EVAL_SPLIT="$3"
  local SHARDS="$4"
  local REPORT_OUT="$5"
  local CACHE_OUT="$6"
  local DOC_OUT="$7"
  local TOTAL
  TOTAL="$(split_item_count "${SPLIT_REPORT}" "${EVAL_SPLIT}")"
  if [[ -f "${REPORT_OUT}" && "${FORCE_RERUN}" != "1" ]]; then
    echo "[lodo-v3] reuse materialization report ${REPORT_OUT}" >&2
    return
  fi
  if [[ "${TOTAL}" -le 0 ]]; then
    echo "[lodo-v3] skip materialization ${DIR_NAME}/${EVAL_SPLIT} total=${TOTAL}" >&2
    return
  fi
  echo "[lodo-v3] sharded materialize dir=${DIR_NAME} split=${EVAL_SPLIT} total=${TOTAL} shards=${SHARDS}" >&2
  local shard_reports=()
  local shard_sessions=()
  local BASE_DIR
  BASE_DIR="$(dirname "${CACHE_OUT}")"
  mkdir -p "${BASE_DIR}"
  for SHARD in $(seq 0 $((SHARDS - 1))); do
    local START=$((SHARD * TOTAL / SHARDS))
    local END=$(((SHARD + 1) * TOTAL / SHARDS))
    local SHARD_TAG="${DIR_NAME}_${EVAL_SPLIT}_shard$(printf '%02d' "${SHARD}")"
    local SHARD_REPORT="reports/stwm_final_lodo_v3_${SHARD_TAG}_20260428.json"
    local SHARD_CACHE="${BASE_DIR}/${EVAL_SPLIT}_eval_batches_shard$(printf '%02d' "${SHARD}").pt"
    local SHARD_DOC="docs/STWM_FINAL_LODO_V3_${DIR_NAME}_${EVAL_SPLIT}_SHARD$(printf '%02d' "${SHARD}")_20260428.md"
    local SHARD_LOG="outputs/logs/stwm_final_lodo_v3_${SHARD_TAG}.log"
    local SHARD_SESSION="stwm_final_lodo_v3_mat_${DIR_NAME}_${EVAL_SPLIT}_s$(printf '%02d' "${SHARD}")"
    shard_reports+=("${SHARD_REPORT}")
    shard_sessions+=("${SHARD_SESSION}")
    tmux kill-session -t "${SHARD_SESSION}" 2>/dev/null || true
    tmux new-session -d -s "${SHARD_SESSION}" "bash -lc 'cd ${REPO_ROOT}; exec env STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS} NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS} STWM_TORCH_NUM_THREADS=${STWM_TORCH_NUM_THREADS} PYTHONPATH=${PYTHONPATH} ${PY} code/stwm/tools/materialize_semantic_memory_eval_set_20260428.py --split-report ${SPLIT_REPORT} --eval-split ${EVAL_SPLIT} --strict-split --allow-scan-all-stage2-splits --requested-heldout-count 1 --max-samples-per-dataset 999999 --timeout-seconds 60 --retries 2 --batch-size 4 --item-start ${START} --item-end ${END} --progress-every 50 --cache-output ${SHARD_CACHE} --output ${SHARD_REPORT} --doc ${SHARD_DOC} --audit-name stwm_final_lodo_v3_materialization_${EVAL_SPLIT}_shard --title \"STWM Final LODO V3 Materialization ${DIR_NAME} ${EVAL_SPLIT} shard\" > ${SHARD_LOG} 2>&1'"
  done
  while true; do
    local ACTIVE=0
    for SESSION in "${shard_sessions[@]}"; do
      if tmux has-session -t "${SESSION}" 2>/dev/null; then
        ACTIVE=$((ACTIVE + 1))
      fi
    done
    if [[ "${ACTIVE}" == "0" ]]; then
      break
    fi
    echo "[lodo-v3] materialize dir=${DIR_NAME} split=${EVAL_SPLIT} active_shards=${ACTIVE}" >&2
    sleep "${MAT_POLL_SECONDS}"
  done
  "${PY}" code/stwm/tools/merge_semantic_memory_eval_set_materialization_shards_20260428.py \
    --shard-reports "${shard_reports[@]}" \
    --cache-output "${CACHE_OUT}" \
    --output "${REPORT_OUT}" \
    --doc "${DOC_OUT}" \
    --audit-name "stwm_final_lodo_v3_materialization_${DIR_NAME}_${EVAL_SPLIT}" \
    --title "STWM Final LODO V3 Materialization ${DIR_NAME} ${EVAL_SPLIT}" \
    --requested-heldout-count 1
}

prepare_direction_materialization() {
  local NAME="$1"
  local SPLIT_REPORT="$2"
  local BASE="outputs/cache/stwm_final_lodo_v3_${NAME}_20260428"
  mkdir -p "${BASE}"
  materialize_split_sharded "${NAME}" "${SPLIT_REPORT}" train "${MAT_TRAIN_SHARDS}" \
    "reports/stwm_final_lodo_v3_${NAME}_materialization_train_20260428.json" \
    "${BASE}/train_eval_batches.pt" \
    "docs/STWM_FINAL_LODO_V3_${NAME}_MATERIALIZATION_TRAIN_20260428.md" &
  local pid_train=$!
  materialize_split_sharded "${NAME}" "${SPLIT_REPORT}" val "${MAT_VAL_SHARDS}" \
    "reports/stwm_final_lodo_v3_${NAME}_materialization_val_20260428.json" \
    "${BASE}/val_eval_batches.pt" \
    "docs/STWM_FINAL_LODO_V3_${NAME}_MATERIALIZATION_VAL_20260428.md" &
  local pid_val=$!
  materialize_split_sharded "${NAME}" "${SPLIT_REPORT}" test "${MAT_TEST_SHARDS}" \
    "reports/stwm_final_lodo_v3_${NAME}_materialization_test_20260428.json" \
    "${BASE}/test_eval_batches.pt" \
    "docs/STWM_FINAL_LODO_V3_${NAME}_MATERIALIZATION_TEST_20260428.md" &
  local pid_test=$!

  wait "${pid_train}"
  wait "${pid_val}"
  echo "[lodo-v3] train+val ready direction=${NAME}" >&2
  touch "outputs/run_status/stwm_final_lodo_v3_${NAME}.trainval_ready"

  wait "${pid_test}"
  echo "[lodo-v3] test ready direction=${NAME}" >&2
  touch "outputs/run_status/stwm_final_lodo_v3_${NAME}.test_ready"
}

prepare_direction_materialization "vspw_to_vipseg" "reports/stwm_mixed_fullscale_v2_lodo_vspw_to_vipseg_splits_20260428.json" &
pid_mat_a=$!
prepare_direction_materialization "vipseg_to_vspw" "reports/stwm_mixed_fullscale_v2_lodo_vipseg_to_vspw_splits_20260428.json" &
pid_mat_b=$!

pending_runs=()
for DIR_NAME in vspw_to_vipseg vipseg_to_vspw; do
  for C in 32 64; do
    for SEED in 42 123 456 789 1001; do
      SUMMARY="reports/stwm_final_lodo_v3_${DIR_NAME}_c${C}_seed${SEED}_train_20260428.json"
      CKPT="outputs/checkpoints/stwm_final_lodo_v3_20260428/${DIR_NAME}_c${C}_seed${SEED}_final.pt"
      if [[ "${FORCE_RERUN}" != "1" && -f "${SUMMARY}" && -f "${CKPT}" ]]; then
        echo "[lodo-v3] skip completed ${DIR_NAME} c${C} seed${SEED}" >&2
        continue
      fi
      pending_runs+=("${DIR_NAME}|${C}|${SEED}|${SUMMARY}|${CKPT}")
    done
  done
done

select_gpu() {
  local used_list
  used_list="$(tmux ls 2>/dev/null | awk -F: '/^stwm_final_lodo_v3_/ {print $1}' | awk '!/_mat_/ && !/_launcher$/' | while read -r sess; do tmux show-environment -t "$sess" CUDA_VISIBLE_DEVICES 2>/dev/null | sed 's/^CUDA_VISIBLE_DEVICES=//'; done | sort | uniq -c || true)"
  local stats
  stats="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
  while IFS=, read -r gpu mem util; do
    gpu="$(echo "$gpu" | xargs)"
    mem="$(echo "$mem" | xargs)"
    util="$(echo "$util" | xargs)"
    [[ -z "${gpu}" ]] && continue
    local active_count
    active_count="$(echo "${used_list}" | awk -v g="${gpu}" '$2==g {print $1}' | tail -n1)"
    active_count="${active_count:-0}"
    local gpu_capacity=0
    if (( util <= GPU_UTIL_MAX_FOR_TRIPLE )) && (( mem <= GPU_MEM_USED_MAX_MB_FOR_TRIPLE )); then
      gpu_capacity=3
    elif (( util <= GPU_UTIL_MAX_FOR_DOUBLE )) && (( mem <= GPU_MEM_USED_MAX_MB_FOR_DOUBLE )); then
      gpu_capacity=2
    elif (( util <= GPU_UTIL_MAX )) && (( mem <= GPU_MEM_USED_MAX_MB )); then
      gpu_capacity=1
    fi
    if (( gpu_capacity > 0 )) && (( active_count < gpu_capacity )); then
      echo "${gpu}"
      return 0
    fi
  done <<< "${stats}"
  return 1
}

launch_one() {
  local DIR_NAME="$1"
  local C="$2"
  local SEED="$3"
  local SUMMARY="$4"
  local CKPT="$5"
  local GPU_ID="$6"
  local FUTURE_REPORT="${FUTURE_C32}"
  [[ "${C}" == "64" ]] && FUTURE_REPORT="${FUTURE_C64}"
  local BASE="outputs/cache/stwm_final_lodo_v3_${DIR_NAME}_20260428"
  local TRAIN_REPORT="reports/stwm_final_lodo_v3_${DIR_NAME}_materialization_train_20260428.json"
  local VAL_REPORT="reports/stwm_final_lodo_v3_${DIR_NAME}_materialization_val_20260428.json"
  local SESSION="stwm_final_lodo_v3_${DIR_NAME}_c${C}_s${SEED}"
  local LOG="outputs/logs/${SESSION}.log"
  local DOC="docs/STWM_FINAL_LODO_V3_${DIR_NAME}_C${C}_SEED${SEED}_TRAIN_20260428.md"
  local STATUS="outputs/run_status/${SESSION}.status.json"
  tmux kill-session -t "${SESSION}" 2>/dev/null || true
  cat > "${LOG}.run.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export STWM_PROC_TITLE="python"
export STWM_PROC_TITLE_MODE="generic"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}"
export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS}"
export PYTHONPATH="${PYTHONPATH}"
cat > "${STATUS}" <<JSON
{"status":"running","direction":"${DIR_NAME}","prototype_count":${C},"seed":${SEED},"gpu_id":"${GPU_ID}"}
JSON
"${PY}" code/stwm/tools/train_fullscale_semantic_trace_world_model_single_20260428.py \
  --prototype-count "${C}" \
  --seed "${SEED}" \
  --train-cache-report "${TRAIN_REPORT}" \
  --val-cache-report "${VAL_REPORT}" \
  --start-checkpoint "${START_CKPT}" \
  --observed-report "${OBS_REPORT}" \
  --future-cache-report "${FUTURE_REPORT}" \
  --steps "${STEPS}" \
  --lr "${LR}" \
  --residual-scale "${RESIDUAL_SCALE}" \
  --device cuda \
  --checkpoint-output "${CKPT}" \
  --summary-output "${SUMMARY}" \
  --doc "${DOC}" \
  > "${LOG}" 2>&1
python - <<PY2
import json
from pathlib import Path
Path("${STATUS}").write_text(json.dumps({"status":"completed","direction":"${DIR_NAME}","prototype_count":${C},"seed":${SEED},"gpu_id":"${GPU_ID}","summary":"${SUMMARY}","checkpoint":"${CKPT}"}, indent=2)+"\\n")
PY2
EOF
  chmod +x "${LOG}.run.sh"
  tmux new-session -d -s "${SESSION}" "bash -lc '${LOG}.run.sh'"
  tmux set-environment -t "${SESSION}" CUDA_VISIBLE_DEVICES "${GPU_ID}"
  echo "[lodo-v3] launched ${SESSION} gpu=${GPU_ID}" >&2
}

ready_marker_for_direction() {
  local DIR_NAME="$1"
  echo "outputs/run_status/stwm_final_lodo_v3_${DIR_NAME}.trainval_ready"
}

find_ready_run_index() {
  local idx=0
  for run in "${pending_runs[@]}"; do
    IFS='|' read -r DIR_NAME _C _SEED _SUMMARY _CKPT <<< "${run}"
    local marker
    marker="$(ready_marker_for_direction "${DIR_NAME}")"
    if [[ -f "${marker}" ]]; then
      echo "${idx}"
      return 0
    fi
    idx=$((idx + 1))
  done
  return 1
}

while (( ${#pending_runs[@]} > 0 )); do
  active="$(tmux ls 2>/dev/null | awk -F: '/^stwm_final_lodo_v3_/ {print $1}' | awk '!/_mat_/ && !/_launcher$/' | awk 'END {print NR+0}')"
  if (( active < MAX_PARALLEL )); then
    if ready_idx="$(find_ready_run_index)" && gpu_id="$(select_gpu)"; then
      run="${pending_runs[${ready_idx}]}"
      unset 'pending_runs[ready_idx]'
      pending_runs=("${pending_runs[@]}")
      IFS='|' read -r DIR_NAME C SEED SUMMARY CKPT <<< "${run}"
      launch_one "${DIR_NAME}" "${C}" "${SEED}" "${SUMMARY}" "${CKPT}" "${gpu_id}"
      sleep 5
      continue
    fi
  fi
  echo "[lodo-v3] pending=${#pending_runs[@]} active=${active} waiting_for_gpu" >&2
  sleep "${POLL_SECONDS}"
done

wait "${pid_mat_a}"
wait "${pid_mat_b}"
echo "[lodo-v3] all pending LODO runs launched" >&2
