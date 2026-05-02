#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE=python
export STWM_PROC_TITLE_MODE=generic
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export STWM_TORCH_NUM_THREADS="${STWM_TORCH_NUM_THREADS:-4}"
cd "${REPO_ROOT}"
mkdir -p reports docs outputs/cache outputs/logs/stwm_fstf_scaling_v12_20260502

gpu_for_slot() {
  local slot="$1"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '{free=$3-$2; score=free-$4*256; print $1,score}' |
    sort -k2,2nr | awk -v s="${slot}" 'NR==((s%8)+1){print $1}'
}

wait_prefix() {
  local prefix="$1"
  while [[ "$(tmux list-sessions 2>/dev/null | grep -c "^${prefix}")" -ge "${MAX_PARALLEL}" ]]; do sleep 20; done
}

build_horizon() {
  local H="$1"
  local FEATURE_REPORT="reports/stwm_fstf_horizon_h${H}_feature_targets_v12_20260502.json"
  local TARGET_REPORT="reports/stwm_fstf_horizon_h${H}_prototype_targets_c32_v12_20260502.json"
  local SHARD_REPORTS=()
  local slot=0
  if [[ ! -s "${FEATURE_REPORT}" ]]; then
    for SPLIT in train val; do
      local TOTAL=5612
      local SHARDS=8
      if [[ "${SPLIT}" == "val" ]]; then TOTAL=686; SHARDS=2; fi
      for SHARD in $(seq 0 $((SHARDS - 1))); do
        local START=$((SHARD * TOTAL / SHARDS))
        local END=$(((SHARD + 1) * TOTAL / SHARDS))
        local NAME="h${H}_${SPLIT}_shard$(printf '%02d' "${SHARD}")"
        local REPORT="reports/stwm_fstf_horizon_${NAME}_feature_targets_v12_20260502.json"
        local CACHE_DIR="outputs/cache/stwm_fstf_horizon_${NAME}_feature_targets_v12_20260502"
        local LOG="outputs/logs/stwm_fstf_scaling_v12_20260502/${NAME}_feature.log"
        local SESSION="stwm_fstf_v12_${NAME}_feature"
        SHARD_REPORTS+=("${REPORT}")
        if [[ -s "${REPORT}" ]]; then continue; fi
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
          continue
        fi
        wait_prefix "stwm_fstf_v12_h"
        local GPU_ID
        GPU_ID="$(gpu_for_slot "${slot}")"
        slot=$((slot+1))
        tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; echo START \$(date -Is) session=${SESSION} cuda=${GPU_ID}; export CUDA_VISIBLE_DEVICES=${GPU_ID} STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 PYTHONPATH=${PYTHONPATH}; ${PY} code/stwm/tools/build_future_semantic_trace_feature_targets_20260428.py --dataset-names vspw vipseg --splits ${SPLIT} --max-samples-train 999999 --max-samples-val 999999 --max-entities-per-sample 8 --fut-len ${H} --device cuda --batch-size 512 --crop-extraction-mode tensor_roi --target-build-mode fast_target_only --entry-start ${START} --entry-end ${END} --progress-every 200 --torch-num-threads ${STWM_TORCH_NUM_THREADS} --image-cache-size 256 --cache-dir ${CACHE_DIR} --output ${REPORT} --doc docs/STWM_FSTF_HORIZON_${NAME}_FEATURE_TARGETS_V12_20260502.md; echo END \$(date -Is)' > ${LOG} 2>&1"
      done
    done
    while tmux list-sessions 2>/dev/null | grep -q '^stwm_fstf_v12_h'; do sleep 30; done
    "${PY}" code/stwm/tools/merge_future_semantic_trace_feature_target_shards_20260428.py \
      --shard-reports "${SHARD_REPORTS[@]}" \
      --cache-dir "outputs/cache/stwm_fstf_horizon_h${H}_feature_targets_v12_20260502" \
      --output "${FEATURE_REPORT}" \
      --doc "docs/STWM_FSTF_HORIZON_H${H}_FEATURE_TARGETS_V12_20260502.md"
  fi
  if [[ ! -s "${TARGET_REPORT}" ]]; then
    "${PY}" code/stwm/tools/build_future_semantic_trace_prototype_targets_20260428.py \
      --feature-cache-report "${FEATURE_REPORT}" \
      --prototype-report reports/stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428.json \
      --cache-dir "outputs/cache/stwm_fstf_horizon_h${H}_prototype_targets_c32_v12_20260502" \
      --output "${TARGET_REPORT}" \
      --doc "docs/STWM_FSTF_HORIZON_H${H}_PROTOTYPE_TARGETS_C32_V12_20260502.md"
  fi
  for SPLIT in train val test; do
    local BATCH_REPORT="reports/stwm_fstf_horizon_h${H}_batch_${SPLIT}_v12_20260502.json"
    if [[ -s "${BATCH_REPORT}" ]]; then continue; fi
    "${PY}" code/stwm/tools/materialize_stwm_fstf_batch_cache_v12_20260502.py \
      --eval-split "${SPLIT}" \
      --fut-len "${H}" \
      --max-entities-per-sample 8 \
      --allow-scan-all-stage2-splits \
      --cache-output "outputs/cache/stwm_fstf_horizon_h${H}_${SPLIT}_v12_20260502/eval_batches.pt" \
      --output "${BATCH_REPORT}" \
      --doc "docs/STWM_FSTF_HORIZON_H${H}_BATCH_${SPLIT}_V12_20260502.md" \
      --audit-name "stwm_fstf_horizon_h${H}_batch_${SPLIT}_v12"
  done
  "${PY}" code/stwm/tools/summarize_stwm_fstf_scaling_cache_v12_20260502.py \
    --axis horizon --value "h${H}" \
    --future-report "${TARGET_REPORT}" \
    --observed-report reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json \
    --train-report "reports/stwm_fstf_horizon_h${H}_batch_train_v12_20260502.json" \
    --val-report "reports/stwm_fstf_horizon_h${H}_batch_val_v12_20260502.json" \
    --test-report "reports/stwm_fstf_horizon_h${H}_batch_test_v12_20260502.json" \
    --output "reports/stwm_fstf_horizon_cache_h${H}_v12_20260502.json" \
    --doc "docs/STWM_FSTF_HORIZON_CACHE_H${H}_V12_20260502.md"
}

build_horizon 16
build_horizon 24
