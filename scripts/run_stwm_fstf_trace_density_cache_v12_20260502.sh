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

report_success() {
  local report="$1"
  [[ -s "${report}" ]] || return 1
  "${PY}" - "${report}" <<'PY'
import json
import sys
from pathlib import Path

try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
sys.exit(0 if payload.get("materialization_success") is True else 1)
PY
}

gpu_for_slot() {
  local slot="$1"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
    awk -F', ' '{free=$3-$2; score=free-$4*256; print $1,score}' |
    sort -k2,2nr | awk -v s="${slot}" 'NR==((s%8)+1){print $1}'
}

wait_prefix() {
  while [[ "$(tmux list-sessions 2>/dev/null | grep -c '^stwm_fstf_v12_k')" -ge "${MAX_PARALLEL}" ]]; do sleep 20; done
}

build_k() {
  local K="$1"
  local FEATURE_REPORT="reports/stwm_fstf_trace_density_k${K}_feature_targets_v12_20260502.json"
  local TARGET_REPORT="reports/stwm_fstf_trace_density_k${K}_prototype_targets_c32_v12_20260502.json"
  local OBS_REPORT="reports/stwm_fstf_trace_density_k${K}_observed_targets_v12_20260502.json"
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
        local NAME="k${K}_${SPLIT}_shard$(printf '%02d' "${SHARD}")"
        local REPORT="reports/stwm_fstf_trace_density_${NAME}_feature_targets_v12_20260502.json"
        local CACHE_DIR="outputs/cache/stwm_fstf_trace_density_${NAME}_feature_targets_v12_20260502"
        local LOG="outputs/logs/stwm_fstf_scaling_v12_20260502/${NAME}_feature.log"
        local SESSION="stwm_fstf_v12_${NAME}_feature"
        SHARD_REPORTS+=("${REPORT}")
        if [[ -s "${REPORT}" ]]; then continue; fi
        if tmux has-session -t "${SESSION}" 2>/dev/null; then
          continue
        fi
        wait_prefix
        local GPU_ID
        GPU_ID="$(gpu_for_slot "${slot}")"
        slot=$((slot+1))
        tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; echo START \$(date -Is) session=${SESSION} cuda=${GPU_ID}; export CUDA_VISIBLE_DEVICES=${GPU_ID} STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 PYTHONPATH=${PYTHONPATH}; ${PY} code/stwm/tools/build_future_semantic_trace_feature_targets_20260428.py --dataset-names vspw vipseg --splits ${SPLIT} --max-samples-train 999999 --max-samples-val 999999 --max-entities-per-sample ${K} --fut-len 8 --device cuda --batch-size 512 --crop-extraction-mode tensor_roi --target-build-mode fast_target_only --entry-start ${START} --entry-end ${END} --progress-every 200 --torch-num-threads ${STWM_TORCH_NUM_THREADS} --image-cache-size 256 --cache-dir ${CACHE_DIR} --output ${REPORT} --doc docs/STWM_FSTF_TRACE_DENSITY_${NAME}_FEATURE_TARGETS_V12_20260502.md; echo END \$(date -Is)' > ${LOG} 2>&1"
      done
    done
    while tmux list-sessions 2>/dev/null | grep -q '^stwm_fstf_v12_k'; do sleep 30; done
    "${PY}" code/stwm/tools/merge_future_semantic_trace_feature_target_shards_20260428.py \
      --shard-reports "${SHARD_REPORTS[@]}" \
      --cache-dir "outputs/cache/stwm_fstf_trace_density_k${K}_feature_targets_v12_20260502" \
      --output "${FEATURE_REPORT}" \
      --doc "docs/STWM_FSTF_TRACE_DENSITY_K${K}_FEATURE_TARGETS_V12_20260502.md"
  fi
  if [[ ! -s "${TARGET_REPORT}" ]]; then
    "${PY}" code/stwm/tools/build_future_semantic_trace_prototype_targets_20260428.py \
      --feature-cache-report "${FEATURE_REPORT}" \
      --prototype-report reports/stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428.json \
      --cache-dir "outputs/cache/stwm_fstf_trace_density_k${K}_prototype_targets_c32_v12_20260502" \
      --output "${TARGET_REPORT}" \
      --doc "docs/STWM_FSTF_TRACE_DENSITY_K${K}_PROTOTYPE_TARGETS_C32_V12_20260502.md"
  fi
  if [[ ! -s "${OBS_REPORT}" ]]; then
    "${PY}" code/stwm/tools/build_observed_semantic_prototype_targets_20260428.py \
      --feature-report "${FEATURE_REPORT}" \
      --prototype-target-reports "${TARGET_REPORT}" \
      --max-samples-per-dataset 999999 \
      --observed-max-samples-per-dataset 999999 \
      --observed-cache-mode predecode_then_raw_fallback \
      --observed-min-coverage 0.05 \
      --device cuda \
      --batch-size 512 \
      --cache-dir "outputs/cache/stwm_fstf_trace_density_k${K}_observed_targets_v12_20260502" \
      --output "${OBS_REPORT}" \
      --doc "docs/STWM_FSTF_TRACE_DENSITY_K${K}_OBSERVED_TARGETS_V12_20260502.md"
  fi
  for SPLIT in train val test; do
    local BATCH_REPORT="reports/stwm_fstf_trace_density_k${K}_batch_${SPLIT}_v12_20260502.json"
    if report_success "${BATCH_REPORT}"; then continue; fi
    "${PY}" code/stwm/tools/materialize_stwm_fstf_batch_cache_v12_20260502.py \
      --eval-split "${SPLIT}" \
      --fut-len 8 \
      --max-entities-per-sample "${K}" \
      --allow-scan-all-stage2-splits \
      --cache-output "outputs/cache/stwm_fstf_trace_density_k${K}_${SPLIT}_v12_20260502/eval_batches.pt" \
      --output "${BATCH_REPORT}" \
      --doc "docs/STWM_FSTF_TRACE_DENSITY_K${K}_BATCH_${SPLIT}_V12_20260502.md" \
      --audit-name "stwm_fstf_trace_density_k${K}_batch_${SPLIT}_v12"
  done
  "${PY}" code/stwm/tools/summarize_stwm_fstf_scaling_cache_v12_20260502.py \
    --axis density --value "k${K}" \
    --future-report "${TARGET_REPORT}" \
    --observed-report "${OBS_REPORT}" \
    --train-report "reports/stwm_fstf_trace_density_k${K}_batch_train_v12_20260502.json" \
    --val-report "reports/stwm_fstf_trace_density_k${K}_batch_val_v12_20260502.json" \
    --test-report "reports/stwm_fstf_trace_density_k${K}_batch_test_v12_20260502.json" \
    --output "reports/stwm_fstf_trace_density_cache_k${K}_v12_20260502.json" \
    --doc "docs/STWM_FSTF_TRACE_DENSITY_CACHE_K${K}_V12_20260502.md"
}

build_k 16
build_k 32
