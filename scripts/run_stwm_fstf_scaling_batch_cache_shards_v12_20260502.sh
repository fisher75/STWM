#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
SPLIT_REPORT="${SPLIT_REPORT:-reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json}"
MAX_PARALLEL_CACHE="${MAX_PARALLEL_CACHE:-24}"
TRAIN_SHARDS="${TRAIN_SHARDS:-16}"
EVAL_SHARDS="${EVAL_SHARDS:-4}"

export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE=python
export STWM_PROC_TITLE_MODE=generic
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"

cd "${REPO_ROOT}"
mkdir -p reports docs outputs/cache outputs/logs/stwm_fstf_scaling_v12_20260502

split_count() {
  local split="$1"
  "${PY}" - "${SPLIT_REPORT}" "${split}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(len(payload["splits"].get(sys.argv[2], [])))
PY
}

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

wait_cache_slot() {
  while [[ "$(tmux list-sessions 2>/dev/null | grep -c '^stwm_fstf_v12_batch_')" -ge "${MAX_PARALLEL_CACHE}" ]]; do
    sleep 10
  done
}

launch_shard() {
  local prefix="$1" split="$2" fut_len="$3" k="$4" shard="$5" shards="$6" total="$7"
  local start=$((shard * total / shards))
  local end=$(((shard + 1) * total / shards))
  local shard_tag
  shard_tag="$(printf '%02d' "${shard}")"
  local report="reports/stwm_fstf_${prefix}_batch_${split}_shard${shard_tag}_v12_20260502.json"
  local cache="outputs/cache/stwm_fstf_${prefix}_${split}_shard${shard_tag}_v12_20260502/eval_batches.pt"
  local doc="docs/STWM_FSTF_${prefix^^}_BATCH_${split}_SHARD${shard_tag}_V12_20260502.md"
  local session="stwm_fstf_v12_batch_${prefix}_${split}_${shard_tag}"
  local log="outputs/logs/stwm_fstf_scaling_v12_20260502/${prefix}_${split}_shard${shard_tag}_batch.log"
  if report_success "${report}"; then
    return 0
  fi
  tmux kill-session -t "${session}" 2>/dev/null || true
  wait_cache_slot
  tmux new-session -d -s "${session}" "bash -lc 'cd ${REPO_ROOT}; echo START \$(date -Is) session=${session} item_start=${start} item_end=${end}; export STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic PYTHONUNBUFFERED=1 PYTHONPATH=${PYTHONPATH}; ${PY} code/stwm/tools/materialize_stwm_fstf_batch_cache_v12_20260502.py --split-report ${SPLIT_REPORT} --eval-split ${split} --fut-len ${fut_len} --max-entities-per-sample ${k} --allow-scan-all-stage2-splits --item-start ${start} --item-end ${end} --cache-output ${cache} --output ${report} --doc ${doc} --audit-name stwm_fstf_${prefix}_batch_${split}_shard${shard_tag}_v12; echo END \$(date -Is)' > ${log} 2>&1"
}

merge_split() {
  local prefix="$1" split="$2" fut_len="$3" k="$4" shards="$5"
  local reports=()
  for shard in $(seq 0 $((shards - 1))); do
    reports+=("reports/stwm_fstf_${prefix}_batch_${split}_shard$(printf '%02d' "${shard}")_v12_20260502.json")
  done
  "${PY}" code/stwm/tools/merge_stwm_fstf_batch_cache_shards_v12_20260502.py \
    --shard-reports "${reports[@]}" \
    --cache-output "outputs/cache/stwm_fstf_${prefix}_${split}_v12_20260502/eval_batches.pt" \
    --output "reports/stwm_fstf_${prefix}_batch_${split}_v12_20260502.json" \
    --doc "docs/STWM_FSTF_${prefix^^}_BATCH_${split}_V12_20260502.md" \
    --audit-name "stwm_fstf_${prefix}_batch_${split}_v12"
}

materialize_prefix() {
  local prefix="$1" fut_len="$2" k="$3" axis="$4" future_report="$5" observed_report="$6" summary_report="$7" summary_doc="$8"
  for split in train val test; do
    local total shards
    total="$(split_count "${split}")"
    if [[ "${split}" == "train" ]]; then shards="${TRAIN_SHARDS}"; else shards="${EVAL_SHARDS}"; fi
    for shard in $(seq 0 $((shards - 1))); do
      launch_shard "${prefix}" "${split}" "${fut_len}" "${k}" "${shard}" "${shards}" "${total}"
    done
  done
  while tmux list-sessions 2>/dev/null | grep -q '^stwm_fstf_v12_batch_'; do
    sleep 15
  done
  for split in train val test; do
    local shards
    if [[ "${split}" == "train" ]]; then shards="${TRAIN_SHARDS}"; else shards="${EVAL_SHARDS}"; fi
    merge_split "${prefix}" "${split}" "${fut_len}" "${k}" "${shards}"
  done
  "${PY}" code/stwm/tools/summarize_stwm_fstf_scaling_cache_v12_20260502.py \
    --axis "${axis}" \
    --value "${prefix}" \
    --future-report "${future_report}" \
    --observed-report "${observed_report}" \
    --train-report "reports/stwm_fstf_${prefix}_batch_train_v12_20260502.json" \
    --val-report "reports/stwm_fstf_${prefix}_batch_val_v12_20260502.json" \
    --test-report "reports/stwm_fstf_${prefix}_batch_test_v12_20260502.json" \
    --output "${summary_report}" \
    --doc "${summary_doc}"
}

materialize_prefix "horizon_h16" 16 8 "horizon" \
  "reports/stwm_fstf_horizon_h16_prototype_targets_c32_v12_20260502.json" \
  "reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json" \
  "reports/stwm_fstf_horizon_cache_h16_v12_20260502.json" \
  "docs/STWM_FSTF_HORIZON_CACHE_H16_V12_20260502.md"

materialize_prefix "horizon_h24" 24 8 "horizon" \
  "reports/stwm_fstf_horizon_h24_prototype_targets_c32_v12_20260502.json" \
  "reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json" \
  "reports/stwm_fstf_horizon_cache_h24_v12_20260502.json" \
  "docs/STWM_FSTF_HORIZON_CACHE_H24_V12_20260502.md"

materialize_prefix "trace_density_k16" 8 16 "density" \
  "reports/stwm_fstf_trace_density_k16_prototype_targets_c32_v12_20260502.json" \
  "reports/stwm_fstf_trace_density_k16_observed_targets_v12_20260502.json" \
  "reports/stwm_fstf_trace_density_cache_k16_v12_20260502.json" \
  "docs/STWM_FSTF_TRACE_DENSITY_CACHE_K16_V12_20260502.md"

materialize_prefix "trace_density_k32" 8 32 "density" \
  "reports/stwm_fstf_trace_density_k32_prototype_targets_c32_v12_20260502.json" \
  "reports/stwm_fstf_trace_density_k32_observed_targets_v12_20260502.json" \
  "reports/stwm_fstf_trace_density_cache_k32_v12_20260502.json" \
  "docs/STWM_FSTF_TRACE_DENSITY_CACHE_K32_V12_20260502.md"
