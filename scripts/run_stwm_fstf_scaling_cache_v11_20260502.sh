#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

mkdir -p reports docs outputs/cache outputs/logs/stwm_fstf_scaling_v11_20260502

FEATURE_REPORT="reports/stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json"
OBS_CACHE_DIR="outputs/cache/stwm_mixed_observed_semantic_prototype_targets_v2_20260428"

build_c() {
  local C="$1"
  local PROTO_REPORT="reports/stwm_fstf_semantic_trace_prototypes_c${C}_v11_20260502.json"
  local TARGET_REPORT="reports/stwm_fstf_future_semantic_trace_prototype_targets_c${C}_v11_20260502.json"
  if [[ ! -s "${PROTO_REPORT}" ]]; then
    "${PY}" code/stwm/tools/build_semantic_trace_prototypes_20260428.py \
      --feature-cache-report "${FEATURE_REPORT}" \
      --prototype-count "${C}" \
      --iterations 20 \
      --cache-dir "outputs/cache/stwm_fstf_semantic_trace_prototypes_c${C}_v11_20260502" \
      --output "${PROTO_REPORT}" \
      --doc "docs/STWM_FSTF_SEMANTIC_TRACE_PROTOTYPES_C${C}_V11_20260502.md"
  fi
  if [[ ! -s "${TARGET_REPORT}" ]]; then
    "${PY}" code/stwm/tools/build_future_semantic_trace_prototype_targets_20260428.py \
      --feature-cache-report "${FEATURE_REPORT}" \
      --prototype-report "${PROTO_REPORT}" \
      --cache-dir "outputs/cache/stwm_fstf_future_semantic_trace_prototype_targets_c${C}_v11_20260502" \
      --output "${TARGET_REPORT}" \
      --doc "docs/STWM_FSTF_FUTURE_SEMANTIC_TRACE_PROTOTYPE_TARGETS_C${C}_V11_20260502.md"
  fi
}

build_c 16
build_c 128

"${PY}" code/stwm/tools/build_observed_semantic_prototype_targets_20260428.py \
  --feature-report "${FEATURE_REPORT}" \
  --prototype-target-reports \
    reports/stwm_fstf_future_semantic_trace_prototype_targets_c16_v11_20260502.json \
    reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json \
    reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json \
    reports/stwm_fstf_future_semantic_trace_prototype_targets_c128_v11_20260502.json \
  --max-samples-per-dataset 999999 \
  --observed-max-samples-per-dataset 999999 \
  --observed-cache-mode predecode_then_raw_fallback \
  --observed-min-coverage 0.05 \
  --device cuda \
  --batch-size 512 \
  --cache-dir "${OBS_CACHE_DIR}" \
  --output reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json \
  --doc docs/STWM_MIXED_OBSERVED_SEMANTIC_PROTOTYPE_TARGETS_V11_20260502.md

"${PY}" code/stwm/tools/materialize_stwm_fstf_scaling_cache_v11_20260502.py
