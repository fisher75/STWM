#!/usr/bin/env bash
set -euo pipefail

export STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
export PYTHONPATH="${STWM_ROOT}/code:${PYTHONPATH:-}"
export HF_HOME="${STWM_ROOT}/models/hf_cache"
export TORCH_HOME="${STWM_ROOT}/models/torch_cache"
export TMPDIR="${STWM_ROOT}/tmp"

PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
LOG_PATH="${STWM_ROOT}/logs/tracewm_stage2_ljs_aligned_semantic_diagnosis_and_rescue_20260410.log"

mkdir -p "${STWM_ROOT}/logs" "${STWM_ROOT}/reports" "${STWM_ROOT}/docs" "${STWM_ROOT}/manifests/protocol_v2"

cd "${STWM_ROOT}"
"${PYTHON_BIN}" "${STWM_ROOT}/code/stwm/tools/run_tracewm_stage2_ljs_semantic_diagnosis_and_rescue_20260410.py" 2>&1 | tee -a "${LOG_PATH}"
