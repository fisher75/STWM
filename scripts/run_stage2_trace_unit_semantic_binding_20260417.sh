#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/home/chen034/workspace/stwm}"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
LOG_PATH="${WORK_ROOT}/logs/stage2_trace_unit_semantic_binding_20260417.log"

mkdir -p "${WORK_ROOT}/logs"
cd "${WORK_ROOT}"

export PYTHONPATH="${WORK_ROOT}/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_trace_unit_semantic_binding_20260417.py" --mode run >> "${LOG_PATH}" 2>&1
