#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/raid/chen034/workspace/stwm}"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
LOG_PATH="${WORK_ROOT}/logs/stage2_tusb_v2_20260418.log"

mkdir -p "${WORK_ROOT}/logs"
cd "${WORK_ROOT}"

export PYTHONPATH="${WORK_ROOT}/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${WORK_ROOT}/reports/stage2_tusb_v2_cache_health_20260418.json" ]] || [[ ! -f "${WORK_ROOT}/reports/stage2_multi_entity_tusb_data_20260418.json" ]]; then
  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/build_stage2_predecode_cache_20260418.py" --max-samples-per-dataset 32 >> "${LOG_PATH}" 2>&1
fi
if [[ ! -f "${WORK_ROOT}/reports/stage2_tusb_teacher_prior_v2_20260418.json" ]]; then
  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/build_stage2_teacher_semantic_cache_v2_20260418.py" --device cpu >> "${LOG_PATH}" 2>&1
fi
"${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_tusb_v2_20260418.py" --mode run >> "${LOG_PATH}" 2>&1
