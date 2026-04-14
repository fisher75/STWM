#!/usr/bin/env bash
set -euo pipefail

export STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
export PYTHONPATH="${STWM_ROOT}/code:${PYTHONPATH:-}"
export HF_HOME="${STWM_ROOT}/models/hf_cache"
export TORCH_HOME="${STWM_ROOT}/models/torch_cache"
export TMPDIR="${STWM_ROOT}/tmp"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
LOG_PATH="${STWM_ROOT}/logs/stage2_calibration_only_fullscale_wave1_20260413.log"
mkdir -p "${STWM_ROOT}/logs" "${STWM_ROOT}/tmp"

touch "${LOG_PATH}"
exec >> "${LOG_PATH}" 2>&1
echo "[`date -Iseconds`] calibration_only_wave1_run_script_start"
PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${STWM_ROOT}/code/stwm/tools/run_tracewm_stage2_calibration_only_fullscale_wave1_20260413.py" --mode all "$@"
