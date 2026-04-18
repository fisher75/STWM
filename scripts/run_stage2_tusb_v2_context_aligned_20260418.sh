#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/chen034/workspace/stwm}"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

cd "${ROOT}"
exec "${PYTHON_BIN}" "${ROOT}/code/stwm/tools/run_stage2_tusb_v2_context_aligned_20260418.py" --mode run "$@"
