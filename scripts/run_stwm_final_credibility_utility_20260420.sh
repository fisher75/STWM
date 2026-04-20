#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
cd "$ROOT"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"

exec "$PYTHON_BIN" code/stwm/tools/run_stwm_final_credibility_utility_20260420.py "$@"
