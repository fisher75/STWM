#!/usr/bin/env bash
set -euo pipefail

export STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
export PYTHONPATH="${STWM_ROOT}/code:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"

cd "${STWM_ROOT}"
"${PYTHON_BIN}" "${STWM_ROOT}/code/stwm/tools/run_tracewm_stage2_semantic_rescue_wave0_20260410.py" "$@"
