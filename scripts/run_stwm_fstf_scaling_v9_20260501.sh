#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
cd "${REPO_ROOT}"
"${PY}" code/stwm/tools/run_stwm_fstf_scaling_v9_20260501.py
