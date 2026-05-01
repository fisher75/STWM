#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
PY="${PY:-/home/chen034/miniconda3/envs/stwm/bin/python}"
export PYTHONPATH="${REPO_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED=1
cd "${REPO_ROOT}"

: "${CUDA_VISIBLE_DEVICES:=0}"
"${PY}" code/stwm/tools/eval_stwm_fstf_mechanism_ablation_v9_20260501.py --device cuda
