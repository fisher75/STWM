#!/usr/bin/env bash
set -euo pipefail

export STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
export PYTHONPATH="${STWM_ROOT}/code:${PYTHONPATH:-}"
export HF_HOME="${STWM_ROOT}/models/hf_cache"
export TORCH_HOME="${STWM_ROOT}/models/torch_cache"
export TMPDIR="${STWM_ROOT}/tmp"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python:eval}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
mkdir -p "${STWM_ROOT}/logs" "${STWM_ROOT}/tmp"

exec "${PYTHON_BIN}" "${STWM_ROOT}/code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v4_20260413.py" "$@"
