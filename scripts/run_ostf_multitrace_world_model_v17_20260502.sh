#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

"$PY" code/stwm/tools/train_ostf_multitrace_world_model_v17_20260502.py "$@"
