#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT/code:$ROOT/third_party/TraceAnything:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

"$PY" code/stwm/tools/run_traceanything_object_trajectory_teacher_v2_20260502.py "$@"
