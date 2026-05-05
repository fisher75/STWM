#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
PY=/home/chen034/miniconda3/envs/stwm/bin/python
export PYTHONPATH="$ROOT/code:$ROOT/third_party/TraceAnything:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec "$PY" -u "$ROOT/code/stwm/tools/run_traceanything_object_trajectory_teacher_v25_20260502.py" "$@"
