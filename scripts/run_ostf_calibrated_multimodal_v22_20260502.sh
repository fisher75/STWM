#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

subcmd="${1:-}"
shift || true

case "$subcmd" in
  audit-v21)
    exec "$PY" code/stwm/tools/audit_ostf_v21_mode_selection_v22_20260502.py "$@"
    ;;
  train)
    exec "$PY" code/stwm/tools/train_ostf_calibrated_multimodal_v22_20260502.py "$@"
    ;;
  eval)
    exec "$PY" code/stwm/tools/eval_ostf_calibrated_multimodal_v22_20260502.py "$@"
    ;;
  visualize)
    exec "$PY" code/stwm/tools/render_ostf_calibrated_multimodal_v22_20260502.py "$@"
    ;;
  point-selection)
    exec "$PY" code/stwm/tools/materialize_ostf_point_selection_v22_20260502.py "$@"
    ;;
  all)
    "$0" audit-v21
    "$0" eval
    "$0" visualize
    ;;
  *)
    echo "usage: $0 {audit-v21|train|eval|visualize|point-selection|all} [args...]" >&2
    exit 2
    ;;
esac
