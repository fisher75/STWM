#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
cd "$ROOT"

export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

subcmd="${1:-}"
shift || true

case "$subcmd" in
  audit)
    exec "$PY" code/stwm/tools/audit_ostf_cv_hard_subset_v20_20260502.py "$@"
    ;;
  context-cache)
    exec "$PY" code/stwm/tools/materialize_ostf_context_features_v20_20260502.py "$@"
    ;;
  train)
    exec "$PY" code/stwm/tools/train_ostf_context_residual_v20_20260502.py "$@"
    ;;
  eval)
    exec "$PY" code/stwm/tools/eval_ostf_context_residual_v20_20260502.py "$@"
    ;;
  visualize)
    exec "$PY" code/stwm/tools/render_ostf_context_residual_v20_20260502.py "$@"
    ;;
  *)
    echo "usage: $0 {audit|context-cache|train|eval|visualize} [args...]" >&2
    exit 2
    ;;
esac
