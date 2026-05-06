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
  verify-cache)
    exec "$PY" code/stwm/tools/verify_ostf_traceanything_cache_v26_20260502.py "$@"
    ;;
  train)
    exec "$PY" code/stwm/tools/train_ostf_traceanything_v26_20260502.py "$@"
    ;;
  eval)
    exec "$PY" code/stwm/tools/eval_ostf_traceanything_v26_20260502.py "$@"
    ;;
  visualize)
    exec "$PY" code/stwm/tools/render_ostf_traceanything_v26_20260502.py "$@"
    ;;
  *)
    echo "usage: $0 {verify-cache|train|eval|visualize} [args...]" >&2
    exit 2
    ;;
esac
