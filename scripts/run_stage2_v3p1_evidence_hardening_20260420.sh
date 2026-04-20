#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
cd "$ROOT"

export PYTHONPATH="$ROOT/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED=1

PY_BIN="${PY_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
SCRIPT="$ROOT/code/stwm/tools/run_stage2_v3p1_evidence_hardening_20260420.py"

exec "$PY_BIN" "$SCRIPT" "$@"
