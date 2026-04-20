#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"

cd "$ROOT"
python code/stwm/tools/run_stwm_top_tier_one_last_fix_20260420.py "$@"
