#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
LOG="$ROOT/logs/stwm_decisive_validation_20260420.log"
mkdir -p "$(dirname "$LOG")"

cd "$ROOT"
export STWM_PROC_TITLE=python
python code/stwm/tools/run_stwm_decisive_validation_20260420.py 2>&1 | tee "$LOG"
