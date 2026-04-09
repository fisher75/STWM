#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
LOG_PATH="${TRACEWM_EVIDENCE_HARDENING_LOG:-$WORK_ROOT/logs/tracewm_evidence_hardening_20260409.log}"
PYTHON_BIN="${TRACEWM_PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
DATA_ROOT="${TRACEWM_DATA_ROOT:-/home/chen034/workspace/data}"

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[tracewm-evidence-hardening] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[tracewm-evidence-hardening] work_root=$WORK_ROOT"
echo "[tracewm-evidence-hardening] python=$PYTHON_BIN"

echo "[tracewm-evidence-hardening] step=refresh_stage2_external_eval_completion"
bash "$WORK_ROOT/scripts/run_tracewm_stage2_external_eval_completion_20260408.sh"

echo "[tracewm-evidence-hardening] step=run_evidence_audit"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tools/run_tracewm_evidence_hardening_20260409.py" \
  --repo-root "$WORK_ROOT" \
  --data-root "$DATA_ROOT"

echo "[tracewm-evidence-hardening] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
