#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
LOG_PATH="${TRACEWM_STAGE2_EXTERNAL_EVAL_CLAIM_BOUNDARY_FIX_LOG:-$WORK_ROOT/logs/tracewm_stage2_external_eval_claim_boundary_fix_20260409.log}"

if [[ -n "${TRACEWM_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${TRACEWM_PYTHON_BIN}"
elif [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage2-external-eval-claim-boundary-fix] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-external-eval-claim-boundary-fix] python=$PYTHON_BIN"

echo "[stage2-external-eval-claim-boundary-fix] step=apply_claim_boundary_fix"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tools/run_stage2_external_eval_claim_boundary_fix_20260409.py"

echo "[stage2-external-eval-claim-boundary-fix] step=check_consistency"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tools/check_stage2_external_eval_status_consistency.py" \
  --output-json "$WORK_ROOT/reports/stage2_external_eval_status_consistency_20260409.json"

echo "[stage2-external-eval-claim-boundary-fix] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
