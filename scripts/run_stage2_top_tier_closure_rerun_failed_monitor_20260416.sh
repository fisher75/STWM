#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/raid/chen034/workspace/stwm}"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
PYTHONPATH="${WORK_ROOT}/code:${PYTHONPATH:-}"
LOG_PATH="${WORK_ROOT}/logs/stage2_top_tier_closure_20260415.log"
POLL_SECONDS="${POLL_SECONDS:-120}"

export PYTHONPATH
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

MECH_SCRIPT="${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_20260415.py"
TOP_SCRIPT="${WORK_ROOT}/code/stwm/tools/run_stage2_top_tier_closure_20260415.py"
SUMMARY_JSON="${WORK_ROOT}/reports/stage2_mechanism_ablation_fix_summary_20260415.json"

mkdir -p "$(dirname "${LOG_PATH}")"

while true; do
  "${PYTHON_BIN}" "${MECH_SCRIPT}" --mode summarize --work-root "${WORK_ROOT}" >> "${LOG_PATH}" 2>&1
  if "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
p = Path("/raid/chen034/workspace/stwm/reports/stage2_mechanism_ablation_fix_summary_20260415.json")
with p.open("r", encoding="utf-8") as fh:
    payload = json.load(fh)
raise SystemExit(0 if bool(payload.get("all_runs_terminal", False)) else 1)
PY
  then
    "${PYTHON_BIN}" "${MECH_SCRIPT}" --mode diagnose --work-root "${WORK_ROOT}" >> "${LOG_PATH}" 2>&1
    "${PYTHON_BIN}" "${TOP_SCRIPT}" --mode diagnose >> "${LOG_PATH}" 2>&1
    exit 0
  fi
  sleep "${POLL_SECONDS}"
done
