#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/raid/chen034/workspace/stwm}"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
LOG_PATH="${WORK_ROOT}/logs/stage2_top_tier_hardening_v2_20260416.log"
POLL_SECONDS="${POLL_SECONDS:-120}"

export PYTHONPATH="${WORK_ROOT}/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

mkdir -p "$(dirname "${LOG_PATH}")"

{
  echo "[$(date -Iseconds)] hardening_v2_start"
  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_v2_20260416.py" --mode launch --work-root "${WORK_ROOT}"
  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/build_stage2_state_identifiability_protocol_v2_20260416.py"
  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_state_identifiability_eval_v2_20260416.py"

  while true; do
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_v2_20260416.py" --mode summarize --work-root "${WORK_ROOT}" >/dev/null
    if "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
p = Path("/raid/chen034/workspace/stwm/reports/stage2_mechanism_ablation_fix_v2_summary_20260416.json")
if not p.exists():
    raise SystemExit(1)
with p.open("r", encoding="utf-8") as fh:
    payload = json.load(fh)
raise SystemExit(0 if bool(payload.get("all_runs_terminal", False)) else 1)
PY
    then
      "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_v2_20260416.py" --mode diagnose --work-root "${WORK_ROOT}"
      "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v8_20260416.py"
      "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_top_tier_hardening_v2_20260416.py" --mode diagnose
      break
    fi
    sleep "${POLL_SECONDS}"
  done
  echo "[$(date -Iseconds)] hardening_v2_done"
} >> "${LOG_PATH}" 2>&1
