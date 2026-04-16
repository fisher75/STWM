#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/raid/chen034/workspace/stwm}"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
LOG_PATH="${WORK_ROOT}/logs/stage2_oral_hardening_20260416.log"
PREDECODE_CACHE_PATH="${PREDECODE_CACHE_PATH:-${WORK_ROOT}/data/processed/stage2_predecode_cache_20260416}"

export PYTHONPATH="${WORK_ROOT}/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

mkdir -p "$(dirname "${LOG_PATH}")"

{
  echo "[$(date -Iseconds)] postbench_monitor_start"

  while [ ! -f "${WORK_ROOT}/configs/recommended_stage2_runtime_20260416.json" ]; do
    sleep 20
  done
  echo "[$(date -Iseconds)] postbench_runtime_ready"

  while [ ! -f "${WORK_ROOT}/reports/stage2_state_identifiability_eval_v3_20260416.json" ]; do
    sleep 10
  done
  echo "[$(date -Iseconds)] postbench_eval_ready"

  if [ ! -f "${WORK_ROOT}/reports/stage2_mechanism_ablation_fix_v3_launch_20260416.json" ]; then
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_v3_20260416.py" \
      --mode launch \
      --work-root "${WORK_ROOT}" \
      --predecode-cache-path "${PREDECODE_CACHE_PATH}"
  fi

  if [ ! -f "${WORK_ROOT}/reports/stage2_local_temporal_semantic_branch_launch_20260416.json" ]; then
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_local_temporal_semantic_branch_20260416.py" \
      --mode launch \
      --work-root "${WORK_ROOT}" \
      --predecode-cache-path "${PREDECODE_CACHE_PATH}"
  fi

  while true; do
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_v3_20260416.py" \
      --mode summarize --work-root "${WORK_ROOT}" >/dev/null
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_local_temporal_semantic_branch_20260416.py" \
      --mode summarize --work-root "${WORK_ROOT}" >/dev/null

    if "${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path

root = Path("/raid/chen034/workspace/stwm/reports")
paths = [
    root / "stage2_mechanism_ablation_fix_v3_summary_20260416.json",
    root / "stage2_local_temporal_semantic_branch_summary_20260416.json",
]
for path in paths:
    if not path.exists():
        raise SystemExit(1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not bool(payload.get("all_runs_terminal", False)):
        raise SystemExit(1)
raise SystemExit(0)
PY
    then
      break
    fi

    sleep 120
  done

  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_v3_20260416.py" \
    --mode diagnose \
    --work-root "${WORK_ROOT}" \
    --predecode-cache-path "${PREDECODE_CACHE_PATH}"
  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_local_temporal_semantic_branch_20260416.py" \
    --mode diagnose \
    --work-root "${WORK_ROOT}" \
    --predecode-cache-path "${PREDECODE_CACHE_PATH}"
  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v9_20260416.py"
  "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_oral_hardening_20260416.py" --mode diagnose

  echo "[$(date -Iseconds)] postbench_monitor_done"
} >> "${LOG_PATH}" 2>&1
