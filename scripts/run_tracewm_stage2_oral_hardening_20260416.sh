#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/raid/chen034/workspace/stwm}"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
LOG_PATH="${WORK_ROOT}/logs/stage2_oral_hardening_20260416.log"
POLL_SECONDS="${POLL_SECONDS:-120}"
PREDECODE_CACHE_PATH="${PREDECODE_CACHE_PATH:-${WORK_ROOT}/data/processed/stage2_predecode_cache_20260416}"
export WORK_ROOT

export PYTHONPATH="${WORK_ROOT}/code:${PYTHONPATH:-}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

mkdir -p "$(dirname "${LOG_PATH}")"

step_done() {
  local path="$1"
  [ -f "$path" ]
}

{
  echo "[$(date -Iseconds)] oral_hardening_start"
  if ! step_done "${PREDECODE_CACHE_PATH}/index.json"; then
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/build_stage2_predecode_cache_20260416.py" \
      --cache-root "${PREDECODE_CACHE_PATH}"
  fi
  if ! step_done "${WORK_ROOT}/configs/recommended_stage2_runtime_20260416.json"; then
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/benchmark_stage2_runtime_pipeline_20260416.py" \
      --predecode-cache-path "${PREDECODE_CACHE_PATH}"
  fi
  if ! step_done "${WORK_ROOT}/reports/stage2_state_identifiability_protocol_v3_20260416.json"; then
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/build_stage2_state_identifiability_protocol_v3_20260416.py"
  fi
  if ! step_done "${WORK_ROOT}/reports/stage2_state_identifiability_eval_v3_20260416.json"; then
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_state_identifiability_eval_v3_20260416.py"
  fi
  if ! step_done "${WORK_ROOT}/reports/stage2_mechanism_ablation_fix_v3_launch_20260416.json"; then
    "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_v3_20260416.py" \
      --mode launch \
      --work-root "${WORK_ROOT}" \
      --predecode-cache-path "${PREDECODE_CACHE_PATH}"
  fi
  if ! step_done "${WORK_ROOT}/reports/stage2_local_temporal_semantic_branch_launch_20260416.json"; then
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
import os
from pathlib import Path
root = Path(os.environ["WORK_ROOT"]) / "reports"
paths = [
    root / "stage2_mechanism_ablation_fix_v3_summary_20260416.json",
    root / "stage2_local_temporal_semantic_branch_summary_20260416.json",
]
for p in paths:
    if not p.exists():
        raise SystemExit(1)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not bool(payload.get("all_runs_terminal", False)):
        raise SystemExit(1)
raise SystemExit(0)
PY
    then
      "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_mechanism_ablation_fix_v3_20260416.py" \
        --mode diagnose --work-root "${WORK_ROOT}" --predecode-cache-path "${PREDECODE_CACHE_PATH}"
      "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_local_temporal_semantic_branch_20260416.py" \
        --mode diagnose --work-root "${WORK_ROOT}" --predecode-cache-path "${PREDECODE_CACHE_PATH}"
      "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_tracewm_stage1_stage2_qualitative_pack_v9_20260416.py"
      "${PYTHON_BIN}" "${WORK_ROOT}/code/stwm/tools/run_stage2_oral_hardening_20260416.py" --mode diagnose
      break
    fi
    sleep "${POLL_SECONDS}"
  done
  echo "[$(date -Iseconds)] oral_hardening_done"
} >> "${LOG_PATH}" 2>&1
