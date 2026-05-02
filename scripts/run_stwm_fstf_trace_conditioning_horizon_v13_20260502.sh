#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${ROOT}/logs/stwm_fstf_trace_conditioning_horizon_v13_20260502"
mkdir -p "$LOG_DIR"
LOG_PATH="${LOG_DIR}/h16_h24_audit.log"

export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

{
  echo "[start] $(date -Is)"
  echo "[cmd] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON_BIN} code/stwm/tools/eval_stwm_fstf_trace_conditioning_horizon_v13_20260502.py"
  "${PYTHON_BIN}" code/stwm/tools/eval_stwm_fstf_trace_conditioning_horizon_v13_20260502.py "$@"
  echo "[end] $(date -Is)"
} 2>&1 | tee "$LOG_PATH"
