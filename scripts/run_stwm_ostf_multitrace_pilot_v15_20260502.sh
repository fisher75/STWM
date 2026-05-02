#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/chen034/miniconda3/envs/stwm/bin/python}"
GPU_ID="${GPU_ID:-0}"
LOG_DIR="${ROOT}/logs/stwm_ostf_multitrace_pilot_v15_20260502"
mkdir -p "$LOG_DIR"
export PYTHONPATH="${ROOT}/code:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
LOG_PATH="${LOG_DIR}/pilot_gpu${GPU_ID}.log"
{
  echo "[start] $(date -Is)"
  echo "[cmd] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ${PYTHON_BIN} code/stwm/tools/train_stwm_ostf_multitrace_pilot_v15_20260502.py $*"
  "${PYTHON_BIN}" code/stwm/tools/train_stwm_ostf_multitrace_pilot_v15_20260502.py "$@"
  echo "[end] $(date -Is)"
} 2>&1 | tee "$LOG_PATH"
