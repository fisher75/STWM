#!/usr/bin/env bash
set -euo pipefail

STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
SESSION="tracewm_stage2_ljs_aligned_semantic_diagnosis_and_rescue_20260410"
LOG_PATH="${STWM_ROOT}/logs/tracewm_stage2_ljs_aligned_semantic_diagnosis_and_rescue_20260410.log"

mkdir -p "${STWM_ROOT}/logs"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
else
  tmux new-session -d -s "${SESSION}" -n main "bash ${STWM_ROOT}/scripts/run_tracewm_stage2_ljs_semantic_diagnosis_and_rescue_20260410.sh"
  echo "started tmux session: ${SESSION}"
fi

echo "log: ${LOG_PATH}"
