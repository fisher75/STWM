#!/usr/bin/env bash
set -euo pipefail

STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
SESSION="tracewm_stage2_ljs_aligned_semantic_diagnosis_and_rescue_20260410"

if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${SESSION}" -n main "sleep infinity"
fi

tmux new-window -t "${SESSION}" -n semres_launch "bash ${STWM_ROOT}/scripts/run_tracewm_stage2_semantic_rescue_wave0_20260410.sh --mode launch --tmux-session ${SESSION}"
echo "launched semantic rescue wave0 inside tmux session: ${SESSION}"
