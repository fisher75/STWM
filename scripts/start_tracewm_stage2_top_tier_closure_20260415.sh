#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
SESSION="tracewm_stage2_top_tier_closure_20260415"
LOG_PATH="${ROOT}/logs/stage2_top_tier_closure_20260415.log"
RUN_SCRIPT="${ROOT}/scripts/run_tracewm_stage2_top_tier_closure_20260415.sh"

mkdir -p "$(dirname "${LOG_PATH}")"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${ROOT} && ${RUN_SCRIPT} >> ${LOG_PATH} 2>&1'"
echo "${SESSION}"
