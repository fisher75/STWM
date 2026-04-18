#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/chen034/workspace/stwm}"
SESSION="tracewm_stage2_tusb_v2_context_aligned_20260418"
LOG_PATH="${ROOT}/logs/stage2_tusb_v2_context_aligned_20260418.log"
SCRIPT="${ROOT}/scripts/run_stage2_tusb_v2_context_aligned_20260418.sh"

mkdir -p "${ROOT}/logs"
tmux has-session -t "${SESSION}" 2>/dev/null && tmux kill-session -t "${SESSION}" || true
tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${ROOT} && ${SCRIPT} >> ${LOG_PATH} 2>&1'"
echo "${SESSION}"
