#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/raid/chen034/workspace/stwm}"
SESSION="tracewm_stage2_tusb_v2_20260418"
RUN_SCRIPT="${ROOT}/scripts/run_stage2_tusb_v2_20260418.sh"

cd "${ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" "bash -lc 'bash \"${RUN_SCRIPT}\"'"
tmux rename-window -t "${SESSION}:0" tusb_v2

echo "${SESSION}"
