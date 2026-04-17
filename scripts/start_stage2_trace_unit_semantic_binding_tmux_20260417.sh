#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/chen034/workspace/stwm"
SESSION="tracewm_stage2_trace_unit_semantic_binding_20260417"
RUN_SCRIPT="${ROOT}/scripts/run_stage2_trace_unit_semantic_binding_20260417.sh"

cd "${ROOT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" "bash -lc '${RUN_SCRIPT}'"
tmux rename-window -t "${SESSION}:0" monitor

echo "${SESSION}"
