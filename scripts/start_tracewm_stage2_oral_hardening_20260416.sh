#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${WORK_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="${SESSION:-tracewm_stage2_oral_hardening_20260416}"
RUN_SCRIPT="${WORK_ROOT}/scripts/run_tracewm_stage2_oral_hardening_20260416.sh"

chmod +x "${RUN_SCRIPT}"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "bash -lc '${RUN_SCRIPT}'"
tmux display-message -p -t "${SESSION}" '#S started'
