#!/usr/bin/env bash
set -euo pipefail

STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
SESSION="tracewm_stage2_semantic_objective_redesign_v7_20260413"
MAIN_WINDOW="semobjv7_main"
QUAL_WINDOW="qual_pack_v3"
LOG_PATH="${STWM_ROOT}/logs/stage2_semobjv7_20260413.log"
RUN_SCRIPT="${STWM_ROOT}/scripts/run_tracewm_stage2_semantic_objective_redesign_v7_20260413.sh"
QUAL_SCRIPT="${STWM_ROOT}/scripts/run_tracewm_stage1_stage2_qualitative_pack_v3_20260413.sh"

mkdir -p "${STWM_ROOT}/logs"
chmod +x "${RUN_SCRIPT}"
chmod +x "${QUAL_SCRIPT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  if ! tmux list-windows -t "${SESSION}" -F '#W' | grep -qx "${MAIN_WINDOW}"; then
    tmux new-window -t "${SESSION}" -n "${MAIN_WINDOW}" "cd '${STWM_ROOT}' && '${RUN_SCRIPT}'"
  fi
  if ! tmux list-windows -t "${SESSION}" -F '#W' | grep -qx "${QUAL_WINDOW}"; then
    tmux new-window -t "${SESSION}" -n "${QUAL_WINDOW}" "cd '${STWM_ROOT}' && '${QUAL_SCRIPT}'"
  fi
else
  tmux new-session -d -s "${SESSION}" -n "${MAIN_WINDOW}" "cd '${STWM_ROOT}' && '${RUN_SCRIPT}'"
  tmux new-window -t "${SESSION}" -n "${QUAL_WINDOW}" "cd '${STWM_ROOT}' && '${QUAL_SCRIPT}'"
fi

echo "tmux_session=${SESSION}"
echo "log_path=${LOG_PATH}"
