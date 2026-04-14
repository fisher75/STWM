#!/usr/bin/env bash
set -euo pipefail

STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
SESSION="tracewm_stage2_calibration_only_fullscale_wave1_20260413"
MAIN_WINDOW="calonly_main"
LOG_PATH="${STWM_ROOT}/logs/stage2_calibration_only_fullscale_wave1_20260413.log"
RUN_SCRIPT="${STWM_ROOT}/scripts/run_tracewm_stage2_calibration_only_fullscale_wave1_20260413.sh"

mkdir -p "${STWM_ROOT}/logs"
chmod +x "${RUN_SCRIPT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  if ! tmux list-windows -t "${SESSION}" -F '#W' | grep -qx "${MAIN_WINDOW}"; then
    tmux new-window -t "${SESSION}" -n "${MAIN_WINDOW}" "cd '${STWM_ROOT}' && '${RUN_SCRIPT}'"
  fi
else
  tmux new-session -d -s "${SESSION}" -n "${MAIN_WINDOW}" "cd '${STWM_ROOT}' && '${RUN_SCRIPT}'"
fi

echo "tmux_session=${SESSION}"
echo "log_path=${LOG_PATH}"
