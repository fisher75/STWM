#!/usr/bin/env bash
set -euo pipefail

STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
SESSION="tracewm_stage2_calibration_only_finalization_pack_20260414"
MAIN_WINDOW="calfinal_main"
LOG_PATH="${STWM_ROOT}/logs/stage2_calibration_only_finalization_pack_20260414.log"
RUN_SCRIPT="${STWM_ROOT}/scripts/run_tracewm_stage2_calibration_only_wave2_20260414.sh"
PID_FILE="${STWM_ROOT}/logs/stage2_calibration_only_finalization_pack_20260414.pid"

mkdir -p "${STWM_ROOT}/logs"
chmod +x "${RUN_SCRIPT}"

if [[ -f "${PID_FILE}" ]]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    kill "${OLD_PID}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

tmux new-session -d -s "${SESSION}" -n "${MAIN_WINDOW}" "cd '${STWM_ROOT}' && nohup '${RUN_SCRIPT}' >> '${LOG_PATH}' 2>&1 < /dev/null & echo \$! > '${PID_FILE}'; tail -f '${LOG_PATH}'"

echo "tmux_session=${SESSION}"
echo "log_path=${LOG_PATH}"
