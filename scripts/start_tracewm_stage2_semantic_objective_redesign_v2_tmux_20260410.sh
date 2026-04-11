#!/usr/bin/env bash
set -euo pipefail

STWM_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
SESSION="tracewm_stage2_semobjv2_plus_codec_audit_20260410"
WINDOW="semobjv2_main"
LOG_PATH="${STWM_ROOT}/logs/tracewm_stage2_semobjv2_plus_codec_audit_20260410.log"
RUN_SCRIPT="${STWM_ROOT}/scripts/run_tracewm_stage2_semantic_objective_redesign_v2_20260410.sh"

mkdir -p "${STWM_ROOT}/logs"
chmod +x "${RUN_SCRIPT}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux new-window -t "${SESSION}" -n "${WINDOW}" "cd '${STWM_ROOT}' && '${RUN_SCRIPT}'"
else
  tmux new-session -d -s "${SESSION}" -n "${WINDOW}" "cd '${STWM_ROOT}' && '${RUN_SCRIPT}'"
fi

echo "tmux_session=${SESSION}"
echo "log_path=${LOG_PATH}"
