#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="${SESSION:-stwm_fstf_scaling_v9_gate}"
LOG="${LOG:-${REPO_ROOT}/logs/stwm_fstf_scaling_v9_20260501.log}"
mkdir -p "$(dirname "${LOG}")"
cd "${REPO_ROOT}"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic bash scripts/run_stwm_fstf_scaling_v9_20260501.sh > '${LOG}' 2>&1"
echo "[scaling-v9] launched session=${SESSION} log=${LOG}"
