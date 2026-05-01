#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="${SESSION:-fstf_strong_copyaware_baselines_v8_launcher}"
LOG="${LOG:-${REPO_ROOT}/logs/fstf_strong_copyaware_baselines_v8_20260501/launcher.log}"
mkdir -p "$(dirname "${LOG}")"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "cd '${REPO_ROOT}' && exec bash scripts/run_fstf_strong_copyaware_baselines_v8_20260501.sh > '${LOG}' 2>&1"
echo "${SESSION}"
