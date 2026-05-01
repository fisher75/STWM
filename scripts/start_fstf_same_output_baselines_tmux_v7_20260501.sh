#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="${SESSION:-fstf_same_output_baselines_v7_launcher}"
LOG="${LOG:-${REPO_ROOT}/logs/fstf_same_output_baselines_v7_20260501/launcher.log}"
mkdir -p "$(dirname "${LOG}")"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; exec scripts/run_fstf_same_output_baselines_v7_20260501.sh > ${LOG} 2>&1'"
echo "${SESSION}"
