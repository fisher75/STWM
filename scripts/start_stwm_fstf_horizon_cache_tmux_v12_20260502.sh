#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="${SESSION:-stwm_fstf_horizon_cache_v12}"
mkdir -p "${REPO_ROOT}/outputs/logs/stwm_fstf_scaling_v12_20260502"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; scripts/run_stwm_fstf_horizon_cache_v12_20260502.sh > outputs/logs/stwm_fstf_scaling_v12_20260502/horizon_cache_master.log 2>&1'"
echo "${SESSION}"
