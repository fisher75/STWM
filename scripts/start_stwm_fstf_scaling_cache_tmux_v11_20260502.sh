#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="${SESSION:-stwm_fstf_scaling_cache_v11}"
mkdir -p "${REPO_ROOT}/outputs/logs/stwm_fstf_scaling_v11_20260502"
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; scripts/run_stwm_fstf_scaling_cache_v11_20260502.sh > outputs/logs/stwm_fstf_scaling_v11_20260502/cache_materialization.log 2>&1'"
echo "${SESSION}"
