#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="${SESSION:-stwm_fstf_full_scaling_v11}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
STEPS="${STEPS:-5000}"
cd "${REPO_ROOT}"
mkdir -p outputs/logs/stwm_fstf_scaling_v11_20260502
tmux kill-session -t "${SESSION}" 2>/dev/null || true
tmux new-session -d -s "${SESSION}" "bash -lc 'cd ${REPO_ROOT}; set -euo pipefail; echo START_FULL_SCALING \$(date -Is); scripts/run_stwm_fstf_scaling_cache_v11_20260502.sh; MAX_PARALLEL=${MAX_PARALLEL} STEPS=${STEPS} scripts/run_stwm_fstf_prototype_scaling_v11_20260502.sh; MAX_PARALLEL=${MAX_PARALLEL} STEPS=${STEPS} scripts/run_stwm_fstf_model_size_scaling_v11_20260502.sh; scripts/run_stwm_fstf_horizon_scaling_v11_20260502.sh; scripts/run_stwm_fstf_trace_density_scaling_v11_20260502.sh; echo END_FULL_SCALING \$(date -Is)' > outputs/logs/stwm_fstf_scaling_v11_20260502/full_scaling_master.log 2>&1"
echo "${SESSION}"
