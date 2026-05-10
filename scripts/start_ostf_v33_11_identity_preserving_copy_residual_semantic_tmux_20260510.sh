#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
mkdir -p logs
SESSION=stwm_ostf_v33_11_identity_preserving_copy_residual_20260510
LOG=logs/stwm_ostf_v33_11_identity_preserving_copy_residual_20260510.log
GPU_ID=${GPU_ID:-1}
tmux new-session -d -s "$SESSION" "CUDA_VISIBLE_DEVICES=$GPU_ID bash scripts/run_ostf_v33_11_identity_preserving_copy_residual_semantic_smoke_20260510.sh 2>&1 | tee $LOG"
echo "$SESSION"
echo "$LOG"
