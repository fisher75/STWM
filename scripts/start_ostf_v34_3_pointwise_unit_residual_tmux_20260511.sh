#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-stwm_v34_3_pointwise_unit_residual}
cd /raid/chen034/workspace/stwm
tmux new-session -d -s "$SESSION" "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0} bash scripts/run_ostf_v34_3_pointwise_unit_residual_smoke_20260511.sh 2>&1 | tee logs/stwm_ostf_v34_3_pointwise_unit_residual_20260511.log"
echo "$SESSION"
