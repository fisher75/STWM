#!/usr/bin/env bash
set -euo pipefail

SESSION=${SESSION:-stwm_v34_4_supervised_residual_gate}
cd /raid/chen034/workspace/stwm
mkdir -p logs
tmux new-session -d -s "$SESSION" "CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0} bash scripts/run_ostf_v34_4_supervised_residual_gate_smoke_20260511.sh 2>&1 | tee logs/stwm_ostf_v34_4_supervised_residual_gate_20260511.log"
echo "$SESSION"
