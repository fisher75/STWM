#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
SESSION="${SESSION:-stwm_ostf_v33_8_ablation_safe_identity_semantic_20260510}"
LOG="${LOG:-$ROOT/logs/stwm_ostf_v33_8_ablation_safe_identity_semantic_20260510.log}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$(dirname "$LOG")"
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash -lc 'cd $ROOT && export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 STWM_PYTHON=/home/chen034/miniconda3/envs/stwm/bin/python PYTHONPATH=$ROOT/code:\${PYTHONPATH:-}; bash scripts/run_ostf_v33_8_ablation_safe_identity_semantic_smoke_20260510.sh' > '$LOG' 2>&1"
echo "$SESSION $LOG"
