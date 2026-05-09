#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
mkdir -p logs
SESSION=stwm_ostf_v33_2_visual_semantic_identity_smoke_20260509
LOG=logs/stwm_ostf_v33_2_visual_semantic_identity_smoke_20260509.log
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/run_ostf_v33_2_visual_semantic_identity_smoke_20260509.sh > '$LOG' 2>&1"
echo "$SESSION $LOG"
