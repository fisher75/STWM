#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
cd "$ROOT"
SESSION="stwm_ostf_v31_field_ablation_burst_20260509"
LOG="logs/stwm_ostf_v31_field_ablation_burst_20260509.log"
mkdir -p logs
tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" "bash scripts/run_ostf_v31_field_ablation_burst_20260509.sh 2>&1 | tee '$LOG'"
echo "$SESSION"
echo "$LOG"
