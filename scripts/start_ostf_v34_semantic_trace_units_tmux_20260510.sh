#!/usr/bin/env bash
set -euo pipefail
cd /raid/chen034/workspace/stwm
mkdir -p logs
SESSION=stwm_ostf_v34_semantic_trace_units_20260510
LOG=logs/stwm_ostf_v34_semantic_trace_units_20260510.log
tmux new-session -d -s "$SESSION" "bash scripts/run_ostf_v34_semantic_trace_units_smoke_20260510.sh 2>&1 | tee $LOG"
echo "$SESSION"
echo "$LOG"
