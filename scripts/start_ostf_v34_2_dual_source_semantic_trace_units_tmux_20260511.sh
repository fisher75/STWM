#!/usr/bin/env bash
set -euo pipefail
SESSION="${SESSION:-stwm_ostf_v34_2_dual_source_20260511}"
LOG="/raid/chen034/workspace/stwm/logs/stwm_ostf_v34_2_dual_source_20260511.log"
mkdir -p /raid/chen034/workspace/stwm/logs
tmux new-session -d -s "$SESSION" "bash /raid/chen034/workspace/stwm/scripts/run_ostf_v34_2_dual_source_semantic_trace_units_smoke_20260511.sh 2>&1 | tee -a '$LOG'"
echo "$SESSION"
echo "$LOG"
