#!/usr/bin/env bash
set -euo pipefail
SESSION=tracewm_stage2_final_evidence_closure_20260414
SCRIPT=/home/chen034/workspace/stwm/scripts/run_tracewm_stage2_final_evidence_closure_20260414.sh
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux send-keys -t "$SESSION":0 "bash $SCRIPT" C-m
else
  tmux new-session -d -s "$SESSION" "bash $SCRIPT"
fi
tmux ls | grep "$SESSION" || true
