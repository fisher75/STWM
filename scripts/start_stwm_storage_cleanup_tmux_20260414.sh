#!/usr/bin/env bash
set -euo pipefail
SESSION=stwm_storage_cleanup_20260414
SCRIPT=/home/chen034/workspace/stwm/scripts/run_stwm_storage_cleanup_20260414.sh
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux send-keys -t "$SESSION":0 "bash $SCRIPT" C-m
else
  tmux new-session -d -s "$SESSION" "bash $SCRIPT"
fi
tmux ls | grep "$SESSION" || true
