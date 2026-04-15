#!/usr/bin/env bash
set -euo pipefail

SESSION=stwm_direct_cleanup_20260414
SCRIPT=/home/chen034/workspace/stwm/scripts/run_stwm_direct_cleanup_20260414.sh

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "bash $SCRIPT"
tmux ls | grep "$SESSION" || true
