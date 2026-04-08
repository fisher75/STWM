#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="tracewm_stage1_v2_20260408"
RUN_SCRIPT="/home/chen034/workspace/stwm/scripts/run_tracewm_stage1_v2_20260408.sh"

if ! command -v tmux >/dev/null 2>&1; then
  echo "[stage1-v2-tmux] tmux_not_found"
  exit 127
fi

if [[ ! -x "$RUN_SCRIPT" ]]; then
  chmod +x "$RUN_SCRIPT"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
  echo "[stage1-v2-tmux] killed_existing_session=$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" "bash $RUN_SCRIPT"

echo "[stage1-v2-tmux] started_session=$SESSION_NAME"
echo "[stage1-v2-tmux] attach_command=tmux attach -t $SESSION_NAME"
