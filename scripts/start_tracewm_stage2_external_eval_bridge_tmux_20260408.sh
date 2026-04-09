#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
SESSION="tracewm_stage2_external_eval_bridge_20260408"
RUN_SCRIPT="$WORK_ROOT/scripts/run_tracewm_stage2_external_eval_bridge_20260408.sh"
LOG_PATH="$WORK_ROOT/logs/tracewm_stage2_external_eval_bridge_20260408.log"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  chmod +x "$RUN_SCRIPT"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

mkdir -p "$WORK_ROOT/logs"

tmux new-session -d -s "$SESSION" "bash '$RUN_SCRIPT'"

echo "session=$SESSION"
echo "runner=$RUN_SCRIPT"
echo "log=$LOG_PATH"
echo "attach: tmux attach -t $SESSION"
