#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
SESSION="tracewm_evidence_hardening_20260409"
RUN_SCRIPT="$WORK_ROOT/scripts/run_tracewm_evidence_hardening_20260409.sh"
LOG_PATH="${TRACEWM_EVIDENCE_HARDENING_LOG:-$WORK_ROOT/logs/tracewm_evidence_hardening_20260409.log}"

if [[ ! -x "$RUN_SCRIPT" ]]; then
  chmod +x "$RUN_SCRIPT"
fi

mkdir -p "$WORK_ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "bash '$RUN_SCRIPT'"

echo "session=$SESSION"
echo "runner=$RUN_SCRIPT"
echo "log=$LOG_PATH"
echo "attach: tmux attach -t $SESSION"
