#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
SESSION="stwm_final_credibility_utility_20260420"
LOG="$ROOT/logs/stwm_final_credibility_utility_20260420.log"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

ARGS=("$@")
CMD="cd $ROOT && bash scripts/run_stwm_final_credibility_utility_20260420.sh"
for arg in "${ARGS[@]}"; do
  CMD+=" $(printf '%q' "$arg")"
done
CMD+=" | tee -a $LOG"

tmux new-session -d -s "$SESSION" "$CMD"
echo "$SESSION"
