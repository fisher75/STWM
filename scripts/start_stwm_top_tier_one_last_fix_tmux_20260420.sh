#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
SESSION="stwm_top_tier_one_last_fix_20260420"
LOG="$ROOT/logs/stwm_top_tier_one_last_fix_20260420.log"

mkdir -p "$(dirname "$LOG")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "bash -lc 'cd \"$ROOT\" && export STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic && ./scripts/run_stwm_top_tier_one_last_fix_20260420.sh >> \"$LOG\" 2>&1; exec bash'"
echo "$SESSION"
