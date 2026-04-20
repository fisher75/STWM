#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
SESSION="stwm_decisive_validation_20260420"
LOG="$ROOT/logs/stwm_decisive_validation_20260420.log"
mkdir -p "$(dirname "$LOG")"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash -lc 'cd \"$ROOT\" && ./scripts/run_stwm_decisive_validation_20260420.sh; echo; echo \"[stwm_decisive_validation_20260420 finished]\"; exec bash'"
echo "started tmux session: $SESSION"
