#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="stwm_ostf_v29_benchmark_fix_20260508"
LOG="$ROOT/logs/stwm_ostf_v29_benchmark_fix_20260508.log"
RUN="$ROOT/scripts/run_ostf_v29_benchmark_fix_20260508.sh"

mkdir -p "$(dirname "$LOG")"

if tmux has-session -t "$SESSION" >/dev/null 2>&1; then
  echo "$SESSION already running"
  exit 0
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash '$RUN' > '$LOG' 2>&1"
echo "session=$SESSION"
echo "log=$LOG"
