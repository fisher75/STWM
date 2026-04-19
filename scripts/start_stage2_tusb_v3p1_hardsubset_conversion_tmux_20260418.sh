#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
cd "$ROOT"

SESSION="tracewm_stage2_tusb_v3p1_hardsubset_conversion_20260418"
LOG_PATH="/home/chen034/workspace/stwm/logs/stage2_tusb_v3p1_hardsubset_conversion_20260418.log"

mkdir -p "$(dirname "$LOG_PATH")"
: > "$LOG_PATH"

export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"
export PYTHONUNBUFFERED=1

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

tmux new-session -d -s "$SESSION" "cd '$ROOT' && bash scripts/run_stage2_tusb_v3p1_hardsubset_conversion_20260418.sh --mode run >> '$LOG_PATH' 2>&1"
echo "$SESSION"
