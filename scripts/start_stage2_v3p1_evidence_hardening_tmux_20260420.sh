#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
SESSION="tracewm_stage2_v3p1_evidence_hardening_20260420"
LOG="$ROOT/logs/stage2_v3p1_evidence_hardening_20260420.log"
CMD="$ROOT/scripts/run_stage2_v3p1_evidence_hardening_20260420.sh"

cd "$ROOT"
mkdir -p "$ROOT/logs"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
fi

export STWM_PROC_TITLE="${STWM_PROC_TITLE:-python}"
export STWM_PROC_TITLE_MODE="${STWM_PROC_TITLE_MODE:-generic}"

: > "$LOG"
EXTRA_ARGS=""
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS=" $(printf '%q ' "$@")"
fi

tmux new-session -d -s "$SESSION" "bash -lc 'cd \"$ROOT\"; \"$CMD\" --mode run${EXTRA_ARGS} >> \"$LOG\" 2>&1'"
echo "$SESSION"
