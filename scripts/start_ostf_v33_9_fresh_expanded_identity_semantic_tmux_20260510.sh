#!/usr/bin/env bash
set -euo pipefail
SESSION=stwm_ostf_v33_9_fresh_expanded_20260510
LOG=/raid/chen034/workspace/stwm/logs/stwm_ostf_v33_9_fresh_expanded_20260510.log
mkdir -p "$(dirname "$LOG")"
tmux new-session -d -s "$SESSION" "bash /raid/chen034/workspace/stwm/scripts/run_ostf_v33_9_fresh_expanded_identity_semantic_20260510.sh > '$LOG' 2>&1"
echo "$SESSION"
echo "$LOG"
