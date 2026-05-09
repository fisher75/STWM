#!/usr/bin/env bash
set -euo pipefail
ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
SESSION="stwm_ostf_v33_semantic_identity_smoke_20260509"
LOG="$ROOT/logs/stwm_ostf_v33_semantic_identity_smoke_20260509.log"
mkdir -p "$(dirname "$LOG")"
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION" || true
tmux new-session -d -s "$SESSION" "bash '$ROOT/scripts/run_ostf_v33_semantic_identity_smoke_20260509.sh' 2>&1 | tee '$LOG'"
echo "$SESSION"
