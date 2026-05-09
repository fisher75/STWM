#!/usr/bin/env bash
set -euo pipefail
ROOT="/raid/chen034/workspace/stwm"
SESSION="stwm_ostf_v32_recurrent_field_smoke_20260509"
cd "$ROOT"
tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
tmux new-session -d -s "$SESSION" "bash scripts/run_ostf_v32_recurrent_field_smoke_20260509.sh | tee logs/stwm_ostf_v32_recurrent_field_smoke_20260509.log"
echo "$SESSION"
