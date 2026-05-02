#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
RUN_SCRIPT="$ROOT/scripts/run_ostf_refinement_v19_20260502.sh"
LOG_DIR="$ROOT/logs/stwm_ostf_v19"

GPU_ID="${GPU_ID:-1}"
SESSION="${SESSION:-ostf_v19_refine}"

mkdir -p "$LOG_DIR"

cmd="cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$GPU_ID' PYTHONPATH='$ROOT/code:\${PYTHONPATH:-}' STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic && bash"
tmux kill-session -t "$SESSION" >/dev/null 2>&1 || true
tmux new-session -d -s "$SESSION" "bash -lc \"$cmd\""
echo "$SESSION"
