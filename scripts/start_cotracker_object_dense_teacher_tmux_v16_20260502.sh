#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
cd "$ROOT"
mkdir -p logs/stwm_cotracker_object_dense_teacher_v16_20260502

TARGET_TOTAL="${TARGET_TOTAL:-500}"
MAX_SIDE="${MAX_SIDE:-512}"
GPUS_CSV="${GPUS:-1,4,5,6}"
IFS=',' read -r -a GPUS_ARR <<< "$GPUS_CSV"

COMBOS=("128 8" "512 8" "128 16" "512 16")

for idx in "${!COMBOS[@]}"; do
  read -r M H <<< "${COMBOS[$idx]}"
  GPU="${GPUS_ARR[$((idx % ${#GPUS_ARR[@]}))]}"
  SESSION="stwm_v16_cotracker_M${M}_H${H}"
  LOG="logs/stwm_cotracker_object_dense_teacher_v16_20260502/M${M}_H${H}.log"
  CMD="cd '$ROOT' && CUDA_VISIBLE_DEVICES=$GPU STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic scripts/run_cotracker_object_dense_teacher_v16_20260502.sh --m $M --horizon $H --target-total $TARGET_TOTAL --max-side $MAX_SIDE 2>&1 | tee '$LOG'"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "session_exists $SESSION"
  else
    tmux new-session -d -s "$SESSION" "$CMD"
    echo "launched $SESSION gpu=$GPU log=$LOG"
  fi
done
