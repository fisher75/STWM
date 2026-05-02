#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
RUN_SCRIPT="$ROOT/scripts/run_ostf_physics_residual_v18_20260502.sh"
LOG_DIR="$ROOT/logs/stwm_ostf_v18"
REPORT_DIR="$ROOT/reports/stwm_ostf_v18_runs"
CKPT_DIR="$ROOT/outputs/checkpoints/stwm_ostf_v18"

GPU_IDS="${GPU_IDS:-1}"
HORIZON="${HORIZON:-8}"
SEED="${SEED:-42}"
STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
FORCE_RERUN="${FORCE_RERUN:-0}"

MODELS=(
  constant_velocity_copy
  affine_motion_prior_only
  dct_residual_prior_only
  v18_physics_residual_m128
  v18_physics_residual_m512
  v18_wo_semantic_memory
  v18_wo_dense_points
  v18_wo_residual_decoder
  v18_wo_affine_prior
  v18_wo_cv_prior
)

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$CKPT_DIR"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
gpu_count="${#GPU_LIST[@]}"
if [[ "$gpu_count" -eq 0 ]]; then
  echo "No GPU ids provided" >&2
  exit 1
fi

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPU_LIST[$((idx % gpu_count))]}"
  exp="${model}_seed${SEED}_h${HORIZON}"
  session="ostf_v18_${model}_s${SEED}_h${HORIZON}"
  log="$LOG_DIR/${exp}.log"
  report="$REPORT_DIR/${exp}.json"
  final_ckpt="$CKPT_DIR/${exp}_final.pt"
  best_ckpt="$CKPT_DIR/${exp}_best.pt"
  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -f "$log" "$report" "$final_ckpt" "$best_ckpt"
  elif [[ -f "$report" ]]; then
    echo "skip existing $exp"
    continue
  fi
  tmux kill-session -t "$session" >/dev/null 2>&1 || true
  cmd="cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$ROOT/code:\${PYTHONPATH:-}' STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic && '$RUN_SCRIPT' --experiment-name '$exp' --model-kind '$model' --horizon '$HORIZON' --seed '$SEED' --steps '$STEPS' --batch-size '$BATCH_SIZE' 2>&1 | tee '$log'"
  tmux new-session -d -s "$session" "bash -lc \"$cmd\""
  echo "$session gpu=$gpu log=$log report=$report"
done
