#!/usr/bin/env bash
set -euo pipefail

ROOT="/raid/chen034/workspace/stwm"
PYTHON="/home/chen034/miniconda3/envs/stwm/bin/python"
CODE_ROOT="$ROOT/code"
LOG_DIR="$ROOT/outputs/logs/stwm_ostf_v21"
mkdir -p "$LOG_DIR"

GPU_LIST="${GPU_LIST:-1}"
MAIN_ONLY="${MAIN_ONLY:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

launch_one() {
  local gpu="$1"
  local exp="$2"
  local kind="$3"
  local steps="$4"
  local batch="$5"
  local init_ckpt="$6"
  local session="stwm_v21_${exp}"
  local log_path="$LOG_DIR/${exp}.log"
  local best_ckpt="$ROOT/outputs/checkpoints/stwm_ostf_v21/${exp}_best.pt"
  if [[ "$FORCE_RERUN" != "1" && -f "$best_ckpt" ]]; then
    echo "[skip] $exp already has $best_ckpt"
    return 0
  fi
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" "cd $ROOT && export PYTHONPATH=$CODE_ROOT && export STWM_PROC_TITLE=python && export STWM_PROC_TITLE_MODE=generic && export CUDA_VISIBLE_DEVICES=$gpu && $PYTHON $ROOT/code/stwm/tools/train_ostf_multimodal_v21_20260502.py --experiment-name $exp --model-kind $kind --horizon 8 --seed 42 --steps $steps --batch-size $batch --device cuda --eval-every 1500 --init-from-checkpoint $init_ckpt |& tee $log_path"
  echo "[launch] session=$session gpu=$gpu exp=$exp kind=$kind steps=$steps batch=$batch log=$log_path"
}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
gpu_idx=0
next_gpu() {
  local gpu="${GPUS[$gpu_idx]}"
  gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
  echo "$gpu"
}

launch_one "$(next_gpu)" \
  "v21_multimodal_m128_seed42_h8" \
  "v21_multimodal_m128" \
  "30000" \
  "4" \
  "outputs/checkpoints/stwm_ostf_v20/v20_context_residual_m128_seed42_h8_best.pt"

launch_one "$(next_gpu)" \
  "v21_multimodal_m512_seed42_h8" \
  "v21_multimodal_m512" \
  "30000" \
  "2" \
  "outputs/checkpoints/stwm_ostf_v20/v20_context_residual_m512_seed42_h8_best.pt"

if [[ "$MAIN_ONLY" == "1" ]]; then
  exit 0
fi

launch_one "$(next_gpu)" \
  "v21_multimodal_m512_wo_cv_seed42_h8" \
  "v21_multimodal_m512_wo_cv" \
  "20000" \
  "2" \
  "outputs/checkpoints/stwm_ostf_v20/v20_context_residual_m512_seed42_h8_best.pt"

launch_one "$(next_gpu)" \
  "v21_multimodal_m512_wo_diversity_seed42_h8" \
  "v21_multimodal_m512_wo_diversity" \
  "20000" \
  "2" \
  "outputs/checkpoints/stwm_ostf_v20/v20_context_residual_m512_seed42_h8_best.pt"

launch_one "$(next_gpu)" \
  "v21_multimodal_m512_wo_context_seed42_h8" \
  "v21_multimodal_m512_wo_context" \
  "20000" \
  "2" \
  "outputs/checkpoints/stwm_ostf_v20/v20_context_residual_m512_wo_context_seed42_h8_best.pt"

launch_one "$(next_gpu)" \
  "v21_multimodal_m512_wo_dense_points_seed42_h8" \
  "v21_multimodal_m512_wo_dense_points" \
  "20000" \
  "2" \
  "outputs/checkpoints/stwm_ostf_v20/v20_context_residual_m512_wo_dense_points_seed42_h8_best.pt"

launch_one "$(next_gpu)" \
  "v21_multimodal_m512_single_mode_seed42_h8" \
  "v21_multimodal_m512_single_mode" \
  "20000" \
  "2" \
  "outputs/checkpoints/stwm_ostf_v20/v20_context_residual_m512_single_hypothesis_seed42_h8_best.pt"
