#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/raid/chen034/workspace/stwm}"
RUN_SCRIPT="$ROOT/scripts/run_ostf_calibrated_multimodal_v22_20260502.sh"
LOG_DIR="$ROOT/outputs/logs/stwm_ostf_v22"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"

mkdir -p "$LOG_DIR"

pick_gpus() {
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits \
    | sort -t',' -k2,2n -k3,3n \
    | awk -F',' '{gsub(/ /,"",$1); print $1}'
}

mapfile -t GPUS < <(pick_gpus)
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "no_gpus_available" >&2
  exit 1
fi

gpu_at() {
  local idx="$1"
  echo "${GPUS[$(( idx % ${#GPUS[@]} ))]}"
}

launch() {
  local session="$1"
  local gpu="$2"
  local log_name="$3"
  shift 3
  tmux kill-session -t "$session" >/dev/null 2>&1 || true
  tmux new-session -d -s "$session" \
    "cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$ROOT/code:\${PYTHONPATH:-}' STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic && bash '$RUN_SCRIPT' $* |& tee '$LOG_DIR/$log_name'"
  echo "[launch] session=$session gpu=$gpu log=$LOG_DIR/$log_name cmd=$*"
}

launch "stwm_v22_audit" "$(gpu_at 0)" "audit_v21_mode_selection.log" "audit-v21"
launch "stwm_v22_m128" "$(gpu_at 1)" "v22_m128.log" "train --experiment-name v22_calibrated_m128_seed42_h8 --model-kind v22_calibrated_m128 --horizon 8 --seed 42 --steps 30000 --batch-size 4 --device cuda --eval-every 1500 --init-from-checkpoint outputs/checkpoints/stwm_ostf_v21/v21_multimodal_m128_seed42_h8_best.pt"
launch "stwm_v22_m256" "$(gpu_at 2)" "v22_m256.log" "train --experiment-name v22_calibrated_m256_seed42_h8 --model-kind v22_calibrated_m256 --horizon 8 --seed 42 --steps 30000 --batch-size 2 --device cuda --eval-every 1500 --init-from-checkpoint outputs/checkpoints/stwm_ostf_v21/v21_multimodal_m512_seed42_h8_best.pt"
launch "stwm_v22_m256_nocontext" "$(gpu_at 3)" "v22_m256_wo_context.log" "train --experiment-name v22_calibrated_m256_wo_context_seed42_h8 --model-kind v22_calibrated_m256_wo_context --horizon 8 --seed 42 --steps 20000 --batch-size 2 --device cuda --eval-every 1500 --init-from-checkpoint outputs/checkpoints/stwm_ostf_v21/v21_multimodal_m512_wo_context_seed42_h8_best.pt"
launch "stwm_v22_m256_nodense" "$(gpu_at 4)" "v22_m256_wo_dense.log" "train --experiment-name v22_calibrated_m256_wo_dense_points_seed42_h8 --model-kind v22_calibrated_m256_wo_dense_points --horizon 8 --seed 42 --steps 20000 --batch-size 2 --device cuda --eval-every 1500 --init-from-checkpoint outputs/checkpoints/stwm_ostf_v21/v21_multimodal_m512_wo_dense_points_seed42_h8_best.pt"
launch "stwm_v22_m256_nosem" "$(gpu_at 5)" "v22_m256_wo_semantic.log" "train --experiment-name v22_calibrated_m256_wo_semantic_memory_seed42_h8 --model-kind v22_calibrated_m256_wo_semantic_memory --horizon 8 --seed 42 --steps 20000 --batch-size 2 --device cuda --eval-every 1500 --init-from-checkpoint outputs/checkpoints/stwm_ostf_v21/v21_multimodal_m512_seed42_h8_best.pt"
launch "stwm_v22_m256_single" "$(gpu_at 6)" "v22_m256_single.log" "train --experiment-name v22_calibrated_m256_single_mode_seed42_h8 --model-kind v22_calibrated_m256_single_mode --horizon 8 --seed 42 --steps 20000 --batch-size 2 --device cuda --eval-every 1500 --init-from-checkpoint outputs/checkpoints/stwm_ostf_v21/v21_multimodal_m512_single_mode_seed42_h8_best.pt"

printf '%s\n' \
  "stwm_v22_audit" \
  "stwm_v22_m128" \
  "stwm_v22_m256" \
  "stwm_v22_m256_nocontext" \
  "stwm_v22_m256_nodense" \
  "stwm_v22_m256_nosem" \
  "stwm_v22_m256_single"
