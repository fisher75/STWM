#!/usr/bin/env bash
set -euo pipefail

ROOT="${STWM_ROOT:-/home/chen034/workspace/stwm}"
PY="${STWM_PYTHON:-/home/chen034/miniconda3/envs/stwm/bin/python}"
RUN_SCRIPT="$ROOT/scripts/run_ostf_multitrace_world_model_v17_20260502.sh"
LOG_DIR="$ROOT/logs/stwm_ostf_v17"
REPORT_DIR="$ROOT/reports/stwm_ostf_v17_runs"
CKPT_DIR="$ROOT/outputs/checkpoints/stwm_ostf_v17"

GPU_IDS="${GPU_IDS:-1}"
HORIZON="${HORIZON:-8}"
SEED="${SEED:-42}"
STEPS="${STEPS:-800}"
BATCH_SIZE="${BATCH_SIZE:-8}"
FORCE_RERUN="${FORCE_RERUN:-0}"

mkdir -p "$LOG_DIR" "$REPORT_DIR" "$CKPT_DIR"

MODELS=(
  m1_anchor_stwm
  m1_anchor_stwm_m128
  constant_velocity_copy
  point_transformer_dense
  ostf_multitrace_m128
  ostf_multitrace_m512
  ostf_m512_wo_semantic_memory
  ostf_m512_wo_dense_point_input
  ostf_m512_wo_point_residual_decoder
)

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
  session="ostf_v17_${model}_s${SEED}_h${HORIZON}"
  log_path="$LOG_DIR/${exp}.log"
  report_path="$REPORT_DIR/${exp}.json"
  ckpt_path="$CKPT_DIR/${exp}.pt"
  if [[ "$FORCE_RERUN" == "1" ]]; then
    rm -f "$report_path" "$ckpt_path" "$log_path"
  elif [[ -f "$report_path" && -f "$ckpt_path" ]]; then
    echo "skip existing $exp"
    continue
  fi
  tmux kill-session -t "$session" >/dev/null 2>&1 || true
  cmd="cd '$ROOT' && export CUDA_VISIBLE_DEVICES='$gpu' PYTHONPATH='$ROOT/code:\${PYTHONPATH:-}' STWM_PROC_TITLE=python STWM_PROC_TITLE_MODE=generic && '$RUN_SCRIPT' --experiment-name '$exp' --model-kind '$model' --horizon '$HORIZON' --seed '$SEED' --steps '$STEPS' --batch-size '$BATCH_SIZE' 2>&1 | tee '$log_path'"
  tmux new-session -d -s "$session" "bash -lc \"$cmd\""
  echo "$session gpu=$gpu log=$log_path report=$report_path"
done
