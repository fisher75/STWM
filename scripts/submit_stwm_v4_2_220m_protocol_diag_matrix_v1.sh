#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
TRAIN_SCRIPT="$REPO_ROOT/code/stwm/trainers/train_stwm_v4_2_real.py"
GPU_CLAIM_SCRIPT="$REPO_ROOT/scripts/gpu_auto_claim_run.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${STWM_D1_OUT_ROOT:-$REPO_ROOT/outputs/training/stwm_v4_2_220m_protocol_diag_v1}"
LOG_ROOT="${STWM_D1_LOG_ROOT:-$REPO_ROOT/logs}"
REPORT_ROOT="${STWM_D1_REPORT_ROOT:-$REPO_ROOT/reports}"

TRAIN_MANIFEST="${STWM_D1_TRAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/train_v2.json}"
PROTOCOL_MAIN_MANIFEST="${STWM_D1_PROTOCOL_MAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_main_v1.json}"
MODEL_PRESET="${STWM_D1_MODEL_PRESET:-prototype_220m_v4_2}"
PRESET_FILE="${STWM_D1_PRESET_FILE:-$REPO_ROOT/code/stwm/configs/model_presets_v4_2.json}"
DATA_ROOT="${STWM_D1_DATA_ROOT:-$REPO_ROOT/data/external}"

SEED="${STWM_D1_SEED:-42}"
STEPS="${STWM_D1_STEPS:-2000}"
SAMPLE_LIMIT="${STWM_D1_SAMPLE_LIMIT:-0}"

LAMBDA0="${STWM_D1_LAMBDA0:-0.5}"
LSEM_01="${STWM_D1_LSEM_01:-0.05}"
LSEM_025="${STWM_D1_LSEM_025:-0.125}"
LSEM_05="${STWM_D1_LSEM_05:-0.25}"
LSEM_10="${STWM_D1_LSEM_10:-0.5}"

GRAD_AUDIT_INTERVAL="${STWM_D1_GRAD_AUDIT_INTERVAL:-100}"
PROTOCOL_EVAL_INTERVAL="${STWM_D1_PROTOCOL_EVAL_INTERVAL:-500}"

POLL_SECONDS="${STWM_D1_GPU_POLL_SECONDS:-30}"
MAX_MEM_USED_MIB="${STWM_D1_GPU_MAX_MEM_USED_MIB:-50000}"
MAX_UTIL="${STWM_D1_GPU_MAX_UTIL:-70}"
CANDIDATE_GPUS="${STWM_D1_CANDIDATE_GPUS:-0,1,2,3,4,5,6,7}"

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$REPORT_ROOT"
SUBMIT_TSV="$REPORT_ROOT/stwm_v4_2_220m_protocol_diag_v1_submissions_${STAMP}.tsv"
echo "run_name\tpid\tlog_file\toutput_dir\tgradient_audit_json" > "$SUBMIT_TSV"

submit_run() {
  local run_name="$1"
  local lambda_sem="$2"
  local warmup_flag="$3"
  local extra_flag="$4"

  local out_dir="$OUT_ROOT/seed_${SEED}/${run_name}"
  local log_file="$LOG_ROOT/stwm_v4_2_220m_diag_v1_${run_name}_${STAMP}.log"
  local grad_json="$REPORT_ROOT/stwm_v4_2_gradient_audit_220m_seed${SEED}_${run_name}.json"

  mkdir -p "$out_dir"

  local cmd=(
    env "PYTHONPATH=$REPO_ROOT/code:${PYTHONPATH:-}"
    conda run --no-capture-output -n stwm
    python "$TRAIN_SCRIPT"
    --data-root "$DATA_ROOT"
    --manifest "$TRAIN_MANIFEST"
    --output-dir "$out_dir"
    --run-name "$run_name"
    --seed "$SEED"
    --steps "$STEPS"
    --target-epochs 0
    --min-optimizer-steps 0
    --max-optimizer-steps 0
    --sample-limit "$SAMPLE_LIMIT"
    --model-preset "$MODEL_PRESET"
    --preset-file "$PRESET_FILE"
    --use-teacher-priors
    --save-checkpoint
    --checkpoint-dir-name checkpoints
    --checkpoint-interval 100
    --milestone-interval 0
    --auto-resume
    --micro-batch-per-gpu 2
    --grad-accum 8
    --num-workers 12
    --prefetch-factor 2
    --persistent-workers
    --pin-memory
    --bf16
    --activation-checkpointing
    --lambda-traj 1.0
    --lambda-vis 0.25
    --lambda-sem "$lambda_sem"
    --lambda-reid 0.25
    --lambda-query 0.25
    --lambda-reconnect 0.1
    --gradient-audit-interval "$GRAD_AUDIT_INTERVAL"
    --gradient-audit-output "$grad_json"
    --protocol-eval-interval "$PROTOCOL_EVAL_INTERVAL"
    --protocol-eval-manifest "$PROTOCOL_MAIN_MANIFEST"
    --protocol-eval-dataset all
    --protocol-eval-max-clips 0
    --protocol-eval-seed 42
    --protocol-eval-obs-steps 8
    --protocol-eval-pred-steps 8
    --protocol-eval-run-name protocol_val_main
    --protocol-version v2_4_detached_frozen
    --protocol-best-checkpoint-name best_protocol_main.pt
    --protocol-best-selection-name best_protocol_main_selection.json
  )

  if [[ "$warmup_flag" == "1" ]]; then
    cmd+=(--semantic-warmup --semantic-warmup-start-ratio 0.10 --semantic-warmup-end-ratio 0.30)
  fi
  if [[ "$extra_flag" == "disable_semantics" ]]; then
    cmd+=(--disable-semantics)
  elif [[ "$extra_flag" == "neutralize_object_bias" ]]; then
    cmd+=(--neutralize-object-bias)
  fi

  local launch=(
    bash "$GPU_CLAIM_SCRIPT"
    --prefer-gpus 1
    --min-gpus 1
    --poll-seconds "$POLL_SECONDS"
    --max-mem-used-mib "$MAX_MEM_USED_MIB"
    --max-utilization "$MAX_UTIL"
    --candidate-gpus "$CANDIDATE_GPUS"
    --timeout-seconds 0
    --
    "${cmd[@]}"
  )

  nohup "${launch[@]}" > "$log_file" 2>&1 &
  local pid=$!

  echo -e "${run_name}\t${pid}\t${log_file}\t${out_dir}\t${grad_json}" >> "$SUBMIT_TSV"
  echo "[submitted] run=${run_name} pid=${pid} log=${log_file}"
}

# 1) full, fixed λ, no warm-up
submit_run "full_v4_2_seed42_fixed_nowarm_lambda1" "$LSEM_10" "0" "none"

# 2) full, fixed λ, semantic warm-up
submit_run "full_v4_2_seed42_fixed_warmup_lambda1" "$LSEM_10" "1" "none"

# 3) wo_semantics
submit_run "wo_semantics_v4_2_seed42" "$LSEM_10" "0" "disable_semantics"

# 4) wo_object_bias
submit_run "wo_object_bias_v4_2_seed42" "$LSEM_10" "0" "neutralize_object_bias"

# 5-7) λ_sem sweep for full run (0.1, 0.25, 0.5)
submit_run "full_v4_2_seed42_lsem_0p1_lambda0" "$LSEM_01" "0" "none"
submit_run "full_v4_2_seed42_lsem_0p25_lambda0" "$LSEM_025" "0" "none"
submit_run "full_v4_2_seed42_lsem_0p5_lambda0" "$LSEM_05" "0" "none"

# 8) λ_sem=1.0*λ0 is represented by run #1 (same config, no duplicated launch)

echo "[done] submission_file=$SUBMIT_TSV"
