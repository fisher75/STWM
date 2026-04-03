#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
QUEUE_ROOT="${STWM_PROTOCOL_V2_QUEUE_ROOT:-$REPO_ROOT/outputs/queue/stwm_protocol_v2}"
QUEUE_DIR="$QUEUE_ROOT/d1_train"

TRAIN_SCRIPT="$REPO_ROOT/code/stwm/trainers/train_stwm_v4_2_real.py"
TRAIN_MANIFEST="${STWM_D1_TRAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/train_v2.json}"
PROTOCOL_MAIN_MANIFEST="${STWM_D1_PROTOCOL_MAIN_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_main_v1.json}"
PROTOCOL_EVENTFUL_MANIFEST="${STWM_D1_PROTOCOL_EVENTFUL_MANIFEST:-$REPO_ROOT/manifests/protocol_v2/protocol_val_eventful_v1.json}"
MODEL_PRESET="${STWM_D1_MODEL_PRESET:-prototype_220m_v4_2}"
PRESET_FILE="${STWM_D1_PRESET_FILE:-$REPO_ROOT/code/stwm/configs/model_presets_v4_2.json}"
DATA_ROOT="${STWM_D1_DATA_ROOT:-$REPO_ROOT/data/external}"

DATA_MODE="${STWM_D1_DATA_MODE:-frontend_cache}"
FRONTEND_CACHE_DIR="${STWM_D1_FRONTEND_CACHE_DIR:-$REPO_ROOT/data/cache/frontend_cache_protocol_v2_full_v1}"
FRONTEND_CACHE_INDEX="${STWM_D1_FRONTEND_CACHE_INDEX:-$FRONTEND_CACHE_DIR/index.json}"
FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY="${STWM_D1_FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY:-8}"

SEED="${STWM_D1_SEED:-42}"
STEPS="${STWM_D1_STEPS:-2000}"
SAMPLE_LIMIT="${STWM_D1_SAMPLE_LIMIT:-0}"

LSEM_10="${STWM_D1_LSEM_10:-0.5}"

GRAD_AUDIT_INTERVAL="${STWM_D1_GRAD_AUDIT_INTERVAL:-100}"
PROTOCOL_EVAL_INTERVAL="${STWM_D1_PROTOCOL_EVAL_INTERVAL:-500}"
CHECKPOINT_INTERVAL="${STWM_D1_CHECKPOINT_INTERVAL:-500}"
TASK1_PREFERRED_GPU="${STWM_D1_TASK1_PREFERRED_GPU:-3}"
TASK2_PREFERRED_GPU="${STWM_D1_TASK2_PREFERRED_GPU:-7}"
GRAD_AUDIT_REPORT_DIR="${STWM_D1_GRAD_AUDIT_REPORT_DIR:-$REPO_ROOT/reports}"
GRAD_AUDIT_REPORT_TAG="${STWM_D1_GRAD_AUDIT_REPORT_TAG:-}"

OUT_ROOT="${STWM_D1_OUT_ROOT:-$REPO_ROOT/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_default_v1}"
mkdir -p "$OUT_ROOT"
mkdir -p "$GRAD_AUDIT_REPORT_DIR"

submit_one() {
  local run_name="$1"
  local lambda_sem="$2"
  local warmup="$3"
  local extra_flag="$4"
  local enable_grad_audit="$5"
  local preferred_gpu="${6:-}"

  local out_dir="$OUT_ROOT/seed_${SEED}/${run_name}"
  local grad_tag=""
  if [[ -n "$GRAD_AUDIT_REPORT_TAG" ]]; then
    grad_tag="_${GRAD_AUDIT_REPORT_TAG}"
  fi
  local grad_json="$GRAD_AUDIT_REPORT_DIR/stwm_v4_2_gradient_audit_220m_seed${SEED}_${run_name}${grad_tag}.json"
  local notes="D1 Class-B one-train-one-gpu protocol diagnostic | data_mode=${DATA_MODE}"
  local resume_hint="Resume with same output_dir and --auto-resume; official best from best_protocol_main.pt"

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
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
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
    --protocol-eval-interval "$PROTOCOL_EVAL_INTERVAL"
    --protocol-eval-manifest "$PROTOCOL_MAIN_MANIFEST"
    --protocol-eval-dataset all
    --protocol-eval-max-clips 0
    --protocol-eval-seed 42
    --protocol-eval-obs-steps 8
    --protocol-eval-pred-steps 8
    --protocol-eval-run-name protocol_val_main
    --protocol-diagnostics-manifest "$PROTOCOL_EVENTFUL_MANIFEST"
    --protocol-diagnostics-dataset all
    --protocol-diagnostics-max-clips 0
    --protocol-diagnostics-run-name protocol_val_eventful
    --protocol-version v2_4_detached_frozen
    --protocol-best-checkpoint-name best_protocol_main.pt
    --protocol-best-selection-name best_protocol_main_selection.json
  )

  if [[ "$DATA_MODE" == "frontend_cache" ]]; then
    cmd+=(
      --data-mode frontend_cache
      --frontend-cache-dir "$FRONTEND_CACHE_DIR"
      --frontend-cache-index "$FRONTEND_CACHE_INDEX"
      --frontend-cache-max-shards-in-memory "$FRONTEND_CACHE_MAX_SHARDS_IN_MEMORY"
    )
  elif [[ "$DATA_MODE" == "raw" ]]; then
    cmd+=(--data-mode raw)
  else
    echo "unsupported STWM_D1_DATA_MODE=$DATA_MODE (expected: frontend_cache|raw)" >&2
    exit 2
  fi

  if [[ "$enable_grad_audit" == "1" ]]; then
    cmd+=(
      --gradient-audit-interval "$GRAD_AUDIT_INTERVAL"
      --gradient-audit-output "$grad_json"
    )
  else
    cmd+=(--gradient-audit-interval 0)
  fi

  if [[ "$warmup" == "1" ]]; then
    cmd+=(--semantic-warmup --semantic-warmup-start-ratio 0.10 --semantic-warmup-end-ratio 0.30)
  fi
  if [[ "$extra_flag" == "disable_semantics" ]]; then
    cmd+=(--disable-semantics)
  elif [[ "$extra_flag" == "neutralize_object_bias" ]]; then
    cmd+=(--neutralize-object-bias)
  fi

  local submit_args=(
    --queue-dir "$QUEUE_DIR"
    --job-name "$run_name"
    --class-type B
    --workdir "$REPO_ROOT"
    --notes "$notes"
    --resume-hint "$resume_hint"
  )
  if [[ -n "$preferred_gpu" ]]; then
    submit_args+=(--preferred-gpu "$preferred_gpu")
  fi

  bash "$REPO_ROOT/scripts/protocol_v2_queue_submit.sh" \
    "${submit_args[@]}" \
    -- "${cmd[@]}"
}

# Core 4 runs only in first launch wave.
submit_one "full_v4_2_seed42_fixed_nowarm_lambda1" "$LSEM_10" "0" "none" "1" "$TASK1_PREFERRED_GPU"
submit_one "full_v4_2_seed42_fixed_warmup_lambda1" "$LSEM_10" "1" "none" "1" "$TASK2_PREFERRED_GPU"
submit_one "wo_semantics_v4_2_seed42" "$LSEM_10" "0" "disable_semantics" "0"
submit_one "wo_object_bias_v4_2_seed42" "$LSEM_10" "0" "neutralize_object_bias" "0"

echo "[d1-enqueue] done queue_dir=$QUEUE_DIR"
