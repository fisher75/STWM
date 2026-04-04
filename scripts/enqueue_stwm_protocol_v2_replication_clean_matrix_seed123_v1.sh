#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/chen034/workspace/stwm"
QUEUE_ROOT="${STWM_PROTOCOL_V2_QUEUE_ROOT:-$REPO_ROOT/outputs/queue/stwm_protocol_v2_frontend_default_v1}"
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

SEED="${STWM_D1_SEED:-123}"
STEPS="${STWM_D1_STEPS:-2000}"
SAMPLE_LIMIT="${STWM_D1_SAMPLE_LIMIT:-0}"

LSEM_10="${STWM_D1_LSEM_10:-0.5}"

PROTOCOL_EVAL_INTERVAL="${STWM_D1_PROTOCOL_EVAL_INTERVAL:-500}"
CHECKPOINT_INTERVAL="${STWM_D1_CHECKPOINT_INTERVAL:-500}"
PREFERRED_GPU_ALL="${STWM_D1_PREFERRED_GPU_ALL:-}"

OUT_ROOT="${STWM_D1_REPLICATION_OUT_ROOT:-$REPO_ROOT/outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_replication_seed123_v1}"
REPORT_ROOT="${STWM_D1_REPLICATION_REPORT_ROOT:-$REPO_ROOT/reports}"

mkdir -p "$OUT_ROOT"
mkdir -p "$REPORT_ROOT"

STAMP="$(date +%Y%m%d_%H%M%S)"
SUBMIT_TSV="${STWM_D1_REPLICATION_SUBMIT_TSV:-$REPORT_ROOT/stwm_seed123_replication_clean_matrix_submit_v1_${STAMP}.tsv}"
echo -e "run_name\tjob_id\tstatus_file\tmain_log\toutput_dir" > "$SUBMIT_TSV"

submit_one() {
  local run_name="$1"
  local extra_flag="$2"
  local object_bias_alpha="$3"
  local preferred_gpu="${4:-}"

  local out_dir="$OUT_ROOT/seed_${SEED}/${run_name}"
  mkdir -p "$out_dir"

  local notes="Seed123 replication clean matrix v1 | data_mode=${DATA_MODE}"
  local resume_hint="Resume with same output_dir and --auto-resume; compare protocol-best sidecar under same frozen rule"

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
    --lambda-sem "$LSEM_10"
    --lambda-reid 0.25
    --lambda-query 0.25
    --lambda-reconnect 0.1
    --gradient-audit-interval 0
    --protocol-eval-interval "$PROTOCOL_EVAL_INTERVAL"
    --protocol-eval-manifest "$PROTOCOL_MAIN_MANIFEST"
    --protocol-eval-dataset all
    --protocol-eval-max-clips 0
    --protocol-eval-seed "$SEED"
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

  if [[ "$extra_flag" == "disable_semantics" ]]; then
    cmd+=(--disable-semantics)
  elif [[ "$extra_flag" == "neutralize_object_bias" ]]; then
    cmd+=(--neutralize-object-bias)
  fi

  if [[ -n "$object_bias_alpha" ]]; then
    cmd+=(--object-bias-alpha "$object_bias_alpha")
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

  local submit_output
  submit_output="$(bash "$REPO_ROOT/scripts/protocol_v2_queue_submit.sh" "${submit_args[@]}" -- "${cmd[@]}")"
  echo "$submit_output"

  local job_id
  job_id="$(echo "$submit_output" | sed -n 's/^  job_id:[[:space:]]*//p' | tail -n 1)"
  local status_file
  status_file="$(echo "$submit_output" | sed -n 's/^  status_file:[[:space:]]*//p' | tail -n 1)"
  local main_log
  main_log="$(echo "$submit_output" | sed -n 's/^  main_log:[[:space:]]*//p' | tail -n 1)"

  if [[ -z "$status_file" ]]; then
    status_file="$(ls "$QUEUE_DIR/status"/*"_${run_name}".status.json 2>/dev/null | tail -n 1 || true)"
  fi
  if [[ -z "$job_id" && -n "$status_file" && -f "$status_file" ]]; then
    job_id="$(sed -n 's/^[[:space:]]*"job_id":[[:space:]]*"\([^"]*\)".*/\1/p' "$status_file" | head -n 1)"
  fi

  echo -e "${run_name}\t${job_id}\t${status_file}\t${main_log}\t${out_dir}" >> "$SUBMIT_TSV"
}

submit_one "full_v4_2_seed123_fixed_nowarm_lambda1_rerun_v2" "none" "" "$PREFERRED_GPU_ALL"
submit_one "full_v4_2_seed123_objbias_alpha050_replacement_v1" "none" "0.50" "$PREFERRED_GPU_ALL"
submit_one "wo_semantics_v4_2_seed123_control_v1" "disable_semantics" "" "$PREFERRED_GPU_ALL"
submit_one "wo_object_bias_v4_2_seed123_control_v1" "neutralize_object_bias" "" "$PREFERRED_GPU_ALL"

echo "[seed123-replication-enqueue] done queue_dir=$QUEUE_DIR"
echo "[seed123-replication-enqueue] submissions=$SUBMIT_TSV"
