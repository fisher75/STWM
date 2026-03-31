#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_identity_rescue_round}"
CONT_STEPS="${STWM_V4_2_IDENTITY_RESCUE_STEPS:-60}"
EVAL_STEPS="${STWM_V4_2_IDENTITY_RESCUE_EVAL_STEPS:-60}"
SAMPLE_LIMIT="${STWM_V4_2_IDENTITY_RESCUE_SAMPLE_LIMIT:-18}"
SEEDS_CSV="${STWM_V4_2_IDENTITY_RESCUE_SEEDS:-42,123}"
LEARNING_RATE="${STWM_V4_2_IDENTITY_RESCUE_LR:-5e-5}"

MANIFEST_PREFIX="${STWM_V4_2_IDENTITY_RESCUE_PREFIX:-stwm_v4_2_identity_rescue_v1}"
MANIFEST_DIR="$STWM_ROOT/manifests/minisplits"

BASE_MANIFEST="$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json"
EVENTFUL_MANIFEST="$STWM_ROOT/manifests/minisplits/stwm_v4_2_eventful_minival_v1.json"
HARD_QUERY_MANIFEST="$STWM_ROOT/manifests/minisplits/stwm_v4_2_hard_query_minival_v1.json"

mkdir -p "$OUT_ROOT"

PYTHON_BIN=(/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm python)

echo "[identity-rescue] build variant manifests"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/build_stwm_v4_2_identity_rescue_manifests.py" \
    --base-manifest "$BASE_MANIFEST" \
    --base-clip-ids "$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2_val_clip_ids.json" \
    --eventful-manifest "$EVENTFUL_MANIFEST" \
    --hard-query-manifest "$HARD_QUERY_MANIFEST" \
    --target-size "$SAMPLE_LIMIT" \
    --eventful-mix-base 6 \
    --eventful-mix-eventful 12 \
    --ehq-mix-base 2 \
    --ehq-mix-eventful 8 \
    --ehq-mix-hard-query 8 \
    --output-dir "$MANIFEST_DIR" \
    --name-prefix "$MANIFEST_PREFIX"

CONTROL_MANIFEST="$MANIFEST_DIR/${MANIFEST_PREFIX}_control_resume_base.json"
EVENTFUL_MIX_MANIFEST="$MANIFEST_DIR/${MANIFEST_PREFIX}_resume_eventful_mix.json"
EHQ_MIX_MANIFEST="$MANIFEST_DIR/${MANIFEST_PREFIX}_resume_eventful_hardquery_mix.json"

run_train_variant() {
  local variant="$1"
  local manifest_path="$2"
  local seed="$3"
  local run_name="$4"
  local resume_checkpoint="$5"
  shift 5

  local out_dir="$OUT_ROOT/train/${variant}/seed_${seed}/${run_name}"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_identity_rescue_${variant}_seed${seed}_${run_name}.log"

  if [[ ! -f "$resume_checkpoint" ]]; then
    echo "[identity-rescue] missing resume checkpoint: $resume_checkpoint" >&2
    exit 2
  fi

  echo "[identity-rescue] train variant=${variant} seed=${seed} run=${run_name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$manifest_path" \
      --output-dir "$out_dir" \
      --run-name "$run_name" \
      --model-preset prototype_220m_v4_2 \
      --preset-file "$STWM_ROOT/code/stwm/configs/model_presets_v4_2.json" \
      --steps "$CONT_STEPS" \
      --sample-limit "$SAMPLE_LIMIT" \
      --seed "$seed" \
      --learning-rate "$LEARNING_RATE" \
      --use-teacher-priors \
      --resume-checkpoint "$resume_checkpoint" \
      --summary-name rescue_train_summary.json \
      --log-name rescue_train_log.jsonl \
      --save-checkpoint \
      --checkpoint-name rescue_final_model.pt \
      "$@" \
      >"$log_file" 2>&1
}

run_eval_protocol() {
  local protocol="$1"
  local variant="$2"
  local seed="$3"
  local run_name="$4"
  local checkpoint_path="$5"
  local manifest_path="$6"
  shift 6

  local out_dir="$OUT_ROOT/eval/${protocol}/${variant}/seed_${seed}/${run_name}"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_identity_eval_${protocol}_${variant}_seed${seed}_${run_name}.log"

  if [[ ! -f "$checkpoint_path" ]]; then
    echo "[identity-rescue] missing eval checkpoint: $checkpoint_path" >&2
    exit 2
  fi

  echo "[identity-rescue] eval protocol=${protocol} variant=${variant} seed=${seed} run=${run_name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$manifest_path" \
      --output-dir "$out_dir" \
      --run-name "${run_name}_${protocol}_eval" \
      --model-preset prototype_220m_v4_2 \
      --preset-file "$STWM_ROOT/code/stwm/configs/model_presets_v4_2.json" \
      --steps "$EVAL_STEPS" \
      --sample-limit "$SAMPLE_LIMIT" \
      --seed "$seed" \
      --use-teacher-priors \
      --resume-checkpoint "$checkpoint_path" \
      --eval-only \
      --summary-name mini_val_summary.json \
      --log-name train_log.jsonl \
      "$@" \
      >"$log_file" 2>&1
}

summarize_variant_protocol() {
  local variant="$1"
  local protocol="$2"
  local runs_root="$OUT_ROOT/eval/${protocol}/${variant}"

  echo "[identity-rescue] summarize variant=${variant} protocol=${protocol}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/summarize_stwm_v4_2_minival_multiseed.py" \
      --runs-root "$runs_root" \
      --seeds "$SEEDS_CSV" \
      --runs full_v4_2,wo_identity_v4_2 \
      --summary-name mini_val_summary.json \
      --output-json "$runs_root/comparison_${protocol}.json" \
      --output-md "$runs_root/comparison_${protocol}.md"
}

posthoc_variant() {
  local variant="$1"
  local eventful_root="$OUT_ROOT/eval/eventful/${variant}"
  local hard_root="$OUT_ROOT/eval/hard_query/${variant}"

  echo "[identity-rescue] posthoc variant=${variant}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/occlusion_reconnect_bucket_multiseed.py" \
      --runs-root "$eventful_root" \
      --seeds "$SEEDS_CSV" \
      --runs full_v4_2,wo_identity_v4_2 \
      --output-json "$eventful_root/occlusion_reconnect_eventful.json"

  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/query_trajectory_decoupling_multiseed.py" \
      --runs-root "$hard_root" \
      --seeds "$SEEDS_CSV" \
      --runs full_v4_2,wo_identity_v4_2 \
      --output-json "$hard_root/query_decoupling_hard_query.json"
}

variants=(
  "control_resume_base|$CONTROL_MANIFEST"
  "resume_eventful_mix|$EVENTFUL_MIX_MANIFEST"
  "resume_eventful_hardquery_mix|$EHQ_MIX_MANIFEST"
)

IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"
for pair in "${variants[@]}"; do
  IFS='|' read -r variant manifest_path <<< "$pair"
  for seed in "${seed_list[@]}"; do
    run_train_variant "$variant" "$manifest_path" "$seed" full_v4_2 \
      "$STWM_ROOT/outputs/training/stwm_v4_2_minival_multiseed/seed_${seed}/full_v4_2/final_model.pt"
    run_train_variant "$variant" "$manifest_path" "$seed" wo_identity_v4_2 \
      "$STWM_ROOT/outputs/training/stwm_v4_2_minival_multiseed/seed_${seed}/wo_identity_v4_2/final_model.pt" \
      --disable-identity-memory

    full_ckpt="$OUT_ROOT/train/${variant}/seed_${seed}/full_v4_2/rescue_final_model.pt"
    wi_ckpt="$OUT_ROOT/train/${variant}/seed_${seed}/wo_identity_v4_2/rescue_final_model.pt"

    run_eval_protocol base "$variant" "$seed" full_v4_2 "$full_ckpt" "$BASE_MANIFEST"
    run_eval_protocol base "$variant" "$seed" wo_identity_v4_2 "$wi_ckpt" "$BASE_MANIFEST" --disable-identity-memory

    run_eval_protocol eventful "$variant" "$seed" full_v4_2 "$full_ckpt" "$EVENTFUL_MANIFEST"
    run_eval_protocol eventful "$variant" "$seed" wo_identity_v4_2 "$wi_ckpt" "$EVENTFUL_MANIFEST" --disable-identity-memory

    run_eval_protocol hard_query "$variant" "$seed" full_v4_2 "$full_ckpt" "$HARD_QUERY_MANIFEST"
    run_eval_protocol hard_query "$variant" "$seed" wo_identity_v4_2 "$wi_ckpt" "$HARD_QUERY_MANIFEST" --disable-identity-memory
  done

  summarize_variant_protocol "$variant" base
  summarize_variant_protocol "$variant" eventful
  summarize_variant_protocol "$variant" hard_query
  posthoc_variant "$variant"
done

echo "[identity-rescue] all runs and summaries done: $OUT_ROOT"
