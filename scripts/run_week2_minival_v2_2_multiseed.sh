#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

STEPS="${STWM_WEEK2_V2_2_STEPS:-80}"
EVAL_INTERVAL="${STWM_WEEK2_V2_2_EVAL_INTERVAL:-20}"
SAVE_INTERVAL="${STWM_WEEK2_V2_2_SAVE_INTERVAL:-20}"
OBS_STEPS="${STWM_WEEK2_V2_2_OBS_STEPS:-8}"
PRED_STEPS="${STWM_WEEK2_V2_2_PRED_STEPS:-8}"
TRAIN_MAX_CLIPS="${STWM_WEEK2_V2_2_TRAIN_MAX_CLIPS:-32}"
VAL_MAX_CLIPS="${STWM_WEEK2_V2_2_VAL_MAX_CLIPS:-18}"
SEEDS_CSV="${STWM_WEEK2_V2_2_SEEDS:-42,123,456}"
QUERY_CANDIDATES="${STWM_WEEK2_V2_2_QUERY_CANDIDATES:-7}"
QUERY_HIT_RADIUS="${STWM_WEEK2_V2_2_QUERY_HIT_RADIUS:-0.08}"
QUERY_TOPK="${STWM_WEEK2_V2_2_QUERY_TOPK:-1}"
IDENTITY_HIT_RADIUS="${STWM_WEEK2_V2_2_IDENTITY_HIT_RADIUS:-0.04}"
OCCLUSION_RECOVERY_WINDOW="${STWM_WEEK2_V2_2_OCCLUSION_RECOVERY_WINDOW:-3}"
QUERY_HARD_NEG_JITTER="${STWM_WEEK2_V2_2_QUERY_HARD_NEG_JITTER:-0.03}"
IDENTITY_TARGET_OVERLAP_MIN="${STWM_WEEK2_V2_2_IDENTITY_TARGET_OVERLAP_MIN:-0.02}"
IDENTITY_OTHER_OVERLAP_MIN="${STWM_WEEK2_V2_2_IDENTITY_OTHER_OVERLAP_MIN:-0.15}"
OCCLUSION_MIN_DISAPPEAR_FRAMES="${STWM_WEEK2_V2_2_OCCLUSION_MIN_DISAPPEAR_FRAMES:-2}"
QUERY_NEAR_NEGATIVE_COUNT="${STWM_WEEK2_V2_2_QUERY_NEAR_NEGATIVE_COUNT:-2}"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/week2_minival_v2_2}"
mkdir -p "$OUT_ROOT"

V2_MANIFEST="${STWM_WEEK2_V2_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"
V2_VAL_IDS="${STWM_WEEK2_V2_VAL_IDS:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2_val_clip_ids.json}"

if [[ ! -f "$V2_MANIFEST" || ! -f "$V2_VAL_IDS" ]]; then
  echo "[week2-minival-v2.2] missing v2 manifest or val ids; build v2 protocol first" >&2
  exit 2
fi

run_case() {
  local seed="$1"
  local name="$2"
  shift 2
  local out_dir="$OUT_ROOT/seed_${seed}/${name}"
  local log_file="$STWM_ROOT/logs/week2_minival_v2_2_seed${seed}_${name}.log"

  echo "[week2-minival-v2.2] start seed=${seed} run=${name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
    python "$STWM_ROOT/code/stwm/trainers/train_week2_minival.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$V2_MANIFEST" \
      --val-clip-ids-path "$V2_VAL_IDS" \
      --protocol-version v2_2 \
      --query-candidates "$QUERY_CANDIDATES" \
      --query-hit-radius "$QUERY_HIT_RADIUS" \
      --query-topk "$QUERY_TOPK" \
      --identity-hit-radius "$IDENTITY_HIT_RADIUS" \
      --occlusion-recovery-window "$OCCLUSION_RECOVERY_WINDOW" \
      --query-hard-negative-jitter "$QUERY_HARD_NEG_JITTER" \
      --identity-target-overlap-min "$IDENTITY_TARGET_OVERLAP_MIN" \
      --identity-other-overlap-min "$IDENTITY_OTHER_OVERLAP_MIN" \
      --occlusion-min-disappear-frames "$OCCLUSION_MIN_DISAPPEAR_FRAMES" \
      --query-near-negative-count "$QUERY_NEAR_NEGATIVE_COUNT" \
      --run-name "$name" \
      --output-dir "$out_dir" \
      --seed "$seed" \
      --steps "$STEPS" \
      --eval-interval "$EVAL_INTERVAL" \
      --save-interval "$SAVE_INTERVAL" \
      --obs-steps "$OBS_STEPS" \
      --pred-steps "$PRED_STEPS" \
      --train-max-clips "$TRAIN_MAX_CLIPS" \
      --val-max-clips "$VAL_MAX_CLIPS" \
      --model-preset prototype_220m \
      --preset-file "$STWM_ROOT/code/stwm/configs/model_presets.json" \
      "$@" \
      >"$log_file" 2>&1
  echo "[week2-minival-v2.2] done seed=${seed} run=${name}"
}

IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"
for seed in "${seed_list[@]}"; do
  run_case "$seed" full
  run_case "$seed" wo_semantics --disable-semantics
  run_case "$seed" wo_identity_memory --disable-identity-memory
done

echo "[week2-minival-v2.2] all runs done: $OUT_ROOT"
