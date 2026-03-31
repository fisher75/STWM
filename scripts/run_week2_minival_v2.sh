#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

STEPS="${STWM_WEEK2_V2_STEPS:-80}"
EVAL_INTERVAL="${STWM_WEEK2_V2_EVAL_INTERVAL:-20}"
SAVE_INTERVAL="${STWM_WEEK2_V2_SAVE_INTERVAL:-20}"
SEED="${STWM_WEEK2_V2_SEED:-42}"
OBS_STEPS="${STWM_WEEK2_V2_OBS_STEPS:-8}"
PRED_STEPS="${STWM_WEEK2_V2_PRED_STEPS:-8}"
TRAIN_MAX_CLIPS="${STWM_WEEK2_V2_TRAIN_MAX_CLIPS:-32}"
VAL_MAX_CLIPS="${STWM_WEEK2_V2_VAL_MAX_CLIPS:-18}"
QUERY_CANDIDATES="${STWM_WEEK2_V2_QUERY_CANDIDATES:-5}"
QUERY_HIT_RADIUS="${STWM_WEEK2_V2_QUERY_HIT_RADIUS:-0.08}"
QUERY_TOPK="${STWM_WEEK2_V2_QUERY_TOPK:-1}"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/week2_minival_v2}"
mkdir -p "$OUT_ROOT"

V2_MANIFEST="${STWM_WEEK2_V2_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"
V2_VAL_IDS="${STWM_WEEK2_V2_VAL_IDS:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2_val_clip_ids.json}"

PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
  python "$STWM_ROOT/code/stwm/tools/build_minival_v2_protocol.py" \
  --manifest "$STWM_ROOT/manifests/minisplits/stwm_week1_mini.json" \
  --output-manifest "$V2_MANIFEST" \
  --output-val-ids "$V2_VAL_IDS" \
  --obs-steps "$OBS_STEPS" \
  --pred-steps "$PRED_STEPS" \
  --val-clips "$VAL_MAX_CLIPS" \
  >"$STWM_ROOT/logs/week2_minival_v2_protocol_build.log" 2>&1

run_case() {
  local name="$1"
  shift
  local out_dir="$OUT_ROOT/$name"
  local log_file="$STWM_ROOT/logs/week2_minival_v2_${name}.log"

  echo "[week2-minival-v2] start ${name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
    python "$STWM_ROOT/code/stwm/trainers/train_week2_minival.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$V2_MANIFEST" \
      --val-clip-ids-path "$V2_VAL_IDS" \
      --protocol-version v2 \
      --query-candidates "$QUERY_CANDIDATES" \
      --query-hit-radius "$QUERY_HIT_RADIUS" \
      --query-topk "$QUERY_TOPK" \
      --run-name "$name" \
      --output-dir "$out_dir" \
      --seed "$SEED" \
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
  echo "[week2-minival-v2] done ${name}"
}

run_case full
run_case wo_semantics --disable-semantics
run_case wo_trajectory --disable-trajectory
run_case wo_identity_memory --disable-identity-memory

echo "[week2-minival-v2] all runs done: $OUT_ROOT"
