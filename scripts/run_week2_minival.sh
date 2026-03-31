#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

STEPS="${STWM_WEEK2_STEPS:-60}"
EVAL_INTERVAL="${STWM_WEEK2_EVAL_INTERVAL:-20}"
SAVE_INTERVAL="${STWM_WEEK2_SAVE_INTERVAL:-30}"
SEED="${STWM_WEEK2_SEED:-42}"
OBS_STEPS="${STWM_WEEK2_OBS_STEPS:-8}"
PRED_STEPS="${STWM_WEEK2_PRED_STEPS:-8}"
TRAIN_MAX_CLIPS="${STWM_WEEK2_TRAIN_MAX_CLIPS:-32}"
VAL_MAX_CLIPS="${STWM_WEEK2_VAL_MAX_CLIPS:-20}"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/week2_minival}"
mkdir -p "$OUT_ROOT"

run_case() {
  local name="$1"
  shift
  local out_dir="$OUT_ROOT/$name"
  local log_file="$STWM_ROOT/logs/week2_minival_${name}.log"

  echo "[week2-minival] start ${name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
    python "$STWM_ROOT/code/stwm/trainers/train_week2_minival.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$STWM_ROOT/manifests/minisplits/stwm_week1_mini.json" \
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
  echo "[week2-minival] done ${name}"
}

run_case full
run_case wo_semantics --disable-semantics
run_case wo_trajectory --disable-trajectory
run_case wo_identity_memory --disable-identity-memory

echo "[week2-minival] all runs done: $OUT_ROOT"
