#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_smoke}"
STEPS="${STWM_V4_2_1B_SMOKE_STEPS:-8}"
SAMPLE_LIMIT="${STWM_V4_2_1B_SMOKE_SAMPLE_LIMIT:-8}"
SEED="${STWM_V4_2_1B_SMOKE_SEED:-42}"
RUN_TRIO="${STWM_V4_2_1B_SMOKE_RUN_TRIO:-1}"

MANIFEST="${STWM_V4_2_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"
MODEL_PRESET="${STWM_V4_2_1B_PRESET:-prototype_1b_v4_2}"
PRESET_FILE="${STWM_V4_2_1B_PRESET_FILE:-$STWM_ROOT/code/stwm/configs/model_presets_v4_2_1b.json}"

mkdir -p "$OUT_ROOT"

run_case() {
  local run_name="$1"
  shift
  local out_dir="$OUT_ROOT/$run_name"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_1b_smoke_${run_name}.log"

  echo "[stwm-v4.2-1b-smoke] start run=${run_name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
    python "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$MANIFEST" \
      --output-dir "$out_dir" \
      --run-name "$run_name" \
      --model-preset "$MODEL_PRESET" \
      --preset-file "$PRESET_FILE" \
      --steps "$STEPS" \
      --sample-limit "$SAMPLE_LIMIT" \
      --seed "$SEED" \
      --use-teacher-priors \
      --summary-name mini_val_summary.json \
      --log-name train_log.jsonl \
      "$@" \
      >"$log_file" 2>&1
  echo "[stwm-v4.2-1b-smoke] done run=${run_name}"
}

run_case full_v4_2

if [[ "$RUN_TRIO" == "1" ]]; then
  run_case wo_semantics_v4_2 --disable-semantics
  run_case wo_object_bias_v4_2 --neutralize-object-bias
fi

echo "[stwm-v4.2-1b-smoke] all runs done: $OUT_ROOT"
