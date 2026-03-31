#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_smoke}"
STEPS="${STWM_V4_2_SMOKE_STEPS:-24}"
SAMPLE_LIMIT="${STWM_V4_2_SMOKE_SAMPLE_LIMIT:-18}"
SEED="${STWM_V4_2_SMOKE_SEED:-42}"

MANIFEST="${STWM_V4_2_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"

mkdir -p "$OUT_ROOT"

run_case() {
  local run_name="$1"
  shift
  local out_dir="$OUT_ROOT/$run_name"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_smoke_${run_name}.log"

  echo "[stwm-v4.2-smoke] start run=${run_name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
    python "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$MANIFEST" \
      --output-dir "$out_dir" \
      --run-name "$run_name" \
      --model-preset prototype_220m_v4_2 \
      --preset-file "$STWM_ROOT/code/stwm/configs/model_presets_v4_2.json" \
      --steps "$STEPS" \
      --sample-limit "$SAMPLE_LIMIT" \
      --seed "$SEED" \
      --use-teacher-priors \
      "$@" \
      >"$log_file" 2>&1
  echo "[stwm-v4.2-smoke] done run=${run_name}"
}

run_case full_v4_2
run_case wo_semantics_v4_2 --disable-semantics
run_case wo_identity_v4_2 --disable-identity-memory

echo "[stwm-v4.2-smoke] all runs done: $OUT_ROOT"
