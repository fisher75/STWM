#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

MANIFEST_PATH="${STWM_MANIFEST_PATH:-$STWM_ROOT/manifests/minisplits/stwm_week1_mini.json}"
OUT_DIR="${1:-$STWM_ROOT/outputs/training/week2_ablations}"
mkdir -p "$OUT_DIR"

run_case() {
  local name="$1"
  shift
  local out_json="$OUT_DIR/${name}.json"
  local log_file="$STWM_ROOT/logs/week2_${name}.log"

  echo "[week2-ablation] running ${name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
    python "$STWM_ROOT/code/stwm/trainers/train_stwm.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$MANIFEST_PATH" \
      --limit 1 \
      --model-preset prototype_220m \
      --preset-file "$STWM_ROOT/code/stwm/configs/model_presets.json" \
      --output "$out_json" \
      "$@" \
      >"$log_file" 2>&1

  echo "[week2-ablation] done ${name}: ${out_json}"
}

run_case full
run_case wo_semantics --disable-semantics
run_case wo_trajectory --disable-trajectory
run_case wo_identity_memory --disable-identity-memory

echo "Week-2 ablation outputs: $OUT_DIR"
