#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

PRESET="${1:-prototype_220m}"
OUT_PATH="${2:-$STWM_ROOT/outputs/training/${PRESET}_minimal_train_step.json}"
MANIFEST_PATH="${STWM_MANIFEST_PATH:-$STWM_ROOT/manifests/minisplits/stwm_week1_mini.json}"

PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
  python "$STWM_ROOT/code/stwm/trainers/train_stwm.py" \
    --data-root "$STWM_ROOT/data/external" \
    --manifest "$MANIFEST_PATH" \
    --limit 1 \
    --model-preset "$PRESET" \
    --preset-file "$STWM_ROOT/code/stwm/configs/model_presets.json" \
    --output "$OUT_PATH"

echo "Week-2 prototype smoke report: $OUT_PATH"
