#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <checkpoint_dir> <candidate_checkpoint> <manifest_json> <model_preset> [preset_file] [data_root]"
  echo "Example: $0 outputs/training/stwm_v4_2_real_220m/seed_42/full_v4_2/checkpoints latest.pt manifests/protocol_v2/protocol_val_main_v1.json prototype_220m_v4_2 code/stwm/configs/model_presets_v4_2.json"
  exit 1
fi

REPO_ROOT="/home/chen034/workspace/stwm"
CHECKPOINT_DIR="$1"
CANDIDATE_CHECKPOINT="$2"
MANIFEST_JSON="$3"
MODEL_PRESET="$4"
PRESET_FILE="${5:-$REPO_ROOT/code/stwm/configs/model_presets_v4_2.json}"
DATA_ROOT="${6:-$REPO_ROOT/data/external}"

CHECKPOINT_DIR_ABS="$CHECKPOINT_DIR"
if [[ "$CHECKPOINT_DIR_ABS" != /* ]]; then
  CHECKPOINT_DIR_ABS="$REPO_ROOT/$CHECKPOINT_DIR_ABS"
fi

CANDIDATE_PATH="$CANDIDATE_CHECKPOINT"
if [[ "$CANDIDATE_PATH" != /* ]]; then
  CANDIDATE_PATH="$CHECKPOINT_DIR_ABS/$CANDIDATE_PATH"
fi

MANIFEST_ABS="$MANIFEST_JSON"
if [[ "$MANIFEST_ABS" != /* ]]; then
  MANIFEST_ABS="$REPO_ROOT/$MANIFEST_ABS"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
EVAL_DIR="$CHECKPOINT_DIR_ABS/protocol_eval"
mkdir -p "$EVAL_DIR"
EVAL_JSON="$EVAL_DIR/protocol_val_main_${STAMP}.json"

PYTHONPATH="$REPO_ROOT/code" conda run --no-capture-output -n stwm \
  python "$REPO_ROOT/code/stwm/evaluators/eval_mini_val.py" \
  --data-root "$DATA_ROOT" \
  --manifest "$MANIFEST_ABS" \
  --dataset all \
  --max-clips 0 \
  --obs-steps 8 \
  --pred-steps 8 \
  --seed 42 \
  --checkpoint "$CANDIDATE_PATH" \
  --model-preset "$MODEL_PRESET" \
  --preset-file "$PRESET_FILE" \
  --protocol-version v2_4_detached_frozen \
  --run-name protocol_val_main \
  --output "$EVAL_JSON"

PYTHONPATH="$REPO_ROOT/code" conda run --no-capture-output -n stwm \
  python "$REPO_ROOT/code/stwm/tools/update_protocol_best_main.py" \
  --checkpoint-dir "$CHECKPOINT_DIR_ABS" \
  --candidate-checkpoint "$CANDIDATE_PATH" \
  --eval-summary "$EVAL_JSON" \
  --output-checkpoint best_protocol_main.pt \
  --selection-sidecar best_protocol_main_selection.json

echo "[protocol-candidate-eval] eval_json=$EVAL_JSON"
