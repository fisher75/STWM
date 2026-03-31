#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_minival_multiseed}"
STEPS="${STWM_V4_2_MINIVAL_MULTI_STEPS:-${STWM_V4_2_MINIVAL_STEPS:-120}}"
SAMPLE_LIMIT="${STWM_V4_2_MINIVAL_MULTI_SAMPLE_LIMIT:-${STWM_V4_2_MINIVAL_SAMPLE_LIMIT:-18}}"
SEEDS_CSV="${STWM_V4_2_MINIVAL_MULTI_SEEDS:-42,123,456}"

MANIFEST="${STWM_V4_2_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"

mkdir -p "$OUT_ROOT"

run_case() {
  local seed="$1"
  local run_name="$2"
  shift 2
  local out_dir="$OUT_ROOT/seed_${seed}/$run_name"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_minival_multiseed_seed${seed}_${run_name}.log"

  echo "[stwm-v4.2-multiseed] start seed=${seed} run=${run_name}"
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
      --seed "$seed" \
      --use-teacher-priors \
      --summary-name mini_val_summary.json \
      --save-checkpoint \
      "$@" \
      >"$log_file" 2>&1
  echo "[stwm-v4.2-multiseed] done seed=${seed} run=${run_name}"
}

IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"
for seed in "${seed_list[@]}"; do
  run_case "$seed" full_v4_2
  run_case "$seed" wo_semantics_v4_2 --disable-semantics
  run_case "$seed" wo_identity_v4_2 --disable-identity-memory
done

echo "[stwm-v4.2-multiseed] all runs done: $OUT_ROOT"
