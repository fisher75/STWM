#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_protocol_repair}"
STEPS="${STWM_V4_2_PROTOCOL_REPAIR_STEPS:-120}"
SAMPLE_LIMIT="${STWM_V4_2_PROTOCOL_REPAIR_SAMPLE_LIMIT:-18}"
SEEDS_CSV="${STWM_V4_2_PROTOCOL_REPAIR_SEEDS:-42,123}"

EVENTFUL_MANIFEST="${STWM_V4_2_EVENTFUL_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_v4_2_eventful_minival_v1.json}"
HARD_QUERY_MANIFEST="${STWM_V4_2_HARD_QUERY_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_v4_2_hard_query_minival_v1.json}"
INCLUDE_WO_SEMANTICS="${STWM_V4_2_PROTOCOL_REPAIR_INCLUDE_WO_SEMANTICS:-0}"

mkdir -p "$OUT_ROOT"

run_case() {
  local protocol_name="$1"
  local manifest_path="$2"
  local seed="$3"
  local run_name="$4"
  shift 4
  local out_dir="$OUT_ROOT/${protocol_name}/seed_${seed}/${run_name}"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_protocol_repair_${protocol_name}_seed${seed}_${run_name}.log"

  echo "[stwm-v4.2-protocol-repair] start protocol=${protocol_name} seed=${seed} run=${run_name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    /home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm \
    python "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$manifest_path" \
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
  echo "[stwm-v4.2-protocol-repair] done protocol=${protocol_name} seed=${seed} run=${run_name}"
}

run_protocol() {
  local protocol_name="$1"
  local manifest_path="$2"
  if [[ ! -f "$manifest_path" ]]; then
    echo "[stwm-v4.2-protocol-repair] missing manifest: $manifest_path" >&2
    exit 2
  fi

  IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"
  for seed in "${seed_list[@]}"; do
    run_case "$protocol_name" "$manifest_path" "$seed" full_v4_2
    run_case "$protocol_name" "$manifest_path" "$seed" wo_identity_v4_2 --disable-identity-memory
    if [[ "$INCLUDE_WO_SEMANTICS" == "1" ]]; then
      run_case "$protocol_name" "$manifest_path" "$seed" wo_semantics_v4_2 --disable-semantics
    fi
  done
}

run_protocol "eventful" "$EVENTFUL_MANIFEST"
run_protocol "hard_query" "$HARD_QUERY_MANIFEST"

echo "[stwm-v4.2-protocol-repair] all runs done: $OUT_ROOT"
