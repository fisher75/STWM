#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_state_identifiability}"
EVAL_STEPS="${STWM_V4_2_IDENTIFIABILITY_EVAL_STEPS:-60}"
SAMPLE_LIMIT="${STWM_V4_2_IDENTIFIABILITY_SAMPLE_LIMIT:-18}"
SEEDS_CSV="${STWM_V4_2_IDENTIFIABILITY_SEEDS:-42,123}"

SOURCE_MANIFEST="${STWM_V4_2_IDENTIFIABILITY_SOURCE_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"
EVENTFUL_REPORT="${STWM_V4_2_IDENTIFIABILITY_EVENTFUL_REPORT:-$STWM_ROOT/reports/stwm_v4_2_eventful_protocol_v1.json}"
HARD_QUERY_REPORT="${STWM_V4_2_IDENTIFIABILITY_HARD_QUERY_REPORT:-$STWM_ROOT/reports/stwm_v4_2_hard_query_protocol_v1.json}"

PROTOCOL_MANIFEST="${STWM_V4_2_IDENTIFIABILITY_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_v4_2_state_identifiability_v1.json}"
PROTOCOL_CLIP_IDS="${STWM_V4_2_IDENTIFIABILITY_CLIP_IDS:-$STWM_ROOT/manifests/minisplits/stwm_v4_2_state_identifiability_clip_ids_v1.json}"
PROTOCOL_REPORT="${STWM_V4_2_IDENTIFIABILITY_REPORT:-$STWM_ROOT/reports/stwm_v4_2_state_identifiability_protocol_v1.json}"

SUMMARY_JSON="$OUT_ROOT/comparison_state_identifiability.json"
SUMMARY_MD="$OUT_ROOT/comparison_state_identifiability.md"
DECOUPLING_JSON="$STWM_ROOT/reports/stwm_v4_2_state_identifiability_decoupling_v1.json"
FIGURE_OUT="${STWM_V4_2_IDENTIFIABILITY_FIGURE_OUT:-$STWM_ROOT/outputs/visualizations/stwm_v4_2_state_identifiability_figures}"

mkdir -p "$OUT_ROOT"

PYTHON_BIN=(/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm python)

echo "[state-identifiability] build protocol manifest"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/build_stwm_v4_2_state_identifiability_protocol.py" \
    --source-manifest "$SOURCE_MANIFEST" \
    --eventful-report "$EVENTFUL_REPORT" \
    --hard-query-report "$HARD_QUERY_REPORT" \
    --target-size "$SAMPLE_LIMIT" \
    --output-manifest "$PROTOCOL_MANIFEST" \
    --output-clip-ids "$PROTOCOL_CLIP_IDS" \
    --output-report "$PROTOCOL_REPORT"

run_eval_case() {
  local seed="$1"
  local run_name="$2"
  local resume_checkpoint="$3"
  shift 3

  local out_dir="$OUT_ROOT/seed_${seed}/${run_name}"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_state_identifiability_seed${seed}_${run_name}.log"

  if [[ ! -f "$resume_checkpoint" ]]; then
    echo "[state-identifiability] missing checkpoint: $resume_checkpoint" >&2
    exit 2
  fi

  echo "[state-identifiability] eval seed=${seed} run=${run_name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$PROTOCOL_MANIFEST" \
      --output-dir "$out_dir" \
      --run-name "$run_name" \
      --model-preset prototype_220m_v4_2 \
      --preset-file "$STWM_ROOT/code/stwm/configs/model_presets_v4_2.json" \
      --steps "$EVAL_STEPS" \
      --sample-limit "$SAMPLE_LIMIT" \
      --seed "$seed" \
      --use-teacher-priors \
      --resume-checkpoint "$resume_checkpoint" \
      --eval-only \
      --summary-name mini_val_summary.json \
      --log-name train_log.jsonl \
      "$@" \
      >"$log_file" 2>&1
}

IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"
for seed in "${seed_list[@]}"; do
  full_ckpt="$STWM_ROOT/outputs/training/stwm_v4_2_minival_multiseed/seed_${seed}/full_v4_2/final_model.pt"
  ws_ckpt="$STWM_ROOT/outputs/training/stwm_v4_2_minival_multiseed/seed_${seed}/wo_semantics_v4_2/final_model.pt"

  run_eval_case "$seed" full_v4_2 "$full_ckpt"
  run_eval_case "$seed" wo_semantics_v4_2 "$ws_ckpt" --disable-semantics
  run_eval_case "$seed" wo_object_bias_v4_2 "$full_ckpt" --neutralize-object-bias
done

echo "[state-identifiability] summarize protocol results"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/summarize_stwm_v4_2_state_identifiability.py" \
    --runs-root "$OUT_ROOT" \
    --manifest "$PROTOCOL_MANIFEST" \
    --seeds "$SEEDS_CSV" \
    --runs full_v4_2,wo_semantics_v4_2,wo_object_bias_v4_2 \
    --summary-name mini_val_summary.json \
    --log-name train_log.jsonl \
    --output-json "$SUMMARY_JSON" \
    --output-md "$SUMMARY_MD"

echo "[state-identifiability] run hard-protocol decoupling"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/query_trajectory_decoupling_multiseed.py" \
    --runs-root "$OUT_ROOT" \
    --seeds "$SEEDS_CSV" \
    --runs full_v4_2,wo_semantics_v4_2,wo_object_bias_v4_2 \
    --output-json "$DECOUPLING_JSON"

echo "[state-identifiability] build figure casebook"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/evaluators/build_stwm_v4_2_state_identifiability_figures.py" \
    --runs-root "$OUT_ROOT" \
    --seeds "$SEEDS_CSV" \
    --manifest "$PROTOCOL_MANIFEST" \
    --data-root "$STWM_ROOT/data/external" \
    --output-dir "$FIGURE_OUT" \
    --cases-per-group 8 \
    --min-consistent-seeds 2

echo "[state-identifiability] done"
echo "  protocol_manifest: $PROTOCOL_MANIFEST"
echo "  protocol_report:   $PROTOCOL_REPORT"
echo "  summary_json:      $SUMMARY_JSON"
echo "  summary_md:        $SUMMARY_MD"
echo "  decoupling_json:   $DECOUPLING_JSON"
echo "  figure_manifest:   $FIGURE_OUT/figure_manifest.json"
