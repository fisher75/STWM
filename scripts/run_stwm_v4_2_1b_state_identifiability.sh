#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_state_identifiability}"
EVAL_STEPS="${STWM_V4_2_1B_IDENTIFIABILITY_EVAL_STEPS:-60}"
SAMPLE_LIMIT="${STWM_V4_2_1B_IDENTIFIABILITY_SAMPLE_LIMIT:-18}"
SEEDS_CSV="${STWM_V4_2_1B_IDENTIFIABILITY_SEEDS:-42,123,456}"
RUNS_CSV="${STWM_V4_2_1B_IDENTIFIABILITY_RUNS:-full_v4_2,wo_semantics_v4_2,wo_object_bias_v4_2}"
SKIP_EXISTING="${STWM_V4_2_1B_IDENTIFIABILITY_SKIP_EXISTING:-0}"

BASE_RUNS_ROOT="${STWM_V4_2_1B_BASE_RUNS_ROOT:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_minival_multiseed}"
SOURCE_MANIFEST="${STWM_V4_2_IDENTIFIABILITY_SOURCE_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"
EVENTFUL_REPORT="${STWM_V4_2_IDENTIFIABILITY_EVENTFUL_REPORT:-$STWM_ROOT/reports/stwm_v4_2_eventful_protocol_v1.json}"
HARD_QUERY_REPORT="${STWM_V4_2_IDENTIFIABILITY_HARD_QUERY_REPORT:-$STWM_ROOT/reports/stwm_v4_2_hard_query_protocol_v1.json}"

PROTOCOL_MANIFEST="${STWM_V4_2_1B_IDENTIFIABILITY_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_v4_2_1b_state_identifiability_v1.json}"
PROTOCOL_CLIP_IDS="${STWM_V4_2_1B_IDENTIFIABILITY_CLIP_IDS:-$STWM_ROOT/manifests/minisplits/stwm_v4_2_1b_state_identifiability_clip_ids_v1.json}"
PROTOCOL_REPORT="${STWM_V4_2_1B_IDENTIFIABILITY_REPORT:-$STWM_ROOT/reports/stwm_v4_2_1b_state_identifiability_protocol_v1.json}"

MODEL_PRESET="${STWM_V4_2_1B_PRESET:-prototype_1b_v4_2}"
PRESET_FILE="${STWM_V4_2_1B_PRESET_FILE:-$STWM_ROOT/code/stwm/configs/model_presets_v4_2_1b.json}"

SUMMARY_JSON="$OUT_ROOT/comparison_state_identifiability.json"
SUMMARY_MD="$OUT_ROOT/comparison_state_identifiability.md"
DECOUPLING_JSON="${STWM_V4_2_1B_IDENTIFIABILITY_DECOUPLING_JSON:-$STWM_ROOT/reports/stwm_v4_2_1b_state_identifiability_decoupling_v1.json}"
FIGURE_OUT="${STWM_V4_2_1B_IDENTIFIABILITY_FIGURE_OUT:-$STWM_ROOT/outputs/visualizations/stwm_v4_2_1b_state_identifiability_figures}"
BUILD_FIGURES="${STWM_V4_2_1B_IDENTIFIABILITY_BUILD_FIGURES:-1}"
FIGURE_MIN_CONSISTENT_SEEDS="${STWM_V4_2_1B_IDENTIFIABILITY_FIGURE_MIN_CONSISTENT_SEEDS:-2}"

mkdir -p "$OUT_ROOT"

PYTHON_BIN=(/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm python)

echo "[1b-state-identifiability] build protocol manifest"
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
  local log_file="$STWM_ROOT/logs/stwm_v4_2_1b_state_identifiability_seed${seed}_${run_name}.log"
  local summary_file="$out_dir/mini_val_summary.json"
  local run_log="$out_dir/train_log.jsonl"

  if [[ "$SKIP_EXISTING" == "1" && -f "$summary_file" && -f "$run_log" ]]; then
    echo "[1b-state-identifiability] skip existing seed=${seed} run=${run_name}"
    return
  fi

  if [[ ! -f "$resume_checkpoint" ]]; then
    echo "[1b-state-identifiability] missing checkpoint: $resume_checkpoint" >&2
    exit 2
  fi

  echo "[1b-state-identifiability] eval seed=${seed} run=${run_name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$PROTOCOL_MANIFEST" \
      --output-dir "$out_dir" \
      --run-name "$run_name" \
      --model-preset "$MODEL_PRESET" \
      --preset-file "$PRESET_FILE" \
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
IFS=',' read -r -a run_list <<< "$RUNS_CSV"
for seed in "${seed_list[@]}"; do
  for run_name in "${run_list[@]}"; do
    case "$run_name" in
      full_v4_2)
        run_eval_case "$seed" "$run_name" "$BASE_RUNS_ROOT/seed_${seed}/full_v4_2/final_model.pt"
        ;;
      wo_semantics_v4_2)
        run_eval_case "$seed" "$run_name" "$BASE_RUNS_ROOT/seed_${seed}/wo_semantics_v4_2/final_model.pt" --disable-semantics
        ;;
      wo_object_bias_v4_2)
        run_eval_case "$seed" "$run_name" "$BASE_RUNS_ROOT/seed_${seed}/wo_object_bias_v4_2/final_model.pt" --neutralize-object-bias
        ;;
      *)
        echo "[1b-state-identifiability] unsupported run: $run_name" >&2
        exit 2
        ;;
    esac
  done
done

echo "[1b-state-identifiability] summarize protocol results"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/summarize_stwm_v4_2_state_identifiability.py" \
    --runs-root "$OUT_ROOT" \
    --manifest "$PROTOCOL_MANIFEST" \
    --seeds "$SEEDS_CSV" \
    --runs "$RUNS_CSV" \
    --summary-name mini_val_summary.json \
    --log-name train_log.jsonl \
    --output-json "$SUMMARY_JSON" \
    --output-md "$SUMMARY_MD"

echo "[1b-state-identifiability] run harder-protocol decoupling"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/query_trajectory_decoupling_multiseed.py" \
    --runs-root "$OUT_ROOT" \
    --seeds "$SEEDS_CSV" \
    --runs "$RUNS_CSV" \
    --output-json "$DECOUPLING_JSON"

if [[ "$BUILD_FIGURES" == "1" ]]; then
  echo "[1b-state-identifiability] build figure casebook"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/evaluators/build_stwm_v4_2_state_identifiability_figures.py" \
      --runs-root "$OUT_ROOT" \
      --seeds "$SEEDS_CSV" \
      --manifest "$PROTOCOL_MANIFEST" \
      --data-root "$STWM_ROOT/data/external" \
      --output-dir "$FIGURE_OUT" \
      --cases-per-group 8 \
      --min-consistent-seeds "$FIGURE_MIN_CONSISTENT_SEEDS"
else
  echo "[1b-state-identifiability] skip figure casebook (BUILD_FIGURES=$BUILD_FIGURES)"
fi

echo "[1b-state-identifiability] done"
echo "  protocol_manifest: $PROTOCOL_MANIFEST"
echo "  protocol_report:   $PROTOCOL_REPORT"
echo "  summary_json:      $SUMMARY_JSON"
echo "  summary_md:        $SUMMARY_MD"
echo "  decoupling_json:   $DECOUPLING_JSON"
echo "  figure_manifest:   $FIGURE_OUT/figure_manifest.json"
