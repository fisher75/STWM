#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_minival_multiseed}"
STEPS="${STWM_V4_2_1B_MINIVAL_STEPS:-120}"
SAMPLE_LIMIT="${STWM_V4_2_1B_MINIVAL_SAMPLE_LIMIT:-18}"
SEEDS_CSV="${STWM_V4_2_1B_MINIVAL_SEEDS:-42,123,456}"
RUNS_CSV="${STWM_V4_2_1B_MINIVAL_RUNS:-full_v4_2,wo_semantics_v4_2,wo_object_bias_v4_2}"
SKIP_EXISTING="${STWM_V4_2_1B_MINIVAL_SKIP_EXISTING:-0}"

MANIFEST="${STWM_V4_2_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"
MODEL_PRESET="${STWM_V4_2_1B_PRESET:-prototype_1b_v4_2}"
PRESET_FILE="${STWM_V4_2_1B_PRESET_FILE:-$STWM_ROOT/code/stwm/configs/model_presets_v4_2_1b.json}"

SUMMARY_JSON="$OUT_ROOT/comparison_multiseed.json"
SUMMARY_MD="$OUT_ROOT/comparison_multiseed.md"
DECOUPLING_JSON="${STWM_V4_2_1B_BASE_DECOUPLING_JSON:-$STWM_ROOT/reports/stwm_v4_2_1b_query_decoupling_multiseed.json}"

mkdir -p "$OUT_ROOT"

PYTHON_BIN=(/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm python)

run_case() {
  local seed="$1"
  local run_name="$2"
  shift 2
  local out_dir="$OUT_ROOT/seed_${seed}/$run_name"
  local log_file="$STWM_ROOT/logs/stwm_v4_2_1b_minival_seed${seed}_${run_name}.log"
  local final_ckpt="$out_dir/final_model.pt"
  local summary_file="$out_dir/mini_val_summary.json"

  if [[ "$SKIP_EXISTING" == "1" && -f "$final_ckpt" && -f "$summary_file" ]]; then
    echo "[stwm-v4.2-1b-base] skip existing seed=${seed} run=${run_name}"
    return
  fi

  echo "[stwm-v4.2-1b-base] start seed=${seed} run=${run_name}"
  PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
    "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/trainers/train_stwm_v4_2.py" \
      --data-root "$STWM_ROOT/data/external" \
      --manifest "$MANIFEST" \
      --output-dir "$out_dir" \
      --run-name "$run_name" \
      --model-preset "$MODEL_PRESET" \
      --preset-file "$PRESET_FILE" \
      --steps "$STEPS" \
      --sample-limit "$SAMPLE_LIMIT" \
      --seed "$seed" \
      --use-teacher-priors \
      --summary-name mini_val_summary.json \
      --log-name train_log.jsonl \
      --save-checkpoint \
      "$@" \
      >"$log_file" 2>&1
  echo "[stwm-v4.2-1b-base] done seed=${seed} run=${run_name}"
}

IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"
IFS=',' read -r -a run_list <<< "$RUNS_CSV"
for seed in "${seed_list[@]}"; do
  for run_name in "${run_list[@]}"; do
    case "$run_name" in
      full_v4_2)
        run_case "$seed" "$run_name"
        ;;
      wo_semantics_v4_2)
        run_case "$seed" "$run_name" --disable-semantics
        ;;
      wo_object_bias_v4_2)
        run_case "$seed" "$run_name" --neutralize-object-bias
        ;;
      *)
        echo "[stwm-v4.2-1b-base] unsupported run: $run_name" >&2
        exit 2
        ;;
    esac
  done
done

echo "[stwm-v4.2-1b-base] summarize multi-seed"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/summarize_stwm_v4_2_minival_multiseed.py" \
    --runs-root "$OUT_ROOT" \
    --seeds "$SEEDS_CSV" \
    --runs "$RUNS_CSV" \
    --summary-name mini_val_summary.json \
    --output-json "$SUMMARY_JSON" \
    --output-md "$SUMMARY_MD"

echo "[stwm-v4.2-1b-base] run query decoupling"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/query_trajectory_decoupling_multiseed.py" \
    --runs-root "$OUT_ROOT" \
    --seeds "$SEEDS_CSV" \
    --runs "$RUNS_CSV" \
    --output-json "$DECOUPLING_JSON"

echo "[stwm-v4.2-1b-base] done"
echo "  summary_json:    $SUMMARY_JSON"
echo "  summary_md:      $SUMMARY_MD"
echo "  decoupling_json: $DECOUPLING_JSON"
