#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_real_confirmation}"
SEEDS_CSV="${STWM_V4_2_REAL_1B_SEEDS:-42,123}"
RUNS_CSV="${STWM_V4_2_REAL_1B_RUNS:-full_v4_2,wo_semantics_v4_2,wo_object_bias_v4_2}"

BASE_ROOT="$OUT_ROOT/base"
STATE_ROOT="$OUT_ROOT/state"

BASE_SUMMARY_JSON="$BASE_ROOT/comparison_multiseed.json"
BASE_SUMMARY_MD="$BASE_ROOT/comparison_multiseed.md"
BASE_DECOUPLING_JSON="${STWM_V4_2_REAL_1B_BASE_DECOUPLING_JSON:-$STWM_ROOT/reports/stwm_v4_2_1b_real_query_decoupling_multiseed.json}"

STATE_SUMMARY_JSON="$STATE_ROOT/comparison_state_identifiability.json"
STATE_SUMMARY_MD="$STATE_ROOT/comparison_state_identifiability.md"
STATE_DECOUPLING_JSON="${STWM_V4_2_REAL_1B_STATE_DECOUPLING_JSON:-$STWM_ROOT/reports/stwm_v4_2_1b_real_state_identifiability_decoupling_v1.json}"

STATE_MANIFEST="${STWM_V4_2_1B_IDENTIFIABILITY_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_v4_2_1b_state_identifiability_v1.json}"

BASE_220M_SUMMARY="${STWM_V4_2_220M_BASE_SUMMARY:-$STWM_ROOT/outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.json}"
STATE_220M_SUMMARY="${STWM_V4_2_220M_STATE_SUMMARY:-$STWM_ROOT/outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.json}"
DECOUPLING_220M="${STWM_V4_2_220M_STATE_DECOUPLING:-$STWM_ROOT/reports/stwm_v4_2_state_identifiability_decoupling_v1.json}"

COMPARE_JSON="${STWM_V4_2_220M_VS_1B_REAL_JSON:-$STWM_ROOT/reports/stwm_v4_2_220m_vs_1b_real_confirmation.json}"
COMPARE_MD="${STWM_V4_2_3B_GO_NO_GO_REAL_DOC:-$STWM_ROOT/docs/STWM_V4_2_3B_GO_NO_GO_REAL_1B.md}"

PYTHON_BIN=(/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm python)

check_required_artifacts() {
  local seed="$1"
  local run_name="$2"

  local base_dir="$BASE_ROOT/seed_${seed}/${run_name}"
  local state_dir="$STATE_ROOT/seed_${seed}/${run_name}"

  local required=(
    "$base_dir/final_model.pt"
    "$base_dir/train_log.jsonl"
    "$base_dir/mini_val_summary.json"
    "$state_dir/eval_model.pt"
    "$state_dir/train_log.jsonl"
    "$state_dir/mini_val_summary.json"
  )

  local f
  for f in "${required[@]}"; do
    if [[ ! -f "$f" ]]; then
      echo "[1b-real-finalize] missing required artifact: $f" >&2
      return 1
    fi
  done
}

echo "[1b-real-finalize] validate required artifacts"
IFS=',' read -r -a seed_list <<< "$SEEDS_CSV"
IFS=',' read -r -a run_list <<< "$RUNS_CSV"

for seed in "${seed_list[@]}"; do
  for run_name in "${run_list[@]}"; do
    check_required_artifacts "$seed" "$run_name"
  done
done

echo "[1b-real-finalize] summarize base protocol"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/summarize_stwm_v4_2_minival_multiseed.py" \
    --runs-root "$BASE_ROOT" \
    --seeds "$SEEDS_CSV" \
    --runs "$RUNS_CSV" \
    --summary-name mini_val_summary.json \
    --output-json "$BASE_SUMMARY_JSON" \
    --output-md "$BASE_SUMMARY_MD"

echo "[1b-real-finalize] decoupling on base protocol"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/query_trajectory_decoupling_multiseed.py" \
    --runs-root "$BASE_ROOT" \
    --seeds "$SEEDS_CSV" \
    --runs "$RUNS_CSV" \
    --output-json "$BASE_DECOUPLING_JSON"

echo "[1b-real-finalize] summarize state-identifiability protocol"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/summarize_stwm_v4_2_state_identifiability.py" \
    --runs-root "$STATE_ROOT" \
    --manifest "$STATE_MANIFEST" \
    --seeds "$SEEDS_CSV" \
    --runs "$RUNS_CSV" \
    --summary-name mini_val_summary.json \
    --log-name train_log.jsonl \
    --output-json "$STATE_SUMMARY_JSON" \
    --output-md "$STATE_SUMMARY_MD"

echo "[1b-real-finalize] decoupling on harder protocol"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/query_trajectory_decoupling_multiseed.py" \
    --runs-root "$STATE_ROOT" \
    --seeds "$SEEDS_CSV" \
    --runs "$RUNS_CSV" \
    --output-json "$STATE_DECOUPLING_JSON"

echo "[1b-real-finalize] compare 220M vs real 1B"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/compare_stwm_v4_2_220m_vs_1b.py" \
    --base-220m "$BASE_220M_SUMMARY" \
    --base-1b "$BASE_SUMMARY_JSON" \
    --state-220m "$STATE_220M_SUMMARY" \
    --state-1b "$STATE_SUMMARY_JSON" \
    --decoupling-220m "$DECOUPLING_220M" \
    --decoupling-1b "$STATE_DECOUPLING_JSON" \
    --output-json "$COMPARE_JSON" \
    --output-md "$COMPARE_MD"

echo "[1b-real-finalize] done"
echo "  base_summary:      $BASE_SUMMARY_JSON"
echo "  base_decoupling:   $BASE_DECOUPLING_JSON"
echo "  state_summary:     $STATE_SUMMARY_JSON"
echo "  state_decoupling:  $STATE_DECOUPLING_JSON"
echo "  compare_json:      $COMPARE_JSON"
echo "  compare_md:        $COMPARE_MD"
