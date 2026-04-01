#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

BASE_ROOT="${STWM_V4_2_1B_BASE_ROOT:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_minival_multiseed}"
STATE_ROOT="${STWM_V4_2_1B_STATE_ROOT:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_state_identifiability}"

BASE_220M_SUMMARY="${STWM_V4_2_220M_BASE_SUMMARY:-$STWM_ROOT/outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.json}"
STATE_220M_SUMMARY="${STWM_V4_2_220M_STATE_SUMMARY:-$STWM_ROOT/outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.json}"
DECOUPLING_220M="${STWM_V4_2_220M_STATE_DECOUPLING:-$STWM_ROOT/reports/stwm_v4_2_state_identifiability_decoupling_v1.json}"

BASE_1B_SUMMARY="$BASE_ROOT/comparison_multiseed.json"
STATE_1B_SUMMARY="$STATE_ROOT/comparison_state_identifiability.json"
DECOUPLING_1B="${STWM_V4_2_1B_IDENTIFIABILITY_DECOUPLING_JSON:-$STWM_ROOT/reports/stwm_v4_2_1b_state_identifiability_decoupling_v1.json}"

COMPARE_JSON="${STWM_V4_2_220M_VS_1B_JSON:-$STWM_ROOT/reports/stwm_v4_2_220m_vs_1b.json}"
GO_NO_GO_MD="${STWM_V4_2_3B_GO_NO_GO_DOC:-$STWM_ROOT/docs/STWM_V4_2_3B_GO_NO_GO.md}"

RUN_VISUALS="${STWM_V4_2_1B_RUN_VISUALS:-1}"

PYTHON_BIN=(/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm python)

echo "[1b-confirmation] start base multi-seed runs"
bash "$SCRIPT_DIR/run_stwm_v4_2_1b_minival_multiseed.sh" "$BASE_ROOT"

echo "[1b-confirmation] start state-identifiability runs"
STWM_V4_2_1B_BASE_RUNS_ROOT="$BASE_ROOT" \
  bash "$SCRIPT_DIR/run_stwm_v4_2_1b_state_identifiability.sh" "$STATE_ROOT"

echo "[1b-confirmation] compare 220M vs 1B and answer 5 questions"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/compare_stwm_v4_2_220m_vs_1b.py" \
    --base-220m "$BASE_220M_SUMMARY" \
    --base-1b "$BASE_1B_SUMMARY" \
    --state-220m "$STATE_220M_SUMMARY" \
    --state-1b "$STATE_1B_SUMMARY" \
    --decoupling-220m "$DECOUPLING_220M" \
    --decoupling-1b "$DECOUPLING_1B" \
    --output-json "$COMPARE_JSON" \
    --output-md "$GO_NO_GO_MD"

if [[ "$RUN_VISUALS" == "1" ]]; then
  echo "[1b-confirmation] build visualization/video package"
  STWM_V4_2_1B_BASE_RUNS_ROOT="$BASE_ROOT" \
  STWM_V4_2_1B_STATE_RUNS_ROOT="$STATE_ROOT" \
    bash "$SCRIPT_DIR/build_stwm_v4_2_1b_visualization.sh"
fi

echo "[1b-confirmation] done"
echo "  base_summary_1b:  $BASE_1B_SUMMARY"
echo "  state_summary_1b: $STATE_1B_SUMMARY"
echo "  compare_json:     $COMPARE_JSON"
echo "  go_no_go_doc:     $GO_NO_GO_MD"
