#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

OUT_ROOT="${1:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_real_confirmation}"
SEEDS_CSV="${STWM_V4_2_REAL_1B_VIS_SEEDS:-42,123}"

BASE_RUNS_ROOT="$OUT_ROOT/base"
STATE_RUNS_ROOT="$OUT_ROOT/state"

BASE_VIS_OUT="${STWM_V4_2_REAL_1B_BASE_VIS_OUT:-$STWM_ROOT/outputs/visualizations/stwm_v4_2_real_1b_multiseed_casebook}"
STATE_VIS_OUT="${STWM_V4_2_REAL_1B_STATE_VIS_OUT:-$STWM_ROOT/outputs/visualizations/stwm_v4_2_real_1b_state_identifiability_figures}"
DEMO_OUT="${STWM_V4_2_REAL_1B_DEMO_OUT:-$STWM_ROOT/outputs/visualizations/stwm_v4_2_real_1b_demo}"

FPS="${STWM_V4_2_REAL_1B_DEMO_FPS:-4}"
MAX_FRAMES="${STWM_V4_2_REAL_1B_DEMO_MAX_FRAMES:-240}"

echo "[1b-real-vis] build casebooks and demo package"
STWM_V4_2_1B_BASE_RUNS_ROOT="$BASE_RUNS_ROOT" \
STWM_V4_2_1B_STATE_RUNS_ROOT="$STATE_RUNS_ROOT" \
STWM_V4_2_1B_VIS_SEEDS="$SEEDS_CSV" \
STWM_V4_2_1B_BASE_VIS_OUT="$BASE_VIS_OUT" \
STWM_V4_2_1B_STATE_VIS_OUT="$STATE_VIS_OUT" \
STWM_V4_2_1B_DEMO_OUT="$DEMO_OUT" \
STWM_V4_2_1B_DEMO_FPS="$FPS" \
STWM_V4_2_1B_DEMO_MAX_FRAMES="$MAX_FRAMES" \
  bash "$SCRIPT_DIR/build_stwm_v4_2_1b_visualization.sh"

echo "[1b-real-vis] done"
echo "  semantic-sensitive:      $BASE_VIS_OUT/semantic_sensitive_cases"
echo "  instance-disambiguation: $STATE_VIS_OUT/instance_disambiguation_cases"
echo "  future-grounding:        $STATE_VIS_OUT/future_grounding_cases"
echo "  demo_manifest:           $DEMO_OUT/demo_manifest.json"
echo "  demo_video:              $DEMO_OUT/stwm_v4_2_1b_demo.mp4"
