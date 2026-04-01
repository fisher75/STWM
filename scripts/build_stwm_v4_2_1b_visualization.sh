#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

BASE_RUNS_ROOT="${STWM_V4_2_1B_BASE_RUNS_ROOT:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_minival_multiseed}"
STATE_RUNS_ROOT="${STWM_V4_2_1B_STATE_RUNS_ROOT:-$STWM_ROOT/outputs/training/stwm_v4_2_1b_state_identifiability}"
SEEDS_CSV="${STWM_V4_2_1B_VIS_SEEDS:-42,123,456}"

BASE_MANIFEST="${STWM_V4_2_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_week2_minival_v2.json}"
STATE_MANIFEST="${STWM_V4_2_1B_IDENTIFIABILITY_MANIFEST:-$STWM_ROOT/manifests/minisplits/stwm_v4_2_1b_state_identifiability_v1.json}"

BASE_VIS_OUT="${STWM_V4_2_1B_BASE_VIS_OUT:-$STWM_ROOT/outputs/visualizations/stwm_v4_2_1b_multiseed_casebook}"
STATE_VIS_OUT="${STWM_V4_2_1B_STATE_VIS_OUT:-$STWM_ROOT/outputs/visualizations/stwm_v4_2_1b_state_identifiability_figures}"
DEMO_OUT="${STWM_V4_2_1B_DEMO_OUT:-$STWM_ROOT/outputs/visualizations/stwm_v4_2_1b_demo}"

FPS="${STWM_V4_2_1B_DEMO_FPS:-2}"
MAX_FRAMES="${STWM_V4_2_1B_DEMO_MAX_FRAMES:-120}"

PYTHON_BIN=(/home/chen034/miniconda3/bin/conda run --no-capture-output -n stwm python)

echo "[1b-visualization] build multi-seed casebook"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/evaluators/build_stwm_v4_2_multiseed_casebook.py" \
    --runs-root "$BASE_RUNS_ROOT" \
    --seeds "$SEEDS_CSV" \
    --manifest "$BASE_MANIFEST" \
    --data-root "$STWM_ROOT/data/external" \
    --output-dir "$BASE_VIS_OUT" \
    --cases-per-group 8 \
    --min-consistent-seeds 2

echo "[1b-visualization] build state-identifiability figures"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/evaluators/build_stwm_v4_2_state_identifiability_figures.py" \
    --runs-root "$STATE_RUNS_ROOT" \
    --seeds "$SEEDS_CSV" \
    --manifest "$STATE_MANIFEST" \
    --data-root "$STWM_ROOT/data/external" \
    --output-dir "$STATE_VIS_OUT" \
    --cases-per-group 8 \
    --min-consistent-seeds 2

echo "[1b-visualization] package storyboard/video"
PYTHONPATH="$STWM_ROOT/code:${PYTHONPATH:-}" \
  "${PYTHON_BIN[@]}" "$STWM_ROOT/code/stwm/tools/package_stwm_v4_2_video_demo.py" \
    --figure-dirs "$BASE_VIS_OUT,$STATE_VIS_OUT" \
    --output-dir "$DEMO_OUT" \
    --fps "$FPS" \
    --max-frames "$MAX_FRAMES"

echo "[1b-visualization] done"
echo "  base_figure_manifest:  $BASE_VIS_OUT/figure_manifest.json"
echo "  state_figure_manifest: $STATE_VIS_OUT/figure_manifest.json"
echo "  demo_manifest:         $DEMO_OUT/demo_manifest.json"
echo "  demo_video:            $DEMO_OUT/stwm_v4_2_1b_demo.mp4"
