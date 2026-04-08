#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"
SEED="20260408"

CONTRACT_PATH="$DATA_ROOT/_manifests/stage1_data_contract_${DATE_TAG}.json"
MINISPLIT_PATH="$DATA_ROOT/_manifests/stage1_minisplits_${DATE_TAG}.json"
ITER1_SPLITS_PATH="$DATA_ROOT/_manifests/stage1_iter1_splits_${DATE_TAG}.json"

ROUND2_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_MODEL_FIX_ROUND2_${DATE_TAG}.md"

RUN_NOWARMUP="tracewm_stage1_fix2_joint_balanced_lossnorm"
RUN_POINT_WARMUP="tracewm_stage1_fix2_point_warmup_then_joint_balanced_lossnorm"
RUN_KUBRIC_WARMUP="tracewm_stage1_fix2_kubric_warmup_then_joint_balanced_lossnorm"

NOWARMUP_SUMMARY="$WORK_ROOT/reports/${RUN_NOWARMUP}_summary.json"
POINT_WARMUP_SUMMARY="$WORK_ROOT/reports/${RUN_POINT_WARMUP}_summary.json"
KUBRIC_WARMUP_SUMMARY="$WORK_ROOT/reports/${RUN_KUBRIC_WARMUP}_summary.json"

POINT_ITER1_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_pointodyssey_only_summary.json"
KUBRIC_ITER1_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_kubric_only_summary.json"

FIX2_COMPARISON_JSON="$WORK_ROOT/reports/tracewm_stage1_fix2_comparison_${DATE_TAG}.json"
FIX2_RESULTS_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_FIX2_RESULTS_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$WORK_ROOT/outputs/training" "$WORK_ROOT/logs"

export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

for f in "$CONTRACT_PATH" "$MINISPLIT_PATH" "$ITER1_SPLITS_PATH" "$ROUND2_DOC" "$POINT_ITER1_SUMMARY" "$KUBRIC_ITER1_SUMMARY"; do
  if [[ ! -f "$f" ]]; then
    echo "[fix2] required_file_missing=$f"
    exit 23
  fi
done

echo "[fix2] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[fix2] work_root=$WORK_ROOT"
echo "[fix2] data_root=$DATA_ROOT"
echo "[fix2] python=$PYTHON_BIN"
echo "[fix2] contract_frozen=$CONTRACT_PATH"
echo "[fix2] minisplit_frozen=$MINISPLIT_PATH"
echo "[fix2] iter1_splits=$ITER1_SPLITS_PATH"
echo "[fix2] round2_doc=$ROUND2_DOC"

run_fix2() {
  local run_name="$1"
  local fix_mode="$2"
  local summary_json="$3"

  local output_dir="$WORK_ROOT/outputs/training/${run_name}"

  echo "[fix2] run=${run_name} mode=${fix_mode}"
  "$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/trainers/train_tracewm_stage1_fix_round2.py" \
    --run-name "$run_name" \
    --fix-mode "$fix_mode" \
    --data-root "$DATA_ROOT" \
    --splits-path "$ITER1_SPLITS_PATH" \
    --output-dir "$output_dir" \
    --summary-json "$summary_json" \
    --seed "$SEED" \
    --epochs 6 \
    --warmup-epochs 2 \
    --steps-per-epoch 80 \
    --batch-size 4 \
    --hidden-dim 128 \
    --lr 1e-3 \
    --weight-decay 0.0 \
    --free-loss-weight 0.5 \
    --eval-max-tapvid-samples 6 \
    --eval-max-tapvid3d-samples 12 \
    --no-auto-resume
}

run_fix2 "$RUN_NOWARMUP" "joint_balanced_lossnorm" "$NOWARMUP_SUMMARY"
run_fix2 "$RUN_POINT_WARMUP" "point_warmup_then_joint_balanced_lossnorm" "$POINT_WARMUP_SUMMARY"
run_fix2 "$RUN_KUBRIC_WARMUP" "kubric_warmup_then_joint_balanced_lossnorm" "$KUBRIC_WARMUP_SUMMARY"

echo "[fix2] step=summarize_comparison"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/tools/summarize_stage1_fix_round2.py" \
  --round2-doc "$ROUND2_DOC" \
  --iter1-point-summary "$POINT_ITER1_SUMMARY" \
  --iter1-kubric-summary "$KUBRIC_ITER1_SUMMARY" \
  --run-nowarmup-summary "$NOWARMUP_SUMMARY" \
  --run-point-warmup-summary "$POINT_WARMUP_SUMMARY" \
  --run-kubric-warmup-summary "$KUBRIC_WARMUP_SUMMARY" \
  --comparison-json "$FIX2_COMPARISON_JSON" \
  --results-md "$FIX2_RESULTS_DOC"

echo "[fix2] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[fix2] nowarmup_summary=$NOWARMUP_SUMMARY"
echo "[fix2] point_warmup_summary=$POINT_WARMUP_SUMMARY"
echo "[fix2] kubric_warmup_summary=$KUBRIC_WARMUP_SUMMARY"
echo "[fix2] comparison_json=$FIX2_COMPARISON_JSON"
echo "[fix2] results_doc=$FIX2_RESULTS_DOC"
