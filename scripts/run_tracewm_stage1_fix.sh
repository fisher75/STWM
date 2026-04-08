#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"
SEED="20260408"

CONTRACT_PATH="$DATA_ROOT/_manifests/stage1_data_contract_${DATE_TAG}.json"
MINISPLIT_PATH="$DATA_ROOT/_manifests/stage1_minisplits_${DATE_TAG}.json"
ITER1_SPLITS_PATH="$DATA_ROOT/_manifests/stage1_iter1_splits_${DATE_TAG}.json"

DIAG_JSON="$WORK_ROOT/reports/tracewm_stage1_iter1_diagnosis_${DATE_TAG}.json"
DIAG_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_MODEL_FIX_ROUND_${DATE_TAG}.md"

RUN_BAL="tracewm_stage1_fix_joint_balanced_sampler"
RUN_LOSS="tracewm_stage1_fix_joint_loss_normalized"
RUN_SRC="tracewm_stage1_fix_joint_source_conditioned"

BAL_SUMMARY="$WORK_ROOT/reports/${RUN_BAL}_summary.json"
LOSS_SUMMARY="$WORK_ROOT/reports/${RUN_LOSS}_summary.json"
SRC_SUMMARY="$WORK_ROOT/reports/${RUN_SRC}_summary.json"

POINT_ITER1_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_pointodyssey_only_summary.json"
KUBRIC_ITER1_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_kubric_only_summary.json"
JOINT_ITER1_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_joint_po_kubric_summary.json"

FIX_COMPARISON_JSON="$WORK_ROOT/reports/tracewm_stage1_fix_comparison_${DATE_TAG}.json"
FIX_RESULTS_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_FIX_RESULTS_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$WORK_ROOT/outputs/training"

export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

for f in "$CONTRACT_PATH" "$MINISPLIT_PATH" "$ITER1_SPLITS_PATH" "$DIAG_JSON" "$DIAG_DOC" "$POINT_ITER1_SUMMARY" "$KUBRIC_ITER1_SUMMARY" "$JOINT_ITER1_SUMMARY"; do
  if [[ ! -f "$f" ]]; then
    echo "[fix] required_file_missing=$f"
    exit 23
  fi
done

echo "[fix] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[fix] work_root=$WORK_ROOT"
echo "[fix] data_root=$DATA_ROOT"
echo "[fix] python=$PYTHON_BIN"
echo "[fix] contract_frozen=$CONTRACT_PATH"
echo "[fix] minisplit_frozen=$MINISPLIT_PATH"
echo "[fix] iter1_splits=$ITER1_SPLITS_PATH"
echo "[fix] diagnosis_json=$DIAG_JSON"

run_fix() {
  local run_name="$1"
  local fix_mode="$2"
  local summary_json="$3"

  local output_dir="$WORK_ROOT/outputs/training/${run_name}"

  echo "[fix] run=${run_name} mode=${fix_mode}"
  "$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/trainers/train_tracewm_stage1_fix.py" \
    --run-name "$run_name" \
    --fix-mode "$fix_mode" \
    --data-root "$DATA_ROOT" \
    --splits-path "$ITER1_SPLITS_PATH" \
    --output-dir "$output_dir" \
    --summary-json "$summary_json" \
    --seed "$SEED" \
    --epochs 6 \
    --steps-per-epoch 80 \
    --batch-size 4 \
    --hidden-dim 128 \
    --source-emb-dim 8 \
    --lr 1e-3 \
    --weight-decay 0.0 \
    --free-loss-weight 0.5 \
    --eval-max-tapvid-samples 6 \
    --eval-max-tapvid3d-samples 12 \
    --no-auto-resume
}

run_fix "$RUN_BAL" "joint_balanced_sampler" "$BAL_SUMMARY"
run_fix "$RUN_LOSS" "joint_loss_normalized" "$LOSS_SUMMARY"
run_fix "$RUN_SRC" "joint_source_conditioned" "$SRC_SUMMARY"

echo "[fix] step=summarize_comparison"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/tools/summarize_stage1_fix.py" \
  --diag-json "$DIAG_JSON" \
  --iter1-joint-summary "$JOINT_ITER1_SUMMARY" \
  --iter1-point-summary "$POINT_ITER1_SUMMARY" \
  --iter1-kubric-summary "$KUBRIC_ITER1_SUMMARY" \
  --fix-balanced-summary "$BAL_SUMMARY" \
  --fix-lossnorm-summary "$LOSS_SUMMARY" \
  --fix-sourcecond-summary "$SRC_SUMMARY" \
  --comparison-json "$FIX_COMPARISON_JSON" \
  --results-md "$FIX_RESULTS_DOC"

echo "[fix] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[fix] balanced_summary=$BAL_SUMMARY"
echo "[fix] lossnorm_summary=$LOSS_SUMMARY"
echo "[fix] sourcecond_summary=$SRC_SUMMARY"
echo "[fix] comparison_json=$FIX_COMPARISON_JSON"
echo "[fix] results_doc=$FIX_RESULTS_DOC"
