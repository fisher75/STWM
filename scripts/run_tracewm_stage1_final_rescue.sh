#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"
SEED="20260408"

CONTRACT_PATH="$DATA_ROOT/_manifests/stage1_data_contract_${DATE_TAG}.json"
MINISPLIT_PATH="$DATA_ROOT/_manifests/stage1_minisplits_${DATE_TAG}.json"
ITER1_SPLITS_PATH="$DATA_ROOT/_manifests/stage1_iter1_splits_${DATE_TAG}.json"

FREEZE_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_FINAL_JOINT_RESCUE_${DATE_TAG}.md"

RUN_PCGRAD="tracewm_stage1_rescue_pcgrad"
RUN_GRADNORM="tracewm_stage1_rescue_gradnorm"
RUN_SHARED_PRIVATE="tracewm_stage1_rescue_shared_private"
RUN_COMBO="tracewm_stage1_rescue_shared_private_plus_best_grad"

PCGRAD_SUMMARY="$WORK_ROOT/reports/${RUN_PCGRAD}_summary.json"
GRADNORM_SUMMARY="$WORK_ROOT/reports/${RUN_GRADNORM}_summary.json"
SHARED_PRIVATE_SUMMARY="$WORK_ROOT/reports/${RUN_SHARED_PRIVATE}_summary.json"
COMBO_SUMMARY="$WORK_ROOT/reports/${RUN_COMBO}_summary.json"

ITER1_POINT_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_pointodyssey_only_summary.json"
ITER1_KUBRIC_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_kubric_only_summary.json"
FIX2_BEST_JOINT_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_fix2_joint_balanced_lossnorm_summary.json"

FINAL_COMPARISON_JSON="$WORK_ROOT/reports/tracewm_stage1_final_rescue_comparison_${DATE_TAG}.json"
FINAL_RESULTS_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_FINAL_RESCUE_RESULTS_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$WORK_ROOT/outputs/training" "$WORK_ROOT/logs"

export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

for f in "$CONTRACT_PATH" "$MINISPLIT_PATH" "$ITER1_SPLITS_PATH" "$FREEZE_DOC" "$ITER1_POINT_SUMMARY" "$ITER1_KUBRIC_SUMMARY" "$FIX2_BEST_JOINT_SUMMARY"; do
  if [[ ! -f "$f" ]]; then
    echo "[final_rescue] required_file_missing=$f"
    exit 23
  fi
done

echo "[final_rescue] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[final_rescue] work_root=$WORK_ROOT"
echo "[final_rescue] data_root=$DATA_ROOT"
echo "[final_rescue] python=$PYTHON_BIN"
echo "[final_rescue] contract_frozen=$CONTRACT_PATH"
echo "[final_rescue] minisplit_frozen=$MINISPLIT_PATH"
echo "[final_rescue] iter1_splits=$ITER1_SPLITS_PATH"
echo "[final_rescue] freeze_doc=$FREEZE_DOC"

run_rescue() {
  local run_name="$1"
  local train_mode="$2"
  local summary_json="$3"

  local output_dir="$WORK_ROOT/outputs/training/${run_name}"

  echo "[final_rescue] run=${run_name} train_mode=${train_mode}"
  "$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/trainers/train_tracewm_stage1_final_rescue.py" \
    --run-name "$run_name" \
    --train-mode "$train_mode" \
    --data-root "$DATA_ROOT" \
    --splits-path "$ITER1_SPLITS_PATH" \
    --output-dir "$output_dir" \
    --summary-json "$summary_json" \
    --seed "$SEED" \
    --epochs 6 \
    --steps-per-epoch 80 \
    --batch-size 4 \
    --hidden-dim 128 \
    --private-adapter-dim 16 \
    --lr 1e-3 \
    --weight-decay 0.0 \
    --free-loss-weight 0.5 \
    --gradnorm-alpha 1.5 \
    --gradnorm-weight-lr 0.025 \
    --clip-grad-norm 1.0 \
    --eval-max-tapvid-samples 6 \
    --eval-max-tapvid3d-samples 12 \
    --no-auto-resume
}

run_rescue "$RUN_PCGRAD" "pcgrad" "$PCGRAD_SUMMARY"
run_rescue "$RUN_GRADNORM" "gradnorm" "$GRADNORM_SUMMARY"
run_rescue "$RUN_SHARED_PRIVATE" "shared_private" "$SHARED_PRIVATE_SUMMARY"

BEST_GRAD_METHOD=$("$PYTHON_BIN" -c "import json,sys; pc=json.load(open(sys.argv[1], 'r', encoding='utf-8')); gn=json.load(open(sys.argv[2], 'r', encoding='utf-8')); pt=pc['final_metrics']['tapvid']['free_rollout_endpoint_l2']; gt=gn['final_metrics']['tapvid']['free_rollout_endpoint_l2']; print('pcgrad' if pt <= gt else 'gradnorm')" "$PCGRAD_SUMMARY" "$GRADNORM_SUMMARY")

if [[ "$BEST_GRAD_METHOD" == "pcgrad" ]]; then
  COMBO_MODE="shared_private_plus_pcgrad"
else
  COMBO_MODE="shared_private_plus_gradnorm"
fi

echo "[final_rescue] selected_best_gradient_method_for_combo=$BEST_GRAD_METHOD"
echo "[final_rescue] combo_mode=$COMBO_MODE"

run_rescue "$RUN_COMBO" "$COMBO_MODE" "$COMBO_SUMMARY"

echo "[final_rescue] step=summarize_final_comparison"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/tools/summarize_stage1_final_rescue.py" \
  --freeze-doc "$FREEZE_DOC" \
  --iter1-point-summary "$ITER1_POINT_SUMMARY" \
  --iter1-kubric-summary "$ITER1_KUBRIC_SUMMARY" \
  --fix2-best-joint-summary "$FIX2_BEST_JOINT_SUMMARY" \
  --pcgrad-summary "$PCGRAD_SUMMARY" \
  --gradnorm-summary "$GRADNORM_SUMMARY" \
  --shared-private-summary "$SHARED_PRIVATE_SUMMARY" \
  --shared-private-plus-best-grad-summary "$COMBO_SUMMARY" \
  --selected-best-gradient-method "$BEST_GRAD_METHOD" \
  --comparison-json "$FINAL_COMPARISON_JSON" \
  --results-md "$FINAL_RESULTS_DOC"

echo "[final_rescue] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[final_rescue] pcgrad_summary=$PCGRAD_SUMMARY"
echo "[final_rescue] gradnorm_summary=$GRADNORM_SUMMARY"
echo "[final_rescue] shared_private_summary=$SHARED_PRIVATE_SUMMARY"
echo "[final_rescue] combo_summary=$COMBO_SUMMARY"
echo "[final_rescue] comparison_json=$FINAL_COMPARISON_JSON"
echo "[final_rescue] results_doc=$FINAL_RESULTS_DOC"
echo "[final_rescue] selected_best_gradient_method_for_combo=$BEST_GRAD_METHOD"
