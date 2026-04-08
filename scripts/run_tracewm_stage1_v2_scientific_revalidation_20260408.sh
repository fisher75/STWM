#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage1_v2_scientific_revalidation_${DATE_TAG}.log"
PROTOCOL_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_SCIENTIFIC_REVALIDATION_PROTOCOL_${DATE_TAG}.md"
CONTRACT_PATH="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_${DATE_TAG}.json"
RUNTIME_JSON="$WORK_ROOT/reports/stage1_v2_recommended_runtime_${DATE_TAG}.json"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage1-v2-scireval] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-scireval] python=$PYTHON_BIN"
echo "[stage1-v2-scireval] protocol_doc=$PROTOCOL_DOC"
echo "[stage1-v2-scireval] contract_path=$CONTRACT_PATH"
echo "[stage1-v2-scireval] runtime_json=$RUNTIME_JSON"

if [[ ! -f "$PROTOCOL_DOC" ]]; then
  echo "[stage1-v2-scireval] missing_protocol_doc"
  exit 2
fi

if [[ ! -f "$RUNTIME_JSON" ]]; then
  echo "[stage1-v2-scireval] missing_recommended_runtime_json"
  exit 3
fi

SELECTED_GPU=$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$RUNTIME_JSON', 'r', encoding='utf-8'))
policy = p.get('selected_gpu_policy', {}) if isinstance(p.get('selected_gpu_policy', {}), dict) else {}
print(int(policy.get('selected_gpu_id', -1)))
PY
)

if [[ "$SELECTED_GPU" -ge 0 ]]; then
  export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
  echo "[stage1-v2-scireval] selected_gpu=$SELECTED_GPU"
else
  echo "[stage1-v2-scireval] selected_gpu_not_set_using_default_cuda_context"
fi

echo "[stage1-v2-scireval] step=run_scientific_revalidation"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/run_stage1_v2_scientific_revalidation.py" \
  --contract-path "$CONTRACT_PATH" \
  --recommended-runtime-json "$RUNTIME_JSON" \
  --state-report-json "$WORK_ROOT/reports/stage1_v2_ablation_state_${DATE_TAG}.json" \
  --state-report-md "$WORK_ROOT/docs/STAGE1_V2_ABLATION_STATE_${DATE_TAG}.md" \
  --backbone-report-json "$WORK_ROOT/reports/stage1_v2_ablation_backbone_${DATE_TAG}.json" \
  --backbone-report-md "$WORK_ROOT/docs/STAGE1_V2_ABLATION_BACKBONE_${DATE_TAG}.md" \
  --losses-report-json "$WORK_ROOT/reports/stage1_v2_ablation_losses_${DATE_TAG}.json" \
  --losses-report-md "$WORK_ROOT/docs/STAGE1_V2_ABLATION_LOSSES_${DATE_TAG}.md" \
  --final-report-json "$WORK_ROOT/reports/stage1_v2_final_comparison_${DATE_TAG}.json" \
  --final-report-md "$WORK_ROOT/docs/TRACEWM_STAGE1_V2_RESULTS_${DATE_TAG}.md" \
  --run-details-json "$WORK_ROOT/reports/stage1_v2_scientific_revalidation_runs_${DATE_TAG}.json" \
  --epochs 1 \
  --steps-per-variant 8 \
  --eval-steps 6 \
  --batch-size 2 \
  --max-samples-per-dataset-train 128 \
  --max-samples-per-dataset-val 64 \
  --max-tokens 64

echo "[stage1-v2-scireval] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-scireval] log=$LOG_PATH"
