#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATE_TAG="20260408"
RUN_NAME="stage2_core_mainline_train_20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage2_core_mainline_train_${DATE_TAG}.log"

PROTOCOL_DOC="$WORK_ROOT/docs/TRACEWM_STAGE2_CORE_MAINLINE_TRAIN_PROTOCOL_${DATE_TAG}.md"
EVAL_FIX_COMPARISON_JSON="$WORK_ROOT/reports/stage2_eval_fix_comparison_${DATE_TAG}.json"
STAGE2_CONTRACT_JSON="$WORK_ROOT/reports/stage2_bootstrap_data_contract_${DATE_TAG}.json"
RUNTIME_JSON="$WORK_ROOT/reports/stage1_v2_recommended_runtime_${DATE_TAG}.json"
STAGE1_BEST_CKPT="$WORK_ROOT/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_${DATE_TAG}/best.pt"

CHECKPOINT_DIR="$WORK_ROOT/outputs/checkpoints/$RUN_NAME"
RAW_RUN_JSON="$WORK_ROOT/reports/stage2_core_mainline_train_raw_${DATE_TAG}.json"
PROGRESS_JSON="$WORK_ROOT/reports/stage2_core_mainline_train_progress_${DATE_TAG}.json"
FINAL_JSON="$WORK_ROOT/reports/stage2_core_mainline_train_final_${DATE_TAG}.json"
RESULTS_MD="$WORK_ROOT/docs/STAGE2_CORE_MAINLINE_TRAIN_RESULTS_${DATE_TAG}.md"

GPU_SELECT_JSON="$WORK_ROOT/reports/stage2_core_mainline_train_gpu_selection_${DATE_TAG}.json"
GPU_LEASE_PATH="$WORK_ROOT/reports/stage1_v2_gpu_lease_${DATE_TAG}.json"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$CHECKPOINT_DIR"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage2-core-mainline] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-core-mainline] python=$PYTHON_BIN"

for f in \
  "$PROTOCOL_DOC" \
  "$EVAL_FIX_COMPARISON_JSON" \
  "$STAGE2_CONTRACT_JSON" \
  "$RUNTIME_JSON" \
  "$STAGE1_BEST_CKPT"
do
  if [[ ! -f "$f" ]]; then
    echo "[stage2-core-mainline] missing_required_file=$f"
    exit 2
  fi
done

echo "[stage2-core-mainline] step=validate_eval_fix_gate"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

payload = json.loads(Path("$EVAL_FIX_COMPARISON_JSON").read_text(encoding="utf-8"))

if str(payload.get("final_recommended_mainline", "")) != "stage2_core_cropenc":
    raise SystemExit("final_recommended_mainline must be stage2_core_cropenc")
if not bool(payload.get("can_continue_stage2_training", False)):
    raise SystemExit("can_continue_stage2_training must be true")
if str(payload.get("next_step_choice", "")) != "continue_stage2_training_core_only":
    raise SystemExit("next_step_choice must be continue_stage2_training_core_only")
if not bool(payload.get("core_only_better_than_core_plus_burst", False)):
    raise SystemExit("core_only_better_than_core_plus_burst must be true")

print("[stage2-core-mainline] final_recommended_mainline=stage2_core_cropenc")
print("[stage2-core-mainline] can_continue_stage2_training=True")
print("[stage2-core-mainline] next_step_choice=continue_stage2_training_core_only")
print("[stage2-core-mainline] core_only_better_than_core_plus_burst=True")
PY

echo "[stage2-core-mainline] step=validate_contract_exclusions"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path

contract = json.loads(Path("$STAGE2_CONTRACT_JSON").read_text(encoding="utf-8"))
excluded = contract.get("excluded_datasets", []) if isinstance(contract.get("excluded_datasets", []), list) else []
emap = {str(x.get("dataset_name", "")).upper(): x for x in excluded if isinstance(x, dict)}

if not bool(emap.get("TAO", {}).get("not_in_current_bootstrap", False)):
    raise SystemExit("TAO must remain excluded")
if not bool(emap.get("VISOR", {}).get("not_in_current_bootstrap", False)):
    raise SystemExit("VISOR must remain excluded")

print(f"[stage2-core-mainline] TAO_status={emap.get('TAO', {}).get('status_from_audit', '')}")
print(f"[stage2-core-mainline] VISOR_status={emap.get('VISOR', {}).get('status_from_audit', '')}")
PY

echo "[stage2-core-mainline] step=select_gpu_with_selector_and_lease"
GPU_DECISION=$(
  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

from stwm.infra.gpu_lease import acquire_lease
from stwm.infra.gpu_selector import select_single_gpu

runtime = json.load(open('$RUNTIME_JSON', 'r', encoding='utf-8'))
required_mem_gb = float(runtime.get('required_mem_gb', 40.0) or 40.0)
safety_margin_gb = float(runtime.get('safety_margin_gb', 8.0) or 8.0)
policy = runtime.get('selected_gpu_policy', {}) if isinstance(runtime.get('selected_gpu_policy', {}), dict) else {}
recommended_gpu_id = int(policy.get('selected_gpu_id', -1))

payload = select_single_gpu(
    required_mem_gb=required_mem_gb,
    safety_margin_gb=safety_margin_gb,
    sample_count=8,
    interval_sec=1.0,
    lease_path='$GPU_LEASE_PATH',
)

selected_gpu_id = int(payload.get('selected_gpu_id', -1))
if selected_gpu_id < 0:
    result = {
        'selected_gpu_id': -1,
        'lease_id': '',
        'fallback_reason': 'no_candidate_after_selector_filter',
        'recommended_gpu_id': recommended_gpu_id,
        'selector_payload': payload,
    }
    Path('$GPU_SELECT_JSON').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(-1)
    print('')
    raise SystemExit(23)

selected_row = None
recommended_row = None
for row in payload.get('gpus', []) if isinstance(payload.get('gpus', []), list) else []:
    if int(row.get('gpu_id', -1)) == selected_gpu_id:
        selected_row = row
    if int(row.get('gpu_id', -1)) == recommended_gpu_id:
        recommended_row = row

fallback_reason = ''
if selected_gpu_id != recommended_gpu_id:
    if recommended_gpu_id < 0:
        fallback_reason = 'recommended_gpu_missing'
    elif isinstance(recommended_row, dict):
        fallback_reason = str(recommended_row.get('selected_reason', 'recommended_gpu_not_selected'))
    else:
        fallback_reason = 'recommended_gpu_not_observed'

lease = acquire_lease(
    gpu_id=selected_gpu_id,
    owner='tracewm_stage2_core_mainline_train_20260408',
    ttl_seconds=12 * 3600,
    lease_path='$GPU_LEASE_PATH',
)

result = {
    'selected_gpu_id': selected_gpu_id,
    'avg_gpu_util': float((selected_row or {}).get('avg_gpu_util', 0.0)),
    'avg_mem_util': float((selected_row or {}).get('avg_mem_util', 0.0)),
    'free_mem_gb': float((selected_row or {}).get('free_mem_gb', 0.0)),
    'lease_id': str(lease.get('lease_id', '')),
    'fallback_reason': str(fallback_reason),
    'recommended_gpu_id': recommended_gpu_id,
    'selected_reason': str((selected_row or {}).get('selected_reason', '')),
    'selector_payload': payload,
    'lease': lease,
}
Path('$GPU_SELECT_JSON').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(int(result['selected_gpu_id']))
print(str(result['lease_id']))
PY
)

mapfile -t GPU_INFO <<<"$GPU_DECISION"
SELECTED_GPU="${GPU_INFO[0]:--1}"
LEASE_ID="${GPU_INFO[1]:-}"

if [[ "$SELECTED_GPU" -lt 0 ]]; then
  echo "[stage2-core-mainline] gpu_selection_failed"
  exit 3
fi

GPU_METADATA_JSON=$(
  "$PYTHON_BIN" - <<PY
import json
p=json.load(open('$GPU_SELECT_JSON','r',encoding='utf-8'))
print(json.dumps(p, ensure_ascii=False))
PY
)
GPU_METADATA_STR=$(
  "$PYTHON_BIN" - <<PY
import json
p=json.load(open('$GPU_SELECT_JSON','r',encoding='utf-8'))
fields=['selected_gpu_id','avg_gpu_util','avg_mem_util','free_mem_gb','lease_id','fallback_reason']
print(';'.join(f"{k}={p.get(k,'')}" for k in fields))
PY
)

export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
export TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA="$GPU_METADATA_STR"
export TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON="$GPU_METADATA_JSON"

echo "[stage2-core-mainline] selected_gpu=$SELECTED_GPU lease_id=$LEASE_ID"
echo "[stage2-core-mainline] gpu_metadata=$GPU_METADATA_STR"

cleanup() {
  local rc=$?
  if [[ -n "$LEASE_ID" ]]; then
    "$PYTHON_BIN" - <<PY >/dev/null 2>&1 || true
from stwm.infra.gpu_lease import release_lease
release_lease(lease_id='$LEASE_ID', lease_path='$GPU_LEASE_PATH')
PY
  fi
  exit $rc
}
trap cleanup EXIT INT TERM

echo "[stage2-core-mainline] step=run_stage2_core_mainline_training"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2_stage2/trainers/train_tracewm_stage2_smalltrain.py" \
  --stage2-contract-path "$STAGE2_CONTRACT_JSON" \
  --recommended-runtime-json "$RUNTIME_JSON" \
  --use-recommended-runtime \
  --stage1-backbone-checkpoint "$STAGE1_BEST_CKPT" \
  --dataset-names vspw vipseg \
  --train-split train \
  --val-split val \
  --max-samples-train 48 \
  --max-samples-val 24 \
  --batch-size 2 \
  --train-steps 1200 \
  --eval-interval 100 \
  --eval-max-batches 8 \
  --save-every-n-steps 1000 \
  --semantic-source-mainline crop_visual_encoder \
  --legacy-semantic-source hand_crafted_stats \
  --semantic-crop-size 64 \
  --output-dir "$CHECKPOINT_DIR" \
  --auto-resume-latest \
  --run-name "$RUN_NAME" \
  --run-summary-json "$RAW_RUN_JSON"

echo "[stage2-core-mainline] step=summarize_core_mainline_training"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2_stage2/tools/summarize_stage2_core_mainline_train_round.py" \
  --input-run-json "$RAW_RUN_JSON" \
  --progress-json "$PROGRESS_JSON" \
  --final-json "$FINAL_JSON" \
  --results-md "$RESULTS_MD"

echo "[stage2-core-mainline] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage2-core-mainline] progress_json=$PROGRESS_JSON"
echo "[stage2-core-mainline] final_json=$FINAL_JSON"
echo "[stage2-core-mainline] checkpoint_dir=$CHECKPOINT_DIR"
echo "[stage2-core-mainline] results_md=$RESULTS_MD"
