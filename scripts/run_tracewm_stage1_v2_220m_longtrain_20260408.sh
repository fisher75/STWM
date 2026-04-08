#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage1_v2_220m_longtrain_continue_${DATE_TAG}.log"
PROTOCOL_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_220M_LONGTRAIN_CONTINUATION_PROTOCOL_${DATE_TAG}.md"
CONTRACT_PATH="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_${DATE_TAG}.json"
RUNTIME_JSON="$WORK_ROOT/reports/stage1_v2_recommended_runtime_${DATE_TAG}.json"
MINISPLIT_PATH="/home/chen034/workspace/data/_manifests/stage1_minisplits_${DATE_TAG}.json"
FREEZE_CMP_JSON="$WORK_ROOT/reports/stage1_v2_220m_mainline_freeze_comparison_${DATE_TAG}.json"

CHECKPOINT_DIR="$WORK_ROOT/outputs/checkpoints/stage1_v2_longtrain_220m_mainline_${DATE_TAG}"
OUTPUT_DIR="$WORK_ROOT/outputs/training/stage1_v2_longtrain_220m_mainline_${DATE_TAG}"
PROGRESS_JSON="$WORK_ROOT/reports/stage1_v2_220m_longtrain_progress_${DATE_TAG}.json"
FINAL_JSON="$WORK_ROOT/reports/stage1_v2_220m_longtrain_final_${DATE_TAG}.json"
CONFIRM_10000_JSON="$WORK_ROOT/reports/stage1_v2_220m_longtrain_10000_confirmation_${DATE_TAG}.json"
RESULTS_MD="$WORK_ROOT/docs/STAGE1_V2_220M_LONGTRAIN_RESULTS_${DATE_TAG}.md"
SUMMARY_JSON="$WORK_ROOT/reports/stage1_v2_220m_longtrain_summary_${DATE_TAG}.json"
PERF_JSON="$WORK_ROOT/reports/stage1_v2_220m_longtrain_step_timing_${DATE_TAG}.json"
GPU_SELECT_JSON="$WORK_ROOT/reports/stage1_v2_220m_longtrain_continue_gpu_selection_${DATE_TAG}.json"
GPU_LEASE_PATH="$WORK_ROOT/reports/stage1_v2_gpu_lease_${DATE_TAG}.json"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$CHECKPOINT_DIR" "$OUTPUT_DIR"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage1-v2-longtrain-continue] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-longtrain] python=$PYTHON_BIN"

if [[ ! -f "$PROTOCOL_DOC" ]]; then
  echo "[stage1-v2-longtrain] missing_protocol_doc"
  exit 2
fi
if [[ ! -f "$RUNTIME_JSON" ]]; then
  echo "[stage1-v2-longtrain] missing_recommended_runtime_json"
  exit 3
fi
if [[ ! -f "$MINISPLIT_PATH" ]]; then
  echo "[stage1-v2-longtrain] missing_stage1_minisplit_json"
  exit 4
fi
if [[ ! -f "$FREEZE_CMP_JSON" ]]; then
  echo "[stage1-v2-longtrain] missing_freeze_comparison_json"
  exit 5
fi
if [[ ! -f "$CHECKPOINT_DIR/latest.pt" ]]; then
  echo "[stage1-v2-longtrain] missing_latest_checkpoint=$CHECKPOINT_DIR/latest.pt"
  exit 8
fi

FREEZE_DECISION=$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$FREEZE_CMP_JSON', 'r', encoding='utf-8'))
print(str(p.get('final_stage1_backbone_decision', '')))
PY
)
FREEZE_DECISION="$(echo "$FREEZE_DECISION" | tail -n 1 | tr -d '[:space:]')"
if [[ "$FREEZE_DECISION" != "freeze_220m_as_stage1_backbone" ]]; then
  echo "[stage1-v2-longtrain] unexpected_freeze_decision=$FREEZE_DECISION"
  exit 6
fi

echo "[stage1-v2-longtrain] step=select_gpu_with_selector_and_lease"
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
    sample_count=12,
    interval_sec=2.0,
    lease_path='$GPU_LEASE_PATH',
)

selected_gpu_id = int(payload.get('selected_gpu_id', -1))
if selected_gpu_id < 0:
    result = {
        'selected_gpu_id': -1,
        'avg_gpu_util': None,
        'avg_mem_util': None,
        'free_mem_gb': None,
        'lease_id': '',
        'fallback_reason': 'no_candidate_after_selector_filter',
        'required_mem_gb': required_mem_gb,
        'safety_margin_gb': safety_margin_gb,
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
    owner='tracewm_stage1_v2_220m_longtrain_continue_20260408',
    ttl_seconds=10 * 3600,
    lease_path='$GPU_LEASE_PATH',
)

result = {
    'selected_gpu_id': selected_gpu_id,
    'avg_gpu_util': float((selected_row or {}).get('avg_gpu_util', 0.0)),
    'avg_mem_util': float((selected_row or {}).get('avg_mem_util', 0.0)),
    'free_mem_gb': float((selected_row or {}).get('free_mem_gb', 0.0)),
    'lease_id': str(lease.get('lease_id', '')),
    'fallback_reason': str(fallback_reason),
    'required_mem_gb': required_mem_gb,
    'safety_margin_gb': safety_margin_gb,
    'recommended_gpu_id': recommended_gpu_id,
    'selected_reason': str((selected_row or {}).get('selected_reason', '')),
    'lease': lease,
    'selector_payload': payload,
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
  echo "[stage1-v2-longtrain] gpu_selection_failed"
  exit 7
fi

GPU_METADATA_STR=$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$GPU_SELECT_JSON', 'r', encoding='utf-8'))
fields = ['selected_gpu_id', 'avg_gpu_util', 'avg_mem_util', 'free_mem_gb', 'lease_id', 'fallback_reason']
print(';'.join(f"{k}={p.get(k, '')}" for k in fields))
PY
)
GPU_METADATA_JSON=$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$GPU_SELECT_JSON', 'r', encoding='utf-8'))
print(json.dumps(p, ensure_ascii=False))
PY
)

export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
export TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA="$GPU_METADATA_STR"
export TRACEWM_STAGE1_V2_GPU_SELECTION_METADATA_JSON="$GPU_METADATA_JSON"

echo "[stage1-v2-longtrain] selected_gpu=$SELECTED_GPU lease_id=$LEASE_ID"
echo "[stage1-v2-longtrain] gpu_metadata=$GPU_METADATA_STR"

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

"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/trainers/train_tracewm_stage1_v2.py" \
  --run-name stage1_v2_longtrain_220m_mainline_continue_10000 \
  --ablation-tag stage1_v2_longtrain_220m_mainline_continue_10000 \
  --run-metadata-note "continuation from latest.pt to 10000 under frozen Stage1-v2 scope" \
  --contract-path "$CONTRACT_PATH" \
  --recommended-runtime-json "$RUNTIME_JSON" \
  --use-recommended-runtime \
  --stage1-minisplit-path "$MINISPLIT_PATH" \
  --data-root /home/chen034/workspace/data \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --summary-json "$SUMMARY_JSON" \
  --progress-json "$PROGRESS_JSON" \
  --final-json "$FINAL_JSON" \
  --results-md "$RESULTS_MD" \
  --perf-step-timing-json "$PERF_JSON" \
  --model-preset prototype_220m \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --batch-size 2 \
  --coord-weight 1.2 \
  --visibility-weight 0.8 \
  --residual-weight 0.25 \
  --velocity-weight 0.25 \
  --endpoint-weight 0.1 \
  --enable-visibility \
  --resume-from "$CHECKPOINT_DIR/latest.pt" \
  --train-steps 10000 \
  --eval-interval 1000 \
  --eval-steps 16 \
  --save-every-n-steps 1000 \
  --max-samples-per-dataset 128 \
  --max-samples-per-dataset-val 64 \
  --max-tokens 64 \
  --seed 20260408

"$PYTHON_BIN" - <<PY
import json
from datetime import datetime, timezone
from pathlib import Path

BASE_PRIMARY_5000 = 0.2102444279531349
final_path = Path('$FINAL_JSON')
progress_path = Path('$PROGRESS_JSON')
confirm_path = Path('$CONFIRM_10000_JSON')
checkpoint_dir = Path('$CHECKPOINT_DIR')

final_payload = json.loads(final_path.read_text(encoding='utf-8'))
progress_payload = json.loads(progress_path.read_text(encoding='utf-8'))

best_metrics = final_payload.get('best_metric_so_far', {}).get('metrics', {})
best_primary = float(best_metrics.get('free_rollout_endpoint_l2', 1e9))
steps_completed = int((final_payload.get('training_budget', {}) or {}).get('optimizer_steps_completed', 0))

history = progress_payload.get('eval_history', []) if isinstance(progress_payload.get('eval_history', []), list) else []
improving_after_5000 = False
for item in history:
    if not isinstance(item, dict):
        continue
    step = int(item.get('global_step', 0) or 0)
    metrics = item.get('metrics', {}) if isinstance(item.get('metrics', {}), dict) else {}
    primary = float(metrics.get('free_rollout_endpoint_l2', 1e9))
    if step > 5000 and primary < BASE_PRIMARY_5000 - 1e-12:
        improving_after_5000 = True
        break
if best_primary < BASE_PRIMARY_5000 - 1e-12:
    improving_after_5000 = True

required_steps = [
    'step_0006000.pt',
    'step_0007000.pt',
    'step_0008000.pt',
    'step_0009000.pt',
    'step_0010000.pt',
]
required_present = {name: (checkpoint_dir / name).exists() for name in required_steps}
required_all_present = all(required_present.values())

whether_ready = bool(
    steps_completed >= 10000
    and required_all_present
    and best_primary <= BASE_PRIMARY_5000 + 1e-12
)

if whether_ready:
    next_step_choice = 'freeze_stage1_and_prepare_stage2'
elif improving_after_5000:
    next_step_choice = 'continue_to_15000_from_latest'
else:
    next_step_choice = 'do_one_targeted_stage1_fix'

payload = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'run_name': 'stage1_v2_longtrain_220m_mainline_continue_10000',
    'baseline_best_primary_metric_at_5000': BASE_PRIMARY_5000,
    'best_primary_metric_at_10000': best_primary,
    'whether_curve_still_improving_after_5000': bool(improving_after_5000),
    'whether_stage1_backbone_is_now_fully_ready': bool(whether_ready),
    'next_step_choice': next_step_choice,
    'allowed_next_step_choice': [
        'freeze_stage1_and_prepare_stage2',
        'continue_to_15000_from_latest',
        'do_one_targeted_stage1_fix',
    ],
    'training_budget': final_payload.get('training_budget', {}),
    'selection_policy': final_payload.get('selection_policy', {}),
    'checkpoint_dir': str(checkpoint_dir),
    'required_checkpoint_presence_6000_to_10000': required_present,
    'evidence': {
        'progress_json': str(progress_path),
        'final_json': str(final_path),
    },
}

confirm_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[stage1-v2-longtrain] confirmation_10000_json={confirm_path}")
print(f"[stage1-v2-longtrain] best_primary_metric_at_10000={best_primary}")
print(f"[stage1-v2-longtrain] whether_curve_still_improving_after_5000={improving_after_5000}")
print(f"[stage1-v2-longtrain] next_step_choice={next_step_choice}")
PY

echo "[stage1-v2-longtrain-continue] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-longtrain] progress_json=$PROGRESS_JSON"
echo "[stage1-v2-longtrain] final_json=$FINAL_JSON"
echo "[stage1-v2-longtrain] confirmation_10000_json=$CONFIRM_10000_JSON"
echo "[stage1-v2-longtrain] checkpoint_dir=$CHECKPOINT_DIR"
