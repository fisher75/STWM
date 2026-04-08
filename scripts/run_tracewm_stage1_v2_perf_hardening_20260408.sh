#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage1_v2_perf_hardening_${DATE_TAG}.log"
CONTRACT_PATH="$DATA_ROOT/_manifests/stage1_v2_trace_cache_contract_${DATE_TAG}.json"

DATALOADER_JSON="$WORK_ROOT/reports/stage1_v2_dataloader_profile_${DATE_TAG}.json"
DATALOADER_DOC="$WORK_ROOT/docs/STAGE1_V2_DATALOADER_PROFILE_${DATE_TAG}.md"

RECOMMENDED_RUNTIME_JSON="$WORK_ROOT/reports/stage1_v2_recommended_runtime_${DATE_TAG}.json"
RECOMMENDED_RUNTIME_MD="$WORK_ROOT/docs/STAGE1_V2_RECOMMENDED_RUNTIME_${DATE_TAG}.md"

GPU_SELECT_JSON="$WORK_ROOT/reports/stage1_v2_gpu_selection_audit_${DATE_TAG}.json"
GPU_TELEMETRY_JSON="$WORK_ROOT/reports/stage1_v2_gpu_telemetry_${DATE_TAG}.json"
PREFLIGHT_JSON="$WORK_ROOT/reports/stage1_v2_220m_preflight_${DATE_TAG}.json"
DEBUG_TIMING_JSON="$WORK_ROOT/reports/stage1_v2_perf_step_timing_debug_small_${DATE_TAG}.json"
NSYS_STATUS_JSON="$WORK_ROOT/reports/stage1_v2_nsys_profile_${DATE_TAG}.json"

CONFIRM_TRAIN_SUMMARY_JSON="$WORK_ROOT/reports/stage1_v2_perf_confirmation_train_summary_${DATE_TAG}.json"
CONFIRM_TRAIN_DOC="$WORK_ROOT/docs/STAGE1_V2_PERF_CONFIRMATION_TRAIN_${DATE_TAG}.md"
CONFIRM_TIMING_JSON="$WORK_ROOT/reports/stage1_v2_perf_step_timing_confirmation_${DATE_TAG}.json"

PERF_STEP_TIMING_JSON="$WORK_ROOT/reports/stage1_v2_perf_step_timing_${DATE_TAG}.json"
PERF_SUMMARY_JSON="$WORK_ROOT/reports/stage1_v2_perf_summary_${DATE_TAG}.json"
PERF_SUMMARY_MD="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_PERF_RESULTS_${DATE_TAG}.md"

CONFIRMATION_JSON="$WORK_ROOT/reports/stage1_v2_perf_confirmation_${DATE_TAG}.json"
CONFIRMATION_MD="$WORK_ROOT/docs/STAGE1_V2_PERF_CONFIRMATION_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$WORK_ROOT/outputs/training"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage1-v2-perf-hardening] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-perf-hardening] python=$PYTHON_BIN"

echo "[stage1-v2-perf-hardening] step=C_refresh_dataloader_profile"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/profile_dataloader_stage1_v2.py" \
  --contract-path "$CONTRACT_PATH" \
  --dataset-names pointodyssey kubric \
  --split train \
  --obs-len 8 \
  --fut-len 8 \
  --max-tokens 64 \
  --max-samples-per-dataset 128 \
  --batch-size 2 \
  --warmup-batches 5 \
  --measure-batches 30 \
  --report-json "$DATALOADER_JSON" \
  --report-md "$DATALOADER_DOC"

echo "[stage1-v2-perf-hardening] step=E_export_recommended_runtime"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/export_stage1_v2_recommended_runtime.py" \
  --gpu-selection-json "$GPU_SELECT_JSON" \
  --dataloader-profile-json "$DATALOADER_JSON" \
  --preflight-json "$PREFLIGHT_JSON" \
  --debug-summary-json "$WORK_ROOT/reports/stage1_v2_perf_debug_small_${DATE_TAG}.json" \
  --perf-summary-json "$PERF_SUMMARY_JSON" \
  --report-json "$RECOMMENDED_RUNTIME_JSON" \
  --report-md "$RECOMMENDED_RUNTIME_MD"

eval "$(
  "$PYTHON_BIN" - <<PY
import json
r = json.load(open('$RECOMMENDED_RUNTIME_JSON', 'r', encoding='utf-8'))
policy = r.get('selected_gpu_policy', {}) if isinstance(r.get('selected_gpu_policy', {}), dict) else {}
print(f"SEL_GPU={int(policy.get('selected_gpu_id', -1))}")
print(f"REC_BATCH_P220M={int(r.get('recommended_batch_size_prototype_220m', 1) or 1)}")
print(f"REC_NUM_WORKERS={int(r.get('recommended_num_workers', 8))}")
print(f"REC_PIN_MEMORY={1 if bool(r.get('recommended_pin_memory', True)) else 0}")
print(f"REC_PERSISTENT_WORKERS={1 if bool(r.get('recommended_persistent_workers', True)) else 0}")
print(f"REC_PREFETCH_FACTOR={int(r.get('recommended_prefetch_factor', 4) or 4)}")
PY
)"

if [[ "$SEL_GPU" -ge 0 ]]; then
  export CUDA_VISIBLE_DEVICES="$SEL_GPU"
fi

echo "[stage1-v2-perf-hardening] selected_gpu=$SEL_GPU"
echo "[stage1-v2-perf-hardening] recommended_runtime workers=$REC_NUM_WORKERS pin_memory=$REC_PIN_MEMORY persistent_workers=$REC_PERSISTENT_WORKERS prefetch=$REC_PREFETCH_FACTOR batch_p220m=$REC_BATCH_P220M"

TELEMETRY_PID=""
cleanup() {
  local rc=$?
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" >/dev/null 2>&1 || true
  fi
  echo "[stage1-v2-perf-hardening] cleanup_done rc=$rc"
  exit $rc
}
trap cleanup EXIT INT TERM

echo "[stage1-v2-perf-hardening] step=F_collect_short_telemetry"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/collect_stage1_v2_gpu_telemetry.py" \
  --output-json "$GPU_TELEMETRY_JSON" \
  --interval-sec 2 \
  --max-seconds 24 \
  --tag "tracewm_stage1_v2_perf_hardening_${DATE_TAG}" &
TELEMETRY_PID=$!

echo "[stage1-v2-perf-hardening] step=F_short_confirmation_train_prototype_220m"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/trainers/train_tracewm_stage1_v2.py" \
  --contract-path "$CONTRACT_PATH" \
  --dataset-names pointodyssey kubric \
  --train-split train \
  --obs-len 8 \
  --fut-len 8 \
  --max-tokens 64 \
  --max-samples-per-dataset 128 \
  --model-preset prototype_220m \
  --epochs 1 \
  --steps-per-epoch 8 \
  --batch-size "$REC_BATCH_P220M" \
  --enable-visibility \
  --enable-residual \
  --enable-velocity \
  --ablation-tag stage1_v2_perf_confirmation_prototype_220m \
  --output-dir "$WORK_ROOT/outputs/training/stage1_v2_perf_confirmation_prototype_220m_${DATE_TAG}" \
  --summary-json "$CONFIRM_TRAIN_SUMMARY_JSON" \
  --results-md "$CONFIRM_TRAIN_DOC" \
  --perf-step-timing-json "$CONFIRM_TIMING_JSON"

if [[ -n "$TELEMETRY_PID" ]]; then
  wait "$TELEMETRY_PID" >/dev/null 2>&1 || true
  TELEMETRY_PID=""
fi

echo "[stage1-v2-perf-hardening] step=B_refresh_summary_with_prototype_priority"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/summarize_stage1_v2_perf.py" \
  --preflight-json "$PREFLIGHT_JSON" \
  --gpu-selection-json "$GPU_SELECT_JSON" \
  --dataloader-profile-json "$DATALOADER_JSON" \
  --gpu-telemetry-json "$GPU_TELEMETRY_JSON" \
  --debug-timing-json "$DEBUG_TIMING_JSON" \
  --prototype-timing-json "$CONFIRM_TIMING_JSON" \
  --nsys-status-json "$NSYS_STATUS_JSON" \
  --perf-step-timing-json "$PERF_STEP_TIMING_JSON" \
  --summary-json "$PERF_SUMMARY_JSON" \
  --summary-md "$PERF_SUMMARY_MD"

echo "[stage1-v2-perf-hardening] step=F_write_confirmation_report"
"$PYTHON_BIN" - <<PY
import json
from datetime import datetime, timezone

def now_iso():
    return datetime.now(timezone.utc).isoformat()

runtime = json.load(open('$RECOMMENDED_RUNTIME_JSON', 'r', encoding='utf-8'))
train = json.load(open('$CONFIRM_TRAIN_SUMMARY_JSON', 'r', encoding='utf-8'))
timing = json.load(open('$CONFIRM_TIMING_JSON', 'r', encoding='utf-8'))
summary = json.load(open('$PERF_SUMMARY_JSON', 'r', encoding='utf-8'))

args = train.get('args', {}) if isinstance(train.get('args', {}), dict) else {}
defaults_applied = (
    int(args.get('num_workers', -1)) == int(runtime.get('recommended_num_workers', 8))
    and bool(args.get('pin_memory', False)) == bool(runtime.get('recommended_pin_memory', True))
    and bool(args.get('persistent_workers', False)) == bool(runtime.get('recommended_persistent_workers', True))
    and int(args.get('prefetch_factor', -1)) == int(runtime.get('recommended_prefetch_factor', 4))
)

timing_ok = int(timing.get('num_steps', 0)) > 0 and float(timing.get('timing_stats', {}).get('step_time', {}).get('mean', 0.0)) > 0.0
primary_bottleneck = str(summary.get('primary_bottleneck', 'mixed_bottleneck'))
primary_source = str(summary.get('attribution_basis', {}).get('primary_source', 'unknown'))

payload = {
    'generated_at_utc': now_iso(),
    'single_gpu_only': bool(runtime.get('single_gpu_only', True)),
    'selected_gpu_id': int(runtime.get('selected_gpu_policy', {}).get('selected_gpu_id', -1)),
    'recommended_runtime_path': '$RECOMMENDED_RUNTIME_JSON',
    'confirmation_train_summary_path': '$CONFIRM_TRAIN_SUMMARY_JSON',
    'confirmation_timing_path': '$CONFIRM_TIMING_JSON',
    'updated_perf_summary_path': '$PERF_SUMMARY_JSON',
    'checks': {
        'trainer_default_runtime_matches_recommended': bool(defaults_applied),
        'timing_json_valid': bool(timing_ok),
        'summary_primary_source_is_prototype': bool(primary_source == 'prototype_220m'),
    },
    'updated_primary_bottleneck': primary_bottleneck,
    'updated_primary_source': primary_source,
}

with open('$CONFIRMATION_JSON', 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

lines = [
    '# Stage1-v2 Perf Hardening Confirmation',
    '',
    f"- generated_at_utc: {payload['generated_at_utc']}",
    f"- single_gpu_only: {payload['single_gpu_only']}",
    f"- selected_gpu_id: {payload['selected_gpu_id']}",
    f"- trainer_default_runtime_matches_recommended: {payload['checks']['trainer_default_runtime_matches_recommended']}",
    f"- timing_json_valid: {payload['checks']['timing_json_valid']}",
    f"- summary_primary_source_is_prototype: {payload['checks']['summary_primary_source_is_prototype']}",
    f"- updated_primary_bottleneck: {payload['updated_primary_bottleneck']}",
    f"- updated_primary_source: {payload['updated_primary_source']}",
    '',
    'This confirmation run is runtime hardening only and does not change scientific logic.',
]
with open('$CONFIRMATION_MD', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
PY

echo "[stage1-v2-perf-hardening] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-perf-hardening] log=$LOG_PATH"
