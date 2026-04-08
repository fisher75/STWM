#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage1_v2_perf_${DATE_TAG}.log"
PRECHECK_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_PERF_PREFLIGHT_PROTOCOL_${DATE_TAG}.md"
PREFLIGHT_JSON="$WORK_ROOT/reports/stage1_v2_220m_preflight_${DATE_TAG}.json"
PREFLIGHT_DOC="$WORK_ROOT/docs/STAGE1_V2_220M_PREFLIGHT_${DATE_TAG}.md"

GPU_SELECT_JSON="$WORK_ROOT/reports/stage1_v2_gpu_selection_audit_${DATE_TAG}.json"
GPU_SELECT_DOC="$WORK_ROOT/docs/STAGE1_V2_GPU_SELECTION_AUDIT_${DATE_TAG}.md"
GPU_LEASE_PATH="$WORK_ROOT/reports/stage1_v2_gpu_lease_${DATE_TAG}.json"

DATALOADER_JSON="$WORK_ROOT/reports/stage1_v2_dataloader_profile_${DATE_TAG}.json"
DATALOADER_DOC="$WORK_ROOT/docs/STAGE1_V2_DATALOADER_PROFILE_${DATE_TAG}.md"
GPU_TELEMETRY_JSON="$WORK_ROOT/reports/stage1_v2_gpu_telemetry_${DATE_TAG}.json"

CONTRACT_PATH="$DATA_ROOT/_manifests/stage1_v2_trace_cache_contract_${DATE_TAG}.json"

DEBUG_SUMMARY_JSON="$WORK_ROOT/reports/stage1_v2_perf_debug_small_${DATE_TAG}.json"
DEBUG_SUMMARY_MD="$WORK_ROOT/docs/STAGE1_V2_PERF_DEBUG_SMALL_${DATE_TAG}.md"
DEBUG_TIMING_JSON="$WORK_ROOT/reports/stage1_v2_perf_step_timing_debug_small_${DATE_TAG}.json"

P220M_SUMMARY_JSON="$WORK_ROOT/reports/stage1_v2_perf_prototype_220m_${DATE_TAG}.json"
P220M_SUMMARY_MD="$WORK_ROOT/docs/STAGE1_V2_PERF_PROTOTYPE_220M_${DATE_TAG}.md"
P220M_TIMING_JSON="$WORK_ROOT/reports/stage1_v2_perf_step_timing_prototype_220m_${DATE_TAG}.json"

PROFILER_ROOT="$WORK_ROOT/outputs/profiler/stage1_v2_torch_profiler_${DATE_TAG}"
NSYS_ROOT="$WORK_ROOT/outputs/profiler/stage1_v2_nsys_${DATE_TAG}"
NSYS_STATUS_JSON="$WORK_ROOT/reports/stage1_v2_nsys_profile_${DATE_TAG}.json"

PERF_STEP_TIMING_JSON="$WORK_ROOT/reports/stage1_v2_perf_step_timing_${DATE_TAG}.json"
PERF_SUMMARY_JSON="$WORK_ROOT/reports/stage1_v2_perf_summary_${DATE_TAG}.json"
PERF_SUMMARY_MD="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_PERF_RESULTS_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$WORK_ROOT/outputs/training" "$WORK_ROOT/outputs/profiler"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage1-v2-perf] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-perf] python=$PYTHON_BIN"
echo "[stage1-v2-perf] preflight_doc=$PRECHECK_DOC"

if [[ ! -f "$PRECHECK_DOC" ]]; then
  echo "[stage1-v2-perf] missing_preflight_protocol_doc"
  exit 2
fi

echo "[stage1-v2-perf] step=B_preflight_220m"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/preflight_stage1_v2_220m.py" \
  --python-bin "$PYTHON_BIN" \
  --work-root "$WORK_ROOT" \
  --contract-path "$CONTRACT_PATH" \
  --steps 20 \
  --batch-size-candidates 2,1 \
  --num-workers 4 \
  --max-samples-per-dataset 128 \
  --report-json "$PREFLIGHT_JSON" \
  --report-md "$PREFLIGHT_DOC"

echo "[stage1-v2-perf] step=B_gate_preflight"
PREFLIGHT_PASS=$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$PREFLIGHT_JSON', 'r', encoding='utf-8'))
print('1' if bool(p.get('preflight_pass', False)) else '0')
PY
)
if [[ "$PREFLIGHT_PASS" != "1" ]]; then
  echo "[stage1-v2-perf] preflight_failed_stop_formal_perf_round"
  exit 31
fi

echo "[stage1-v2-perf] step=C_select_single_gpu"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/select_stage1_v2_single_gpu.py" \
  --required-mem-gb "${REQUIRED_MEM_GB:-40}" \
  --safety-margin-gb "${SAFETY_MARGIN_GB:-8}" \
  --sample-count 12 \
  --sample-interval-sec 2 \
  --lease-path "$GPU_LEASE_PATH" \
  --lease-owner "tracewm_stage1_v2_perf_${DATE_TAG}" \
  --lease-ttl-sec 28800 \
  --report-json "$GPU_SELECT_JSON" \
  --report-md "$GPU_SELECT_DOC"

SELECTED_GPU=$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$GPU_SELECT_JSON', 'r', encoding='utf-8'))
print(int(p.get('selected_gpu_id', -1)))
PY
)
LEASE_ID=$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$GPU_SELECT_JSON', 'r', encoding='utf-8'))
print(str(p.get('lease', {}).get('lease_id', '')))
PY
)

if [[ "$SELECTED_GPU" == "-1" ]]; then
  echo "[stage1-v2-perf] no_gpu_selected"
  exit 41
fi
export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
echo "[stage1-v2-perf] selected_gpu=$SELECTED_GPU"
echo "[stage1-v2-perf] lease_id=$LEASE_ID"

TELEMETRY_PID=""
cleanup() {
  local rc=$?

  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" >/dev/null 2>&1 || true
  fi

  if [[ -n "$LEASE_ID" ]]; then
    "$PYTHON_BIN" - <<PY >/dev/null 2>&1 || true
from stwm.infra.gpu_lease import release_lease
release_lease(lease_id='$LEASE_ID', lease_path='$GPU_LEASE_PATH')
PY
  fi

  echo "[stage1-v2-perf] cleanup_done rc=$rc"
  exit $rc
}
trap cleanup EXIT INT TERM

echo "[stage1-v2-perf] step=E_profile_dataloader"
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

eval "$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$DATALOADER_JSON', 'r', encoding='utf-8'))
cfg = p.get('best_config', {})
print(f"DL_NUM_WORKERS={int(cfg.get('num_workers', 0))}")
print(f"DL_PIN_MEMORY={1 if bool(cfg.get('pin_memory', False)) else 0}")
print(f"DL_PERSISTENT_WORKERS={1 if bool(cfg.get('persistent_workers', False)) else 0}")
print(f"DL_PREFETCH_FACTOR={int(cfg.get('prefetch_factor', 2) or 2)}")
PY
)"

echo "[stage1-v2-perf] dataloader_best workers=$DL_NUM_WORKERS pin_memory=$DL_PIN_MEMORY persistent_workers=$DL_PERSISTENT_WORKERS prefetch=$DL_PREFETCH_FACTOR"

DL_FLAGS=(--num-workers "$DL_NUM_WORKERS")
if [[ "$DL_PIN_MEMORY" == "1" ]]; then
  DL_FLAGS+=(--pin-memory)
fi
if [[ "$DL_NUM_WORKERS" -gt 0 ]]; then
  if [[ "$DL_PERSISTENT_WORKERS" == "1" ]]; then
    DL_FLAGS+=(--persistent-workers)
  fi
  DL_FLAGS+=(--prefetch-factor "$DL_PREFETCH_FACTOR")
fi

echo "[stage1-v2-perf] step=F1_start_gpu_telemetry"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/collect_stage1_v2_gpu_telemetry.py" \
  --output-json "$GPU_TELEMETRY_JSON" \
  --interval-sec 2 \
  --max-seconds 0 \
  --tag "tracewm_stage1_v2_perf_${DATE_TAG}" &
TELEMETRY_PID=$!
echo "[stage1-v2-perf] telemetry_pid=$TELEMETRY_PID"

P220M_BATCH_SIZE=$(
  "$PYTHON_BIN" - <<PY
import json
p = json.load(open('$PREFLIGHT_JSON', 'r', encoding='utf-8'))
bs = int(p.get('selected_batch_size', 1))
print(max(bs, 1))
PY
)

echo "[stage1-v2-perf] step=G_run_stage1_v2_perf_debug_small"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/trainers/train_tracewm_stage1_v2.py" \
  --contract-path "$CONTRACT_PATH" \
  --dataset-names pointodyssey kubric \
  --train-split train \
  --obs-len 8 \
  --fut-len 8 \
  --max-tokens 64 \
  --max-samples-per-dataset 128 \
  --model-preset debug_small \
  --epochs 1 \
  --steps-per-epoch 30 \
  --batch-size 2 \
  --enable-visibility \
  --enable-residual \
  --enable-velocity \
  --ablation-tag stage1_v2_perf_debug_small \
  --output-dir "$WORK_ROOT/outputs/training/stage1_v2_perf_debug_small_${DATE_TAG}" \
  --summary-json "$DEBUG_SUMMARY_JSON" \
  --results-md "$DEBUG_SUMMARY_MD" \
  --perf-step-timing-json "$DEBUG_TIMING_JSON" \
  "${DL_FLAGS[@]}"

echo "[stage1-v2-perf] step=G_run_stage1_v2_perf_prototype_220m"
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
  --steps-per-epoch 20 \
  --batch-size "$P220M_BATCH_SIZE" \
  --enable-visibility \
  --enable-residual \
  --enable-velocity \
  --ablation-tag stage1_v2_perf_prototype_220m \
  --output-dir "$WORK_ROOT/outputs/training/stage1_v2_perf_prototype_220m_${DATE_TAG}" \
  --summary-json "$P220M_SUMMARY_JSON" \
  --results-md "$P220M_SUMMARY_MD" \
  --perf-step-timing-json "$P220M_TIMING_JSON" \
  "${DL_FLAGS[@]}"

echo "[stage1-v2-perf] step=F1_stop_gpu_telemetry"
if [[ -n "$TELEMETRY_PID" ]]; then
  kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
  wait "$TELEMETRY_PID" >/dev/null 2>&1 || true
  TELEMETRY_PID=""
fi

echo "[stage1-v2-perf] step=F2_torch_profiler"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/profile_stage1_v2_torch_profiler.py" \
  --contract-path "$CONTRACT_PATH" \
  --preflight-json "$PREFLIGHT_JSON" \
  --output-root "$PROFILER_ROOT" \
  --max-steps 8 \
  --debug-batch-size 2 \
  --prototype-batch-size "$P220M_BATCH_SIZE" \
  --num-workers 0 \
  --max-samples-per-dataset 64

echo "[stage1-v2-perf] step=F3_nsys_profile"
bash "$WORK_ROOT/scripts/run_stage1_v2_nsys_profile.sh" \
  --python-bin "$PYTHON_BIN" \
  --work-root "$WORK_ROOT" \
  --contract-path "$CONTRACT_PATH" \
  --output-root "$NSYS_ROOT" \
  --status-json "$NSYS_STATUS_JSON" \
  --selected-gpu "$SELECTED_GPU"

echo "[stage1-v2-perf] step=H_summarize_perf"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/summarize_stage1_v2_perf.py" \
  --preflight-json "$PREFLIGHT_JSON" \
  --gpu-selection-json "$GPU_SELECT_JSON" \
  --dataloader-profile-json "$DATALOADER_JSON" \
  --gpu-telemetry-json "$GPU_TELEMETRY_JSON" \
  --debug-timing-json "$DEBUG_TIMING_JSON" \
  --prototype-timing-json "$P220M_TIMING_JSON" \
  --nsys-status-json "$NSYS_STATUS_JSON" \
  --perf-step-timing-json "$PERF_STEP_TIMING_JSON" \
  --summary-json "$PERF_SUMMARY_JSON" \
  --summary-md "$PERF_SUMMARY_MD"

echo "[stage1-v2-perf] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-perf] log=$LOG_PATH"
