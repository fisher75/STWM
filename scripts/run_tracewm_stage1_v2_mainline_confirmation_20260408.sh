#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage1_v2_mainline_confirmation_${DATE_TAG}.log"
PROTOCOL_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_MAINLINE_CONFIRMATION_PROTOCOL_${DATE_TAG}.md"
CONTRACT_PATH="/home/chen034/workspace/data/_manifests/stage1_v2_trace_cache_contract_${DATE_TAG}.json"
RUNTIME_JSON="$WORK_ROOT/reports/stage1_v2_recommended_runtime_${DATE_TAG}.json"
MINISPLIT_PATH="/home/chen034/workspace/data/_manifests/stage1_minisplits_${DATE_TAG}.json"
GAP_CMP_JSON="$WORK_ROOT/reports/stage1_v2_220m_gap_closure_comparison_${DATE_TAG}.json"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs"
export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage1-v2-confirm] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-confirm] python=$PYTHON_BIN"
echo "[stage1-v2-confirm] protocol_doc=$PROTOCOL_DOC"
echo "[stage1-v2-confirm] contract_path=$CONTRACT_PATH"
echo "[stage1-v2-confirm] runtime_json=$RUNTIME_JSON"
echo "[stage1-v2-confirm] minisplit_path=$MINISPLIT_PATH"

if [[ ! -f "$PROTOCOL_DOC" ]]; then
  echo "[stage1-v2-confirm] missing_protocol_doc"
  exit 2
fi

if [[ ! -f "$RUNTIME_JSON" ]]; then
  echo "[stage1-v2-confirm] missing_recommended_runtime_json"
  exit 3
fi

if [[ ! -f "$MINISPLIT_PATH" ]]; then
  echo "[stage1-v2-confirm] missing_stage1_minisplit_json"
  exit 4
fi

if [[ ! -f "$GAP_CMP_JSON" ]]; then
  echo "[stage1-v2-confirm] missing_gap_closure_comparison_json"
  exit 5
fi

GPU_DECISION=$(
  "$PYTHON_BIN" - <<PY
import json
import subprocess

p = json.load(open('$RUNTIME_JSON', 'r', encoding='utf-8'))
policy = p.get('selected_gpu_policy', {}) if isinstance(p.get('selected_gpu_policy', {}), dict) else {}
selected = int(policy.get('selected_gpu_id', -1))
required_gb = float(p.get('required_mem_gb', 40.0) or 40.0)
safety_gb = float(p.get('safety_margin_gb', 8.0) or 8.0)
need_mib = int((required_gb + safety_gb) * 1024.0)

rows = []
try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        text=True,
    )
    for line in out.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [x.strip() for x in line.split(',')]
        if len(parts) < 2:
            continue
        rows.append((int(parts[0]), int(float(parts[1]))))
except Exception:
    rows = []

chosen = -1
reason = 'recommended_gpu_missing'

if selected >= 0:
    selected_free = -1
    for idx, free_mib in rows:
        if idx == selected:
            selected_free = free_mib
            break
    if selected_free >= need_mib:
        chosen = selected
        reason = f'recommended_gpu_ok_free_mib={selected_free}'
    else:
        reason = f'recommended_gpu_insufficient_free_mib={selected_free}_need_mib={need_mib}'

if chosen < 0 and rows:
    feasible = [(idx, free_mib) for idx, free_mib in rows if free_mib >= need_mib]
    if feasible:
        best_idx, best_free = max(feasible, key=lambda x: x[1])
        chosen = int(best_idx)
        reason = reason + f';fallback_gpu_ok={best_idx};free_mib={best_free}'
    else:
        best_idx, best_free = max(rows, key=lambda x: x[1])
        chosen = int(best_idx)
        reason = reason + f';fallback_gpu_best_effort={best_idx};free_mib={best_free}'

print(chosen)
print(need_mib)
print(reason)
PY
)

mapfile -t GPU_INFO <<<"$GPU_DECISION"
SELECTED_GPU="${GPU_INFO[0]:--1}"
NEED_MIB="${GPU_INFO[1]:-0}"
GPU_REASON="${GPU_INFO[2]:-unknown}"

if [[ "$SELECTED_GPU" -ge 0 ]]; then
  export CUDA_VISIBLE_DEVICES="$SELECTED_GPU"
  echo "[stage1-v2-confirm] selected_gpu=$SELECTED_GPU need_mib=$NEED_MIB reason=$GPU_REASON"
else
  echo "[stage1-v2-confirm] gpu_selection_failed need_mib=$NEED_MIB reason=$GPU_REASON"
  exit 6
fi

echo "[stage1-v2-confirm] step=run_mainline_confirmation"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/run_stage1_v2_mainline_confirmation.py" \
  --contract-path "$CONTRACT_PATH" \
  --recommended-runtime-json "$RUNTIME_JSON" \
  --stage1-minisplit-path "$MINISPLIT_PATH" \
  --gap-closure-comparison-json "$GAP_CMP_JSON" \
  --runs-summary-json "$WORK_ROOT/reports/stage1_v2_mainline_confirmation_runs_${DATE_TAG}.json" \
  --comparison-json "$WORK_ROOT/reports/stage1_v2_mainline_confirmation_comparison_${DATE_TAG}.json" \
  --results-md "$WORK_ROOT/docs/STAGE1_V2_MAINLINE_CONFIRMATION_RESULTS_${DATE_TAG}.md" \
  --epochs 1 \
  --train-steps 96 \
  --eval-steps 12 \
  --max-samples-per-dataset-train 128 \
  --max-samples-per-dataset-val 64 \
  --max-tokens 64

echo "[stage1-v2-confirm] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2-confirm] log=$LOG_PATH"
