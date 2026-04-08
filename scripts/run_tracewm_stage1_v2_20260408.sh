#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage1_v2_${DATE_TAG}.log"
POINT_INDEX="$DATA_ROOT/_manifests/stage1_v2_pointodyssey_cache_index_${DATE_TAG}.json"
KUBRIC_INDEX="$DATA_ROOT/_manifests/stage1_v2_kubric_cache_index_${DATE_TAG}.json"
CONTRACT_PATH="$DATA_ROOT/_manifests/stage1_v2_trace_cache_contract_${DATE_TAG}.json"
AUDIT_REPORT="$WORK_ROOT/reports/stage1_v2_trace_cache_audit_${DATE_TAG}.json"
PARAM_REPORT="$WORK_ROOT/reports/tracewm_stage1_v2_param_budget_${DATE_TAG}.json"
TRAIN_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_v2_train_summary_${DATE_TAG}.json"
G1G5_REPORT="$WORK_ROOT/reports/tracewm_stage1_v2_g1_g5_${DATE_TAG}.json"
G1G5_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_G1_G5_${DATE_TAG}.md"
KICKOFF_REPORT="$WORK_ROOT/reports/tracewm_stage1_v2_kickoff_report_${DATE_TAG}.json"
KICKOFF_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_KICKOFF_REPORT_${DATE_TAG}.md"

POINT_CACHE_ROOT="$DATA_ROOT/_cache/tracewm_stage1_v2/pointodyssey"
KUBRIC_CACHE_ROOT="$DATA_ROOT/_cache/tracewm_stage1_v2/kubric"

POINT_MAX_SCENES_PER_SPLIT="${POINT_MAX_SCENES_PER_SPLIT:-12}"
POINT_MAX_ANNO_BYTES="${POINT_MAX_ANNO_BYTES:-600000000}"
KUBRIC_MAX_SCENES="${KUBRIC_MAX_SCENES:-128}"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$WORK_ROOT/outputs/training"

export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage1-v2] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2] python=$PYTHON_BIN"
echo "[stage1-v2] work_root=$WORK_ROOT"
echo "[stage1-v2] data_root=$DATA_ROOT"
echo "[stage1-v2] point_max_scenes_per_split=$POINT_MAX_SCENES_PER_SPLIT"
echo "[stage1-v2] point_max_anno_bytes=$POINT_MAX_ANNO_BYTES"
echo "[stage1-v2] kubric_max_scenes=$KUBRIC_MAX_SCENES"

echo "[stage1-v2] step=P0_build_pointodyssey_cache"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/build_pointodyssey_trace_cache.py" \
  --data-root "$DATA_ROOT" \
  --cache-root "$POINT_CACHE_ROOT" \
  --index-json "$POINT_INDEX" \
  --obs-len 8 \
  --fut-len 8 \
  --stride 8 \
  --max-clips-per-scene 2 \
  --max-tracks 96 \
  --min-valid-frames 12 \
  --max-anno-bytes "$POINT_MAX_ANNO_BYTES" \
  --max-scenes-per-split "$POINT_MAX_SCENES_PER_SPLIT"

echo "[stage1-v2] step=P0_build_kubric_cache"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/build_kubric_trace_cache.py" \
  --data-root "$DATA_ROOT" \
  --cache-root "$KUBRIC_CACHE_ROOT" \
  --index-json "$KUBRIC_INDEX" \
  --obs-len 8 \
  --fut-len 8 \
  --stride 8 \
  --max-clips-per-scene 1 \
  --max-scenes "$KUBRIC_MAX_SCENES" \
  --max-instances 32 \
  --min-valid-frames 6 \
  --min-visible-pixels 1

echo "[stage1-v2] step=P0_audit_and_contract"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/audit_trace_cache.py" \
  --point-index "$POINT_INDEX" \
  --kubric-index "$KUBRIC_INDEX" \
  --contract-out "$CONTRACT_PATH" \
  --audit-out "$AUDIT_REPORT" \
  --sample-per-dataset 64

echo "[stage1-v2] step=P0_gate"
"$PYTHON_BIN" - <<'PY'
import json, sys
p = '/home/chen034/workspace/stwm/reports/stage1_v2_trace_cache_audit_20260408.json'
audit = json.load(open(p, 'r', encoding='utf-8'))
ready = bool(audit.get('p0_trace_cache_ready', False))
print(f"[stage1-v2] p0_trace_cache_ready={ready}")
if not ready:
    sys.exit(23)
PY

echo "[stage1-v2] step=P2_param_budget"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/check_tracewm_v2_param_budget.py" \
  --output-json "$PARAM_REPORT"

echo "[stage1-v2] step=P1P2P3_mainline_train_smoke"
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
  --steps-per-epoch 20 \
  --batch-size 2 \
  --enable-visibility \
  --enable-residual \
  --enable-velocity \
  --output-dir "$WORK_ROOT/outputs/training/tracewm_stage1_v2_mainline_${DATE_TAG}" \
  --summary-json "$TRAIN_SUMMARY" \
  --results-md "$WORK_ROOT/docs/TRACEWM_STAGE1_V2_TRAIN_SUMMARY_${DATE_TAG}.md" \
  --ablation-tag mainline

echo "[stage1-v2] step=G1_G5_ablations"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/run_stage1_v2_ablation_g1_g5.py" \
  --python-bin "$PYTHON_BIN" \
  --work-root "$WORK_ROOT" \
  --contract-path "$CONTRACT_PATH" \
  --epochs 1 \
  --steps-per-epoch 12 \
  --batch-size 2 \
  --max-samples-per-dataset 128 \
  --max-tokens 64 \
  --report-json "$G1G5_REPORT" \
  --report-md "$G1G5_DOC"

echo "[stage1-v2] step=final_kickoff_summary"
"$PYTHON_BIN" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

work = Path('/home/chen034/workspace/stwm')
report_out = work / 'reports/tracewm_stage1_v2_kickoff_report_20260408.json'
doc_out = work / 'docs/TRACEWM_STAGE1_V2_KICKOFF_REPORT_20260408.md'

audit = json.load(open(work / 'reports/stage1_v2_trace_cache_audit_20260408.json', 'r', encoding='utf-8'))
param = json.load(open(work / 'reports/tracewm_stage1_v2_param_budget_20260408.json', 'r', encoding='utf-8'))
train = json.load(open(work / 'reports/tracewm_stage1_v2_train_summary_20260408.json', 'r', encoding='utf-8'))
grid = json.load(open(work / 'reports/tracewm_stage1_v2_g1_g5_20260408.json', 'r', encoding='utf-8'))

selected = grid.get('mainline_recommendation', {}).get('selected', 'none')
selected_loss = grid.get('mainline_recommendation', {}).get('selected_total_loss', None)

payload = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'p0_trace_cache_ready': bool(audit.get('p0_trace_cache_ready', False)),
    'pointodyssey_audit_status': audit.get('datasets', {}).get('pointodyssey', {}).get('status', 'unknown'),
    'kubric_audit_status': audit.get('datasets', {}).get('kubric', {}).get('status', 'unknown'),
    'p2_prototype_220m_estimated_params': param.get('presets', {}).get('prototype_220m', {}).get('estimated_parameter_count', 0),
    'p2_prototype_220m_in_range': bool(param.get('presets', {}).get('prototype_220m', {}).get('in_220m_range', False)),
    'mainline_smoke_total_loss': float(train.get('final_metrics', {}).get('total_loss', 0.0)),
    'g1_g5_selected': selected,
    'g1_g5_selected_total_loss': selected_loss,
    'evidence_paths': {
        'audit': str(work / 'reports/stage1_v2_trace_cache_audit_20260408.json'),
        'param_budget': str(work / 'reports/tracewm_stage1_v2_param_budget_20260408.json'),
        'mainline_train': str(work / 'reports/tracewm_stage1_v2_train_summary_20260408.json'),
        'g1_g5': str(work / 'reports/tracewm_stage1_v2_g1_g5_20260408.json'),
    },
}

report_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
    '# TRACEWM Stage1 v2 Kickoff Report',
    '',
    f"- generated_at_utc: {payload['generated_at_utc']}",
    f"- p0_trace_cache_ready: {payload['p0_trace_cache_ready']}",
    f"- pointodyssey_audit_status: {payload['pointodyssey_audit_status']}",
    f"- kubric_audit_status: {payload['kubric_audit_status']}",
    f"- p2_prototype_220m_estimated_params: {payload['p2_prototype_220m_estimated_params']}",
    f"- p2_prototype_220m_in_range: {payload['p2_prototype_220m_in_range']}",
    f"- mainline_smoke_total_loss: {payload['mainline_smoke_total_loss']:.6f}",
    f"- g1_g5_selected: {payload['g1_g5_selected']}",
    f"- g1_g5_selected_total_loss: {payload['g1_g5_selected_total_loss']}",
]

doc_out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(f"[stage1-v2] kickoff_report={report_out}")
print(f"[stage1-v2] kickoff_doc={doc_out}")
PY

echo "[stage1-v2] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1-v2] log=$LOG_PATH"
