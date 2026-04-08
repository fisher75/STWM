#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/tracewm_stage1_v2_${DATE_TAG}.log"
POINT_INDEX="$DATA_ROOT/_manifests/stage1_v2_pointodyssey_cache_index_${DATE_TAG}.json"
KUBRIC_INDEX="$DATA_ROOT/_manifests/stage1_v2_kubric_cache_index_${DATE_TAG}.json"
POINT_SKIPPED_MANIFEST="$DATA_ROOT/_manifests/stage1_v2_pointodyssey_skipped_${DATE_TAG}.json"
CONTRACT_PATH="$DATA_ROOT/_manifests/stage1_v2_trace_cache_contract_${DATE_TAG}.json"
AUDIT_REPORT="$WORK_ROOT/reports/stage1_v2_trace_cache_audit_${DATE_TAG}.json"
PARAM_REPORT="$WORK_ROOT/reports/tracewm_stage1_v2_param_budget_${DATE_TAG}.json"
TRAIN_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_v2_train_summary_${DATE_TAG}.json"
G1G5_REPORT="$WORK_ROOT/reports/tracewm_stage1_v2_g1_g5_${DATE_TAG}.json"
G1G5_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_G1_G5_${DATE_TAG}.md"
ABLATION_STATE_REPORT="$WORK_ROOT/reports/stage1_v2_ablation_state_${DATE_TAG}.json"
ABLATION_BACKBONE_REPORT="$WORK_ROOT/reports/stage1_v2_ablation_backbone_${DATE_TAG}.json"
ABLATION_LOSSES_REPORT="$WORK_ROOT/reports/stage1_v2_ablation_losses_${DATE_TAG}.json"
FINAL_COMPARISON_REPORT="$WORK_ROOT/reports/stage1_v2_final_comparison_${DATE_TAG}.json"
RESULTS_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_RESULTS_${DATE_TAG}.md"
KICKOFF_REPORT="$WORK_ROOT/reports/tracewm_stage1_v2_kickoff_report_${DATE_TAG}.json"
KICKOFF_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_V2_KICKOFF_REPORT_${DATE_TAG}.md"

POINT_CACHE_ROOT="$DATA_ROOT/_cache/tracewm_stage1_v2/pointodyssey"
KUBRIC_CACHE_ROOT="$DATA_ROOT/_cache/tracewm_stage1_v2/kubric"

POINT_MAX_SCENES_PER_SPLIT="${POINT_MAX_SCENES_PER_SPLIT:-12}"
POINT_MAX_ANNO_BYTES="${POINT_MAX_ANNO_BYTES:-600000000}"
KUBRIC_MAX_SCENES="${KUBRIC_MAX_SCENES:-128}"
KUBRIC_FIRST_WAVE_MODE="${KUBRIC_FIRST_WAVE_MODE:-panning_raw_first_wave}"

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
echo "[stage1-v2] pointodyssey_first_wave_splits=train,val"
echo "[stage1-v2] kubric_first_wave_mode=$KUBRIC_FIRST_WAVE_MODE"

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
  --max-scenes-per-split "$POINT_MAX_SCENES_PER_SPLIT" \
  --splits train val \
  --skipped-manifest-json "$POINT_SKIPPED_MANIFEST"

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
  --min-visible-pixels 1 \
  --first-wave-mode "$KUBRIC_FIRST_WAVE_MODE"

echo "[stage1-v2] step=P0_audit_and_contract"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm_v2/tools/audit_trace_cache.py" \
  --point-index "$POINT_INDEX" \
  --kubric-index "$KUBRIC_INDEX" \
  --point-skipped-manifest "$POINT_SKIPPED_MANIFEST" \
  --kubric-first-wave-mode "$KUBRIC_FIRST_WAVE_MODE" \
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

echo "[stage1-v2] step=first_wave_artifacts"
"$PYTHON_BIN" - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


work = Path('/home/chen034/workspace/stwm')
data = Path('/home/chen034/workspace/data')

audit_path = work / 'reports/stage1_v2_trace_cache_audit_20260408.json'
contract_path = data / '_manifests/stage1_v2_trace_cache_contract_20260408.json'
point_skipped_path = data / '_manifests/stage1_v2_pointodyssey_skipped_20260408.json'
train_path = work / 'reports/tracewm_stage1_v2_train_summary_20260408.json'
g1g5_path = work / 'reports/tracewm_stage1_v2_g1_g5_20260408.json'

state_out = work / 'reports/stage1_v2_ablation_state_20260408.json'
backbone_out = work / 'reports/stage1_v2_ablation_backbone_20260408.json'
losses_out = work / 'reports/stage1_v2_ablation_losses_20260408.json'
final_out = work / 'reports/stage1_v2_final_comparison_20260408.json'
results_md = work / 'docs/TRACEWM_STAGE1_V2_RESULTS_20260408.md'

audit = json.loads(audit_path.read_text(encoding='utf-8'))
contract = json.loads(contract_path.read_text(encoding='utf-8'))
point_skipped = json.loads(point_skipped_path.read_text(encoding='utf-8'))
train = json.loads(train_path.read_text(encoding='utf-8'))
g1g5 = json.loads(g1g5_path.read_text(encoding='utf-8'))

runs = g1g5.get('runs', []) if isinstance(g1g5.get('runs', []), list) else []
run_by_name = {str(r.get('name', '')): r for r in runs if isinstance(r, dict)}


def total_loss(name: str) -> float:
    run = run_by_name.get(name, {})
    fm = run.get('final_metrics', {}) if isinstance(run.get('final_metrics', {}), dict) else {}
    return float(fm.get('total_loss', 1e9))


state_candidates = []
for lane in ['G1', 'G2', 'G3']:
    if lane in run_by_name:
        state_candidates.append(
            {
                'lane': lane,
                'total_loss': total_loss(lane),
                'note': str(run_by_name[lane].get('note', '')),
            }
        )
state_selected = min(state_candidates, key=lambda x: x['total_loss']) if state_candidates else None
state_payload = {
    'generated_at_utc': now_iso(),
    'ablation_type': 'state',
    'candidates': state_candidates,
    'selected': state_selected,
    'evidence': {
        'g1_g5_report': str(g1g5_path),
    },
}
state_out.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding='utf-8')

model = train.get('model', {}) if isinstance(train.get('model', {}), dict) else {}
backbone_payload = {
    'generated_at_utc': now_iso(),
    'ablation_type': 'backbone',
    'first_wave_status': 'single_backbone_declared_no_mixing',
    'selected': {
        'model_preset': str(model.get('preset', 'unknown')),
        'estimated_parameter_count': int(model.get('estimated_parameter_count', 0)),
        'parameter_count': int(model.get('parameter_count', 0)),
        'target_220m_range_pass': bool(model.get('target_220m_range_pass', False)),
    },
    'evidence': {
        'mainline_train_summary': str(train_path),
    },
}
backbone_out.write_text(json.dumps(backbone_payload, ensure_ascii=False, indent=2), encoding='utf-8')

loss_rows = []
for lane in ['G1', 'G2', 'G3', 'G4', 'G5']:
    if lane not in run_by_name:
        continue
    fm = run_by_name[lane].get('final_metrics', {}) if isinstance(run_by_name[lane].get('final_metrics', {}), dict) else {}
    loss_rows.append(
        {
            'lane': lane,
            'total_loss': float(fm.get('total_loss', 0.0)),
            'coord_loss': float(fm.get('coord_loss', 0.0)),
            'visibility_loss': float(fm.get('visibility_loss', 0.0)),
            'residual_loss': float(fm.get('residual_loss', 0.0)),
            'velocity_loss': float(fm.get('velocity_loss', 0.0)),
            'endpoint_loss': float(fm.get('endpoint_loss', 0.0)),
        }
    )

selected_lane = str(g1g5.get('mainline_recommendation', {}).get('selected', 'none'))
losses_payload = {
    'generated_at_utc': now_iso(),
    'ablation_type': 'losses',
    'rows': loss_rows,
    'selected_lane': selected_lane,
    'selection_metric': 'min_total_loss',
    'selected_total_loss': float(g1g5.get('mainline_recommendation', {}).get('selected_total_loss', 0.0) or 0.0),
    'evidence': {
        'g1_g5_report': str(g1g5_path),
    },
}
losses_out.write_text(json.dumps(losses_payload, ensure_ascii=False, indent=2), encoding='utf-8')

kubric_mode = str(
    contract.get('summary', {}).get('kubric_first_wave_mode', '')
    if isinstance(contract.get('summary', {}), dict)
    else ''
)

final_payload = {
    'generated_at_utc': now_iso(),
    'p0_trace_cache_ready': bool(audit.get('p0_trace_cache_ready', False)),
    'pointodyssey_first_wave_splits': ['train', 'val'],
    'pointodyssey_skipped_manifest': str(point_skipped_path),
    'pointodyssey_skipped_scene_count': int(point_skipped.get('skipped_scene_count', 0)),
    'kubric_first_wave_mode': kubric_mode,
    'selected_state_lane': state_selected['lane'] if state_selected else 'none',
    'selected_losses_lane': selected_lane,
    'selected_backbone': backbone_payload['selected'],
    'evidence': {
        'contract': str(contract_path),
        'audit': str(audit_path),
        'ablation_state': str(state_out),
        'ablation_backbone': str(backbone_out),
        'ablation_losses': str(losses_out),
    },
}
final_out.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
    '# TRACEWM Stage1 v2 First-Wave Results',
    '',
    f"- generated_at_utc: {final_payload['generated_at_utc']}",
    f"- p0_trace_cache_ready: {final_payload['p0_trace_cache_ready']}",
    '- pointodyssey_first_wave_splits: train,val',
    f"- pointodyssey_skipped_manifest: {point_skipped_path}",
    f"- pointodyssey_skipped_scene_count: {final_payload['pointodyssey_skipped_scene_count']}",
    f"- kubric_first_wave_mode: {final_payload['kubric_first_wave_mode']}",
    f"- selected_state_lane: {final_payload['selected_state_lane']}",
    f"- selected_losses_lane: {final_payload['selected_losses_lane']}",
    '',
    '## Required Artifacts',
    f"- contract: {contract_path}",
    f"- audit: {audit_path}",
    f"- pointodyssey_skipped_manifest: {point_skipped_path}",
    f"- ablation_state: {state_out}",
    f"- ablation_backbone: {backbone_out}",
    f"- ablation_losses: {losses_out}",
    f"- final_comparison: {final_out}",
]
results_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

print(f"[stage1-v2] wrote {state_out}")
print(f"[stage1-v2] wrote {backbone_out}")
print(f"[stage1-v2] wrote {losses_out}")
print(f"[stage1-v2] wrote {final_out}")
print(f"[stage1-v2] wrote {results_md}")
PY

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
