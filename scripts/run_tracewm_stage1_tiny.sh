#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"

LOG_PATH="$WORK_ROOT/logs/stage1_tracewm_pipeline_${DATE_TAG}.log"
CONTRACT_PATH="$DATA_ROOT/_manifests/stage1_data_contract_${DATE_TAG}.json"
MINISPLIT_PATH="$DATA_ROOT/_manifests/stage1_minisplits_${DATE_TAG}.json"

CONTRACT_SMOKE_REPORT="$WORK_ROOT/reports/stage1_contract_smoke_${DATE_TAG}.json"
LOADER_SMOKE_REPORT="$WORK_ROOT/reports/stage1_loader_smoke_${DATE_TAG}.json"
BATCH_SMOKE_REPORT="$WORK_ROOT/reports/stage1_batch_smoke_${DATE_TAG}.json"
EVAL_SMOKE_REPORT="$WORK_ROOT/reports/tracewm_stage1_eval_smoke_${DATE_TAG}.json"
TINY_SUMMARY_REPORT="$WORK_ROOT/reports/tracewm_stage1_tiny_summary_${DATE_TAG}.json"

VISUAL_DOC="$WORK_ROOT/docs/STAGE1_VISUAL_SMOKE_${DATE_TAG}.md"
EVAL_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_EVAL_SMOKE_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/reports" "$WORK_ROOT/outputs" "$WORK_ROOT/docs"

export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"
export STAGE1_DATA_CONTRACT_PATH="$CONTRACT_PATH"

exec > >(tee "$LOG_PATH") 2>&1

echo "[stage1] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1] work_root=$WORK_ROOT"
echo "[stage1] data_root=$DATA_ROOT"
echo "[stage1] python=$PYTHON_BIN"

echo "[stage1] step=build_stage1_contract"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/tools/audit_stage1_contract.py"

echo "[stage1] step=build_stage1_minisplits"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/tools/build_stage1_minisplits.py"

echo "[stage1] step=loader_smoke"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from typing import Any

import torch

from stwm.tracewm.datasets.stage1_pointodyssey import Stage1PointOdysseyDataset
from stwm.tracewm.datasets.stage1_kubric import Stage1KubricDataset
from stwm.tracewm.datasets.stage1_tapvid import Stage1TapVidDataset
from stwm.tracewm.datasets.stage1_tapvid3d import Stage1TapVid3DDataset
from stwm.tracewm.datasets.stage1_unified import load_stage1_minisplits

DATA_ROOT = Path('/home/chen034/workspace/data')
WORK_ROOT = Path('/home/chen034/workspace/stwm')
MINISPLIT_PATH = DATA_ROOT / '_manifests' / 'stage1_minisplits_20260408.json'
OUT = WORK_ROOT / 'reports' / 'stage1_loader_smoke_20260408.json'

SAMPLE_KEYS = [
    'dataset','split','clip_id','obs_frames','fut_frames','obs_valid','fut_valid',
    'obs_tracks_2d','fut_tracks_2d','obs_tracks_3d','fut_tracks_3d','visibility',
    'intrinsics','extrinsics','point_ids','meta'
]


def records_for(minisplits: dict[str, Any], dataset: str, split: str):
    return minisplits.get('datasets', {}).get(dataset, {}).get(split, [])


def describe(v: Any) -> dict[str, Any]:
    if isinstance(v, torch.Tensor):
        return {'type': 'Tensor', 'shape': list(v.shape), 'dtype': str(v.dtype)}
    if isinstance(v, list):
        return {'type': 'list', 'len': len(v), 'item_type': type(v[0]).__name__ if v else 'empty'}
    if isinstance(v, dict):
        return {'type': 'dict', 'keys': sorted(list(v.keys()))[:20]}
    if v is None:
        return {'type': 'None'}
    return {'type': type(v).__name__, 'repr': str(v)[:120]}


minisplits = load_stage1_minisplits(MINISPLIT_PATH)

point_ds = Stage1PointOdysseyDataset(
    data_root=DATA_ROOT,
    split='train_mini',
    minisplit_records=records_for(minisplits, 'pointodyssey', 'train_mini'),
)
kubric_ds = Stage1KubricDataset(
    data_root=DATA_ROOT,
    split='train_mini',
    minisplit_records=records_for(minisplits, 'kubric', 'train_mini'),
)
tapvid_ds = Stage1TapVidDataset(
    split='eval_mini',
    minisplit_records=records_for(minisplits, 'tapvid', 'eval_mini'),
)
tapvid3d_ds = Stage1TapVid3DDataset(
    data_root=DATA_ROOT,
    split='eval_mini',
    minisplit_records=records_for(minisplits, 'tapvid3d', 'eval_mini'),
)

dataset_map = {
    'pointodyssey': point_ds,
    'kubric': kubric_ds,
    'tapvid': tapvid_ds,
    'tapvid3d': tapvid3d_ds,
}

report = {
    'generated_at_utc': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
    'required_keys': SAMPLE_KEYS,
    'datasets': {},
    'all_passed': True,
}

for name, ds in dataset_map.items():
    n = min(3, len(ds))
    samples = []
    dataset_pass = True
    for i in range(n):
        sample = ds[i]
        missing = [k for k in SAMPLE_KEYS if k not in sample]
        if missing:
            dataset_pass = False
        desc = {k: describe(sample.get(k)) for k in SAMPLE_KEYS}
        samples.append({'index': i, 'missing_keys': missing, 'fields': desc})

    report['datasets'][name] = {
        'sampled': n,
        'dataset_len': len(ds),
        'passed': dataset_pass and n > 0,
        'samples': samples,
    }
    if not report['datasets'][name]['passed']:
        report['all_passed'] = False

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[loader-smoke] wrote: {OUT}")
print(f"[loader-smoke] all_passed={report['all_passed']}")
PY

echo "[stage1] step=visual_smoke"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/tools/visualize_stage1_sample.py"

echo "[stage1] step=batch_smoke"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from stwm.tracewm.datasets.stage1_unified import Stage1UnifiedDataset, stage1_collate_fn

DATA_ROOT = Path('/home/chen034/workspace/data')
WORK_ROOT = Path('/home/chen034/workspace/stwm')
MINISPLIT_PATH = DATA_ROOT / '_manifests' / 'stage1_minisplits_20260408.json'
OUT = WORK_ROOT / 'reports' / 'stage1_batch_smoke_20260408.json'


def field_desc(v: Any) -> dict[str, Any]:
    if isinstance(v, torch.Tensor):
        return {'type': 'Tensor', 'shape': list(v.shape), 'dtype': str(v.dtype)}
    if isinstance(v, list):
        return {'type': 'list', 'len': len(v), 'item_type': type(v[0]).__name__ if v else 'empty'}
    if isinstance(v, dict):
        return {'type': 'dict', 'keys': sorted(list(v.keys()))[:20]}
    if v is None:
        return {'type': 'None'}
    return {'type': type(v).__name__}


train_ds = Stage1UnifiedDataset(
    dataset_names=['pointodyssey', 'kubric'],
    split='train_mini',
    data_root=DATA_ROOT,
    minisplit_path=MINISPLIT_PATH,
    obs_len=8,
    fut_len=8,
)
loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0, collate_fn=stage1_collate_fn)

batches = []
for bi, batch in enumerate(loader):
    if bi >= 2:
        break
    snap = {k: field_desc(v) for k, v in batch.items()}
    batches.append({'batch_index': bi, 'fields': snap})

passed = len(batches) == 2
report = {
    'generated_at_utc': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
    'dataset_len': len(train_ds),
    'batches_collected': len(batches),
    'passed': passed,
    'batches': batches,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(f"[batch-smoke] wrote: {OUT}")
print(f"[batch-smoke] passed={passed}")
PY

echo "[stage1] step=smoke_gate_check"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import sys

WORK_ROOT = Path('/home/chen034/workspace/stwm')
contract = json.loads((WORK_ROOT / 'reports' / 'stage1_contract_smoke_20260408.json').read_text())
loader = json.loads((WORK_ROOT / 'reports' / 'stage1_loader_smoke_20260408.json').read_text())
batch = json.loads((WORK_ROOT / 'reports' / 'stage1_batch_smoke_20260408.json').read_text())

visual_doc = WORK_ROOT / 'docs' / 'STAGE1_VISUAL_SMOKE_20260408.md'
visual_dir = WORK_ROOT / 'outputs' / 'stage1_visual_checks'
expected_pngs = [
    visual_dir / 'pointodyssey_stage1_visual_smoke_20260408.png',
    visual_dir / 'kubric_stage1_visual_smoke_20260408.png',
    visual_dir / 'tapvid_stage1_visual_smoke_20260408.png',
    visual_dir / 'tapvid3d_stage1_visual_smoke_20260408.png',
]
visual_ok = visual_doc.exists() and all(p.exists() for p in expected_pngs)

all_pass = bool(contract.get('all_passed', False) and loader.get('all_passed', False) and batch.get('passed', False) and visual_ok)
print(f"[smoke-gate] contract={contract.get('all_passed')} loader={loader.get('all_passed')} batch={batch.get('passed')} visual={visual_ok}")
print(f"[smoke-gate] all_pass={all_pass}")
if not all_pass:
    sys.exit(23)
PY

echo "[stage1] step=tiny_train"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/trainers/train_tracewm_stage1_tiny.py" \
  --data-root "$DATA_ROOT" \
  --minisplit-path "$MINISPLIT_PATH" \
  --output-dir "$WORK_ROOT/outputs/training/tracewm_stage1_tiny_${DATE_TAG}" \
  --summary-json "$TINY_SUMMARY_REPORT" \
  --results-md "$WORK_ROOT/docs/TRACEWM_STAGE1_TINY_RESULTS_${DATE_TAG}.md" \
  --steps 40 \
  --batch-size 2

echo "[stage1] step=eval_smoke"
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tracewm.datasets.stage1_tapvid import Stage1TapVidDataset
from stwm.tracewm.datasets.stage1_tapvid3d import Stage1TapVid3DDataset
from stwm.tracewm.datasets.stage1_unified import load_stage1_minisplits

DATA_ROOT = Path('/home/chen034/workspace/data')
WORK_ROOT = Path('/home/chen034/workspace/stwm')
MINISPLIT_PATH = DATA_ROOT / '_manifests' / 'stage1_minisplits_20260408.json'
OUT_JSON = WORK_ROOT / 'reports' / 'tracewm_stage1_eval_smoke_20260408.json'
OUT_MD = WORK_ROOT / 'docs' / 'TRACEWM_STAGE1_EVAL_SMOKE_20260408.md'


def records_for(minisplits: dict[str, Any], dataset: str, split: str):
    return minisplits.get('datasets', {}).get(dataset, {}).get(split, [])


def endpoint_error_2d(sample: dict[str, Any]) -> float:
    obs = sample.get('obs_tracks_2d')
    fut = sample.get('fut_tracks_2d')
    if not isinstance(obs, torch.Tensor) or not isinstance(fut, torch.Tensor):
        return float('nan')

    obs = obs.to(torch.float32)
    fut = fut.to(torch.float32)
    if obs.shape[0] < 2:
        return float('nan')
    last = obs[-1]
    prev = obs[-2]
    delta = last - prev

    preds = []
    cur = last
    for _ in range(fut.shape[0]):
        cur = cur + delta
        preds.append(cur)
    pred = torch.stack(preds, dim=0)

    err = torch.linalg.norm(pred - fut, dim=-1).mean().item()
    return float(err)


def endpoint_error_3d(sample: dict[str, Any]) -> float:
    obs = sample.get('obs_tracks_3d')
    fut = sample.get('fut_tracks_3d')
    if not isinstance(obs, torch.Tensor) or not isinstance(fut, torch.Tensor):
        return float('nan')

    obs = obs.to(torch.float32)
    fut = fut.to(torch.float32)
    if obs.shape[0] < 2:
        return float('nan')

    last = obs[-1]
    prev = obs[-2]
    delta = last - prev

    preds = []
    cur = last
    for _ in range(fut.shape[0]):
        cur = cur + delta
        preds.append(cur)
    pred = torch.stack(preds, dim=0)

    err = torch.linalg.norm(pred - fut, dim=-1).mean().item()
    return float(err)


minisplits = load_stage1_minisplits(MINISPLIT_PATH)

tapvid_ds = Stage1TapVidDataset(
    split='eval_mini',
    minisplit_records=records_for(minisplits, 'tapvid', 'eval_mini'),
)
tapvid3d_ds = Stage1TapVid3DDataset(
    data_root=DATA_ROOT,
    split='eval_mini',
    minisplit_records=records_for(minisplits, 'tapvid3d', 'eval_mini'),
)

errs2d = []
for i in range(min(6, len(tapvid_ds))):
    errs2d.append(endpoint_error_2d(tapvid_ds[i]))
errs3d = []
for i in range(min(6, len(tapvid3d_ds))):
    errs3d.append(endpoint_error_3d(tapvid3d_ds[i]))


def safe_mean(xs):
    ys = [x for x in xs if np.isfinite(x)]
    return float(sum(ys) / max(len(ys), 1))


tapvid_ready = len(errs2d) > 0 and np.isfinite(safe_mean(errs2d))
tapvid3d_limited_ready = len(errs3d) > 0 and np.isfinite(safe_mean(errs3d))

report = {
    'generated_at_utc': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
    'tapvid': {
        'eval_mini_samples': len(tapvid_ds),
        'mean_endpoint_error_2d': safe_mean(errs2d),
        'main_eval_ready': bool(tapvid_ready),
    },
    'tapvid3d': {
        'eval_mini_samples': len(tapvid3d_ds),
        'mean_endpoint_error_3d_limited': safe_mean(errs3d),
        'limited_eval_ready': bool(tapvid3d_limited_ready),
        'full_eval_ready': False,
    },
    'notes': [
        'TapVid-3D evaluated under limited mini protocol only.',
        'Full benchmark evaluation is not part of this Stage 1 smoke lane.',
    ],
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
    '# TraceWM Stage 1 Eval Smoke (2026-04-08)',
    '',
    f"- generated_at_utc: {report['generated_at_utc']}",
    '',
    '## TAP-Vid (main eval)',
    '',
    f"- eval_mini_samples: {report['tapvid']['eval_mini_samples']}",
    f"- mean_endpoint_error_2d: {report['tapvid']['mean_endpoint_error_2d']:.6f}",
    f"- main_eval_ready: {report['tapvid']['main_eval_ready']}",
    '',
    '## TAPVid-3D (limited eval)',
    '',
    f"- eval_mini_samples: {report['tapvid3d']['eval_mini_samples']}",
    f"- mean_endpoint_error_3d_limited: {report['tapvid3d']['mean_endpoint_error_3d_limited']:.6f}",
    f"- limited_eval_ready: {report['tapvid3d']['limited_eval_ready']}",
    f"- full_eval_ready: {report['tapvid3d']['full_eval_ready']}",
    '',
    '## Notes',
    '',
]
for n in report['notes']:
    lines.append(f"- {n}")

OUT_MD.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

print(f"[eval-smoke] wrote: {OUT_JSON}")
print(f"[eval-smoke] wrote: {OUT_MD}")
print(f"[eval-smoke] tapvid_main_eval_ready={report['tapvid']['main_eval_ready']}")
print(f"[eval-smoke] tapvid3d_limited_eval_ready={report['tapvid3d']['limited_eval_ready']}")
PY

echo "[stage1] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[stage1] contract=$CONTRACT_PATH"
echo "[stage1] minisplit=$MINISPLIT_PATH"
echo "[stage1] contract_smoke=$CONTRACT_SMOKE_REPORT"
echo "[stage1] loader_smoke=$LOADER_SMOKE_REPORT"
echo "[stage1] batch_smoke=$BATCH_SMOKE_REPORT"
echo "[stage1] eval_smoke=$EVAL_SMOKE_REPORT"
echo "[stage1] tiny_summary=$TINY_SUMMARY_REPORT"
echo "[stage1] visual_doc=$VISUAL_DOC"
echo "[stage1] eval_doc=$EVAL_DOC"
echo "[stage1] log=$LOG_PATH"
