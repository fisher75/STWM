#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/home/chen034/workspace/stwm"
DATA_ROOT="/home/chen034/workspace/data"
DATE_TAG="20260408"
SEED="20260408"

MINISPLIT_PATH="$DATA_ROOT/_manifests/stage1_minisplits_${DATE_TAG}.json"
ITER1_SPLITS_PATH="$DATA_ROOT/_manifests/stage1_iter1_splits_${DATE_TAG}.json"
ITER1_SPLITS_DOC="$WORK_ROOT/docs/STAGE1_ITER1_SPLITS_${DATE_TAG}.md"

POINT_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_pointodyssey_only_summary.json"
KUBRIC_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_kubric_only_summary.json"
JOINT_SUMMARY="$WORK_ROOT/reports/tracewm_stage1_iter1_joint_po_kubric_summary.json"
COMPARISON_JSON="$WORK_ROOT/reports/tracewm_stage1_iter1_comparison_${DATE_TAG}.json"
RESULTS_DOC="$WORK_ROOT/docs/TRACEWM_STAGE1_ITER1_RESULTS_${DATE_TAG}.md"

if [[ -x "/home/chen034/miniconda3/envs/stwm/bin/python" ]]; then
  PYTHON_BIN="/home/chen034/miniconda3/envs/stwm/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$WORK_ROOT/reports" "$WORK_ROOT/docs" "$WORK_ROOT/outputs/training" "$DATA_ROOT/_manifests"

export PYTHONPATH="$WORK_ROOT/code:${PYTHONPATH:-}"


echo "[iter1] start: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[iter1] work_root=$WORK_ROOT"
echo "[iter1] data_root=$DATA_ROOT"
echo "[iter1] python=$PYTHON_BIN"

echo "[iter1] step=build_iter1_splits"
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List
import json

DATA_ROOT = Path('/home/chen034/workspace/data')
WORK_ROOT = Path('/home/chen034/workspace/stwm')
DATE_TAG = '20260408'

MINISPLIT_PATH = DATA_ROOT / '_manifests' / f'stage1_minisplits_{DATE_TAG}.json'
OUT_SPLITS = DATA_ROOT / '_manifests' / f'stage1_iter1_splits_{DATE_TAG}.json'
OUT_DOC = WORK_ROOT / 'docs' / f'STAGE1_ITER1_SPLITS_{DATE_TAG}.md'


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_bucket(text: str, base: int = 10) -> int:
    digest = sha1(text.encode('utf-8')).hexdigest()
    return int(digest, 16) % base


def expand_point(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        seq = str(item.get('sequence_name', '')).strip()
        if not seq:
            continue

        frame_count = int(item.get('frame_count', 0) or 0)
        start0 = int(item.get('start_index', 0) or 0)

        base = dict(item)
        base['clip_id'] = str(item.get('clip_id', f'point_{seq}_00000')) + '__iter1_s0'
        base['selection_reason'] = str(item.get('selection_reason', '')) + ';iter1_expand_s0'
        out.append(base)

        start1 = start0
        if frame_count > 32:
            start1 = min(max(frame_count // 2, 0), max(frame_count - 16, 0))
        elif frame_count > 16:
            start1 = max(frame_count - 16, 0)

        if start1 != start0:
            alt = dict(item)
            alt['start_index'] = int(start1)
            alt['clip_id'] = str(item.get('clip_id', f'point_{seq}_00000')) + '__iter1_s1'
            alt['selection_reason'] = str(item.get('selection_reason', '')) + ';iter1_expand_s1'
            out.append(alt)

    return out


def build_kubric_iter1_train(base_train: List[Dict[str, Any]], target_extra: int = 48) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = [dict(x) for x in base_train if isinstance(x, dict)]

    seen = set()
    for item in out:
        seen.add(str(item.get('tfrecord_path', '')).strip())

    movi_root = DATA_ROOT / 'kubric' / 'tfds' / 'movi_e'
    tfrecords = sorted([
        *movi_root.rglob('*.tfrecord'),
        *movi_root.rglob('*.tfrecord-*'),
        *movi_root.rglob('*.tfrecord.gz'),
    ])

    extras: List[Path] = []
    for p in tfrecords:
        if stable_bucket(str(p), base=10) >= 8:
            continue
        if str(p) in seen:
            continue
        extras.append(p)
        if len(extras) >= target_extra:
            break

    for i, p in enumerate(extras):
        out.append(
            {
                'clip_id': f'kubric_iter1_extra_{i:04d}_{p.stem}',
                'tfrecord_path': str(p),
                'selection_reason': 'iter1_extra_stable_hash_train_pool',
            }
        )

    return out


payload = json.loads(MINISPLIT_PATH.read_text(encoding='utf-8'))
datasets = payload.get('datasets', {})

point_train = list(datasets.get('pointodyssey', {}).get('train_mini', []))
point_val = list(datasets.get('pointodyssey', {}).get('val_mini', []))
kubric_train = list(datasets.get('kubric', {}).get('train_mini', []))
kubric_val = list(datasets.get('kubric', {}).get('val_mini', []))
tapvid_eval = list(datasets.get('tapvid', {}).get('eval_mini', []))
tapvid3d_eval = list(datasets.get('tapvid3d', {}).get('eval_mini', []))

point_train_iter1 = expand_point(point_train)
point_val_iter1 = list(point_val)
kubric_train_iter1 = build_kubric_iter1_train(kubric_train, target_extra=48)
kubric_val_iter1 = list(kubric_val)

out_payload = {
    'generated_at_utc': now_iso(),
    'seed': 20260408,
    'source_minisplit_path': str(MINISPLIT_PATH),
    'task': 'trace_only_future_trace_state_generation',
    'experiment_matrix': ['pointodyssey_only', 'kubric_only', 'joint_po_kubric'],
    'datasets': {
        'pointodyssey': {
            'pointodyssey_iter1_train': point_train_iter1,
            'pointodyssey_iter1_val': point_val_iter1,
            'train_iter1_pointodyssey': point_train_iter1,
            'val_iter1_pointodyssey': point_val_iter1,
            'joint_iter1_train': point_train_iter1,
            'joint_iter1_val': point_val_iter1,
            'train_iter1_joint': point_train_iter1,
            'val_iter1_joint': point_val_iter1,
        },
        'kubric': {
            'kubric_iter1_train': kubric_train_iter1,
            'kubric_iter1_val': kubric_val_iter1,
            'train_iter1_kubric': kubric_train_iter1,
            'val_iter1_kubric': kubric_val_iter1,
            'joint_iter1_train': kubric_train_iter1,
            'joint_iter1_val': kubric_val_iter1,
            'train_iter1_joint': kubric_train_iter1,
            'val_iter1_joint': kubric_val_iter1,
        },
        'tapvid': {
            'eval_mini': tapvid_eval,
        },
        'tapvid3d': {
            'eval_mini': tapvid3d_eval,
        },
    },
    'summary': {
        'pointodyssey_iter1_train_count': len(point_train_iter1),
        'pointodyssey_iter1_val_count': len(point_val_iter1),
        'kubric_iter1_train_count': len(kubric_train_iter1),
        'kubric_iter1_val_count': len(kubric_val_iter1),
        'joint_iter1_pointodyssey_count': len(point_train_iter1),
        'joint_iter1_kubric_count': len(kubric_train_iter1),
        'tapvid_eval_mini_count': len(tapvid_eval),
        'tapvid3d_eval_mini_count': len(tapvid3d_eval),
    },
}

OUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
OUT_SPLITS.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding='utf-8')

lines = [
    '# Stage 1 Iteration-1 Splits (2026-04-08)',
    '',
    f"- generated_at_utc: {out_payload['generated_at_utc']}",
    f"- source_minisplit_path: {MINISPLIT_PATH}",
    f"- out_splits: {OUT_SPLITS}",
    f"- seed: {out_payload['seed']}",
    '',
    '## Policy',
    '',
    '- Contract and base minisplit are read-only references.',
    '- Iter1 split is an extension for Stage 1 trace-only iteration.',
    '- No Stage 2 semantics, no video reconstruction, no WAN, no MotionCrafter VAE.',
    '',
    '## Counts',
    '',
    f"- pointodyssey_iter1_train: {len(point_train_iter1)}",
    f"- pointodyssey_iter1_val: {len(point_val_iter1)}",
    f"- kubric_iter1_train: {len(kubric_train_iter1)}",
    f"- kubric_iter1_val: {len(kubric_val_iter1)}",
    f"- joint_iter1_train(pointodyssey branch): {len(point_train_iter1)}",
    f"- joint_iter1_train(kubric branch): {len(kubric_train_iter1)}",
    f"- tapvid eval_mini: {len(tapvid_eval)}",
    f"- tapvid3d eval_mini: {len(tapvid3d_eval)}",
]

OUT_DOC.parent.mkdir(parents=True, exist_ok=True)
OUT_DOC.write_text('\n'.join(lines) + '\n', encoding='utf-8')

print(f'[iter1-splits] wrote: {OUT_SPLITS}')
print(f'[iter1-splits] wrote: {OUT_DOC}')
print(f"[iter1-splits] point_train={len(point_train_iter1)} point_val={len(point_val_iter1)} kubric_train={len(kubric_train_iter1)} kubric_val={len(kubric_val_iter1)}")
print(f"[iter1-splits] tapvid_eval={len(tapvid_eval)} tapvid3d_eval={len(tapvid3d_eval)}")
PY

run_experiment() {
  local run_name="$1"
  local dataset_choice="$2"

  local output_dir="$WORK_ROOT/outputs/training/${run_name}"
  local summary_json="$WORK_ROOT/reports/${run_name}_summary.json"

  echo "[iter1] run=${run_name} dataset_choice=${dataset_choice}"
  "$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/trainers/train_tracewm_stage1_iter1.py" \
    --run-name "$run_name" \
    --dataset-choice "$dataset_choice" \
    --data-root "$DATA_ROOT" \
    --splits-path "$ITER1_SPLITS_PATH" \
    --output-dir "$output_dir" \
    --summary-json "$summary_json" \
    --seed "$SEED" \
    --epochs 6 \
    --steps-per-epoch 80 \
    --batch-size 4 \
    --hidden-dim 128 \
    --lr 1e-3 \
    --weight-decay 0.0 \
    --free-loss-weight 0.5 \
    --eval-max-tapvid-samples 6 \
    --eval-max-tapvid3d-samples 12 \
    --no-auto-resume
}

run_experiment "tracewm_stage1_iter1_pointodyssey_only" "pointodyssey_only"
run_experiment "tracewm_stage1_iter1_kubric_only" "kubric_only"
run_experiment "tracewm_stage1_iter1_joint_po_kubric" "joint_po_kubric"

echo "[iter1] step=summarize_comparison"
"$PYTHON_BIN" "$WORK_ROOT/code/stwm/tracewm/tools/summarize_stage1_iter1.py" \
  --point-summary "$POINT_SUMMARY" \
  --kubric-summary "$KUBRIC_SUMMARY" \
  --joint-summary "$JOINT_SUMMARY" \
  --comparison-json "$COMPARISON_JSON" \
  --results-md "$RESULTS_DOC" \
  --protocol-doc "$WORK_ROOT/docs/TRACEWM_STAGE1_ITERATION1_PROTOCOL_20260408.md" \
  --splits-doc "$ITER1_SPLITS_DOC"

echo "[iter1] done: $(date '+%Y-%m-%d %H:%M:%S %z')"
echo "[iter1] iter1_splits=$ITER1_SPLITS_PATH"
echo "[iter1] point_summary=$POINT_SUMMARY"
echo "[iter1] kubric_summary=$KUBRIC_SUMMARY"
echo "[iter1] joint_summary=$JOINT_SUMMARY"
echo "[iter1] comparison_json=$COMPARISON_JSON"
echo "[iter1] results_doc=$RESULTS_DOC"
