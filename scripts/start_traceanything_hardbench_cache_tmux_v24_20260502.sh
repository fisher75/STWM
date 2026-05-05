#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
PY=/home/chen034/miniconda3/envs/stwm/bin/python
CODE_PYTHONPATH="$ROOT/code:$ROOT/third_party/TraceAnything"
LOG_DIR="$ROOT/logs/traceanything_hardbench_v24_20260502"
REPORT_DIR="$ROOT/reports/stwm_traceanything_hardbench_v24_shards"
DOC_DIR="$ROOT/docs/stwm_traceanything_hardbench_v24_shards"
SELECT_FILE="$ROOT/reports/stwm_traceanything_hardbench_v24_selected_item_keys_20260502.json"
mkdir -p "$LOG_DIR" "$REPORT_DIR" "$DOC_DIR"

build_selection() {
  "$PY" - <<'PY'
import json
from pathlib import Path
ROOT = Path('/raid/chen034/workspace/stwm')
bench = json.loads((ROOT/'reports/stwm_ostf_hard_benchmark_v2_20260502.json').read_text())
per_item = bench['per_item']
rows = []
for item_key, row in per_item.items():
    st = row.get('horizon_status', {}).get('H64', {})
    if not st.get('feasible', False):
        continue
    if not row.get('raw_frame_available', False):
        continue
    if not row.get('semantic_instance_available', False):
        continue
    tags = set(row.get('reason_tags', []))
    if not tags:
        continue
    score = 0
    if 'top20_cv_hard' in tags: score += 100
    if 'top30_cv_hard' in tags: score += 50
    if 'occlusion_hard' in tags: score += 30
    if 'nonlinear_hard' in tags: score += 20
    if 'interaction_hard' in tags: score += 10
    rows.append({'item_key': item_key, 'dataset': row['dataset'], 'score': score})
rows.sort(key=lambda r: (-r['score'], r['item_key']))
buckets = {'VIPSEG': [r for r in rows if r['dataset']=='VIPSEG'], 'VSPW': [r for r in rows if r['dataset']=='VSPW']}
selected = []
while len(selected) < 300 and (buckets['VIPSEG'] or buckets['VSPW']):
    for ds in ['VIPSEG', 'VSPW']:
        if buckets[ds] and len(selected) < 300:
            selected.append(buckets[ds].pop(0)['item_key'])
(ROOT/'reports/stwm_traceanything_hardbench_v24_selected_item_keys_20260502.json').write_text(json.dumps({'item_keys': selected}, indent=2))
print(len(selected))
PY
}

build_selection

GPUS=(1 2 3 4 5 6 7 4)
job_idx=0
for horizon in 32 64; do
  for shard in 0 1 2 3; do
    gpu=${GPUS[$job_idx]}
    job_idx=$((job_idx + 1))
    session="stwm_ta_v24_h${horizon}_s${shard}"
    log="$LOG_DIR/${session}.log"
    report="$REPORT_DIR/${session}.json"
    doc="$DOC_DIR/${session}.md"
    tmux kill-session -t "$session" 2>/dev/null || true
    tmux new-session -d -s "$session" \
      "cd $ROOT && export PYTHONPATH=$CODE_PYTHONPATH && export CUDA_VISIBLE_DEVICES=$gpu && export STWM_PROC_TITLE=python && export STWM_PROC_TITLE_MODE=generic && $PY -u code/stwm/tools/run_traceanything_object_trajectory_teacher_v24_20260502.py --horizon $horizon --selection-horizon 64 --max-clips 150 --num-shards 4 --shard-index $shard --report-path $report --doc-path $doc > $log 2>&1"
    echo "$session gpu=$gpu log=$log report=$report"
  done
done
