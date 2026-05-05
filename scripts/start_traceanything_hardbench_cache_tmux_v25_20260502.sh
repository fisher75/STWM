#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
PY=/home/chen034/miniconda3/envs/stwm/bin/python
MANIFEST=$ROOT/reports/stwm_traceanything_hardbench_launch_manifest_v25_20260502.json

python - <<'PY'
import json, subprocess, os
from pathlib import Path

ROOT = Path('/raid/chen034/workspace/stwm')
manifest = ROOT / 'reports/stwm_traceanything_hardbench_launch_manifest_v25_20260502.json'
log_dir = ROOT / 'logs/traceanything_hardbench_v24_20260502'
report_dir = ROOT / 'reports/stwm_traceanything_hardbench_v24_shards'
gpu_plan = [(32,0,1),(32,1,2),(32,2,3),(32,3,4),(64,0,5),(64,1,6),(64,2,7),(64,3,4)]
rows = []
for horizon, shard, gpu in gpu_plan:
    session = f'stwm_ta_v24_h{horizon}_s{shard}'
    log_path = log_dir / f'{session}.log'
    report_path = report_dir / f'{session}.json'
    cmd = (
        'cd /raid/chen034/workspace/stwm && '
        'export PYTHONPATH=/raid/chen034/workspace/stwm/code:/raid/chen034/workspace/stwm/third_party/TraceAnything && '
        f'export CUDA_VISIBLE_DEVICES={gpu} && '
        'export STWM_PROC_TITLE=python && export STWM_PROC_TITLE_MODE=generic && '
        '/home/chen034/miniconda3/envs/stwm/bin/python -u '
        'code/stwm/tools/run_traceanything_object_trajectory_teacher_v24_20260502.py '
        f'--horizon {horizon} --selection-horizon 64 --max-clips 150 --num-shards 4 --shard-index {shard} '
        f'--report-path {report_path} --doc-path /raid/chen034/workspace/stwm/docs/stwm_traceanything_hardbench_v24_shards/{session}.md '
        f'> {log_path} 2>&1'
    )
    rows.append(
        {
            'session_name': session,
            'horizon': horizon,
            'shard_index': shard,
            'gpu_id': gpu,
            'log_path': str(log_path.relative_to(ROOT)),
            'report_path': str(report_path.relative_to(ROOT)),
            'command': cmd,
            'min_free_mem_gb': 25,
            'preferred_free_mem_gb': 40,
            'launch_mode': 'retroactive_manifest_for_live_v25_execution',
        }
    )
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps({'launch_manifest': rows}, indent=2) + '\n')
print(manifest)
PY

echo "$MANIFEST"
