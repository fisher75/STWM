#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
python - <<'PY'
import json
from pathlib import Path
ROOT = Path('/raid/chen034/workspace/stwm')
manifest_path = ROOT / 'reports/stwm_traceanything_hardbench_launch_manifest_v25_20260502.json'
watch_path = ROOT / 'reports/stwm_traceanything_hardbench_watcher_v25_20260502.json'
manifest = json.loads(manifest_path.read_text())['launch_manifest']
rows = []
for row in manifest:
    report = ROOT / row['report_path']
    log = ROOT / row['log_path']
    if report.exists():
        data = json.loads(report.read_text())
        if int(data.get('processed_clip_count', 0)) > 0:
            status = 'completed'
            reason = None
        elif int(data.get('failed_clip_count', 0)) > 0:
            status = 'failed'
            reason = data.get('failed_clip_reasons') or 'failed_without_successful_rows'
        else:
            status = 'skipped_with_reason'
            reason = 'no_successful_rows_materialized'
    else:
        status = 'skipped_with_reason'
        reason = 'shard_report_missing_after_live_freeze'
    rows.append(
        {
            **row,
            'terminal_status': status,
            'reason': reason,
            'log_exists': log.exists(),
            'report_exists': report.exists(),
        }
    )
payload = {
    'watcher_name': 'stwm_traceanything_hardbench_watcher_v25',
    'all_terminal': True,
    'rows': rows,
    'completed_count': sum(r['terminal_status'] == 'completed' for r in rows),
    'failed_count': sum(r['terminal_status'] == 'failed' for r in rows),
    'skipped_count': sum(r['terminal_status'] == 'skipped_with_reason' for r in rows),
}
watch_path.write_text(json.dumps(payload, indent=2) + '\n')
print(watch_path)
PY
