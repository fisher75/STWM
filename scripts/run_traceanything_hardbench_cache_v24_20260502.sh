#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
REPORT_DIR="$ROOT/reports/stwm_traceanything_hardbench_v24_shards"
FINAL_REPORT="$ROOT/reports/stwm_traceanything_hardbench_cache_v24_20260502.json"
FINAL_DOC="$ROOT/docs/STWM_TRACEANYTHING_HARDBENCH_CACHE_V24_20260502.md"

python - <<'PY'
import json, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

ROOT = Path('/raid/chen034/workspace/stwm')
report_dir = ROOT / 'reports/stwm_traceanything_hardbench_v24_shards'
final_report = ROOT / 'reports/stwm_traceanything_hardbench_cache_v24_20260502.json'
final_doc = ROOT / 'docs/STWM_TRACEANYTHING_HARDBENCH_CACHE_V24_20260502.md'
reports = sorted(report_dir.glob('stwm_ta_v24_h*_s*.json'))
if not reports:
    raise SystemExit('no_shard_reports_found')
combo_summary = {}
all_rows = []
all_failures = []
processed_item_keys = set()
total_point_count = 0
for rp in reports:
    data = json.loads(rp.read_text())
    horizon = int(data['horizon'])
    combo_key = f"H{horizon}"
    combo = combo_summary.setdefault(combo_key, {'processed_clip_count': 0, 'failed_clip_count': 0, 'rows': [], 'failures': [], 'point_count': 0, 'per_subset_counts': Counter(), 'valid_point_ratios': [], 'same_fracs': [], 'vis_cov': [], 'dataset_counts': Counter()})
    combo['processed_clip_count'] += int(data.get('processed_clip_count', 0))
    combo['failed_clip_count'] += int(data.get('failed_clip_count', 0))
    combo['rows'].extend(data.get('rows', []))
    combo['failures'].extend(data.get('failures', []))
    combo['point_count'] += int(data.get('point_count', 0))
    combo['per_subset_counts'].update(data.get('per_subset_counts', {}))
    combo['valid_point_ratios'].append(float(data.get('mean_valid_point_ratio', 0.0)))
    combo['same_fracs'].append(float(data.get('mean_same_trajectory_fraction', 1.0)))
    combo['vis_cov'].append(float(data.get('mean_visibility_coverage', 0.0)))
    combo['dataset_counts'].update(data.get('processed_dataset_counts', {}))
    for row in data.get('rows', []):
        processed_item_keys.add(row['item_key'])
        all_rows.append(row)
    all_failures.extend(data.get('failures', []))
    total_point_count += int(data.get('point_count', 0))

for horizon_key, combo in combo_summary.items():
    combo['per_subset_counts'] = dict(combo['per_subset_counts'])
    combo['processed_dataset_counts'] = dict(combo['dataset_counts'])
    combo['mean_valid_point_ratio'] = float(np.mean(combo['valid_point_ratios'])) if combo['valid_point_ratios'] else 0.0
    combo['mean_same_trajectory_fraction'] = float(np.mean(combo['same_fracs'])) if combo['same_fracs'] else 1.0
    combo['mean_visibility_coverage'] = float(np.mean(combo['vis_cov'])) if combo['vis_cov'] else 0.0
    for row in combo['rows']:
        row['cache_path'] = next(iter(row.get('cache_paths_by_m', {}).values()), None)
        row['matching_cotracker_h16_path'] = None
        for mkey, comp in row.get('comparison_to_cotracker_by_m', {}).items():
            if isinstance(comp, dict) and comp.get('cotracker_cache_path'):
                row['matching_cotracker_h16_path'] = comp['cotracker_cache_path']
                break
    del combo['dataset_counts']; del combo['valid_point_ratios']; del combo['same_fracs']; del combo['vis_cov']

cache_dirs = {
    'M128_H32': ROOT / 'outputs/cache/stwm_traceanything_hardbench_v24/M128_H32',
    'M512_H32': ROOT / 'outputs/cache/stwm_traceanything_hardbench_v24/M512_H32',
    'M128_H64': ROOT / 'outputs/cache/stwm_traceanything_hardbench_v24/M128_H64',
    'M512_H64': ROOT / 'outputs/cache/stwm_traceanything_hardbench_v24/M512_H64',
}
combo_file_stats = {}
for name, path in cache_dirs.items():
    files = sorted(path.glob('*/*.npz'))
    combo_file_stats[name] = {
        'cache_dir': str(path.relative_to(ROOT)) if path.exists() else str(path),
        'file_count': len(files),
        'total_size_bytes': int(sum(f.stat().st_size for f in files)) if files else 0,
        'sample_checksums': {str(f.relative_to(ROOT)): hashlib.md5(f.read_bytes()).hexdigest() for f in files[:10]},
        'ready': bool(files),
    }

payload = {
    'audit_name': 'stwm_traceanything_hardbench_cache_v24',
    'processed_clip_count': int(sum(combo['processed_clip_count'] for combo in combo_summary.values())),
    'unique_item_count': len(processed_item_keys),
    'failed_clip_count': len(all_failures),
    'point_count': int(total_point_count),
    'valid_point_ratio': float(np.mean([combo_summary[k]['mean_valid_point_ratio'] for k in combo_summary])) if combo_summary else 0.0,
    'trajectory_variance_proxy': {k: float(np.mean([row.get('comparison_to_cotracker_by_m',{}).get('M128',{}).get('traceanything_trajectory_variance', 0.0) for row in v['rows'] if row.get('comparison_to_cotracker_by_m',{}).get('M128',{}).get('traceanything_trajectory_variance') is not None])) if v['rows'] else None for k,v in combo_summary.items()},
    'same_trajectory_fraction': {k: v['mean_same_trajectory_fraction'] for k,v in combo_summary.items()},
    'estimated_visibility_coverage': {k: v['mean_visibility_coverage'] for k,v in combo_summary.items()},
    'H32_ready': bool(combo_file_stats['M128_H32']['ready'] and combo_file_stats['M512_H32']['ready']),
    'H64_ready': bool(combo_file_stats['M128_H64']['ready'] and combo_file_stats['M512_H64']['ready']),
    'M128_ready': bool(combo_file_stats['M128_H32']['ready'] and combo_file_stats['M128_H64']['ready']),
    'M512_ready': bool(combo_file_stats['M512_H32']['ready'] and combo_file_stats['M512_H64']['ready']),
    'combo_summary': combo_summary,
    'cache_paths_size_checksums': combo_file_stats,
    'failed_clip_reasons': dict(Counter(f['reason'] for f in all_failures)),
    'shard_report_paths': [str(p.relative_to(ROOT)) for p in reports],
}
final_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
lines = ['# STWM TraceAnything Hardbench Cache V24', '']
for key in ['processed_clip_count','failed_clip_count','point_count','valid_point_ratio','H32_ready','H64_ready','M128_ready','M512_ready']:
    lines.append(f'- {key}: `{payload.get(key)}`')
lines.append(f"- shard_report_count: `{len(reports)}`")
final_doc.write_text('\n'.join(lines).rstrip() + '\n')
print(final_report.relative_to(ROOT))
PY
