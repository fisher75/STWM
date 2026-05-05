#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
python - <<'PY'
import json
from pathlib import Path
ROOT = Path('/raid/chen034/workspace/stwm')
src = ROOT / 'reports/stwm_traceanything_hardbench_cache_v24_20260502.json'
dst = ROOT / 'reports/stwm_traceanything_hardbench_cache_v25_20260502.json'
doc = ROOT / 'docs/STWM_TRACEANYTHING_HARDBENCH_CACHE_V25_20260502.md'
data = json.loads(src.read_text())
data['audit_name'] = 'stwm_traceanything_hardbench_cache_v25'
data['source_report'] = str(src.relative_to(ROOT))
data['traceanything_hardbench_cache_ready'] = bool(
    data.get('processed_clip_count', 0) >= 300
    and data.get('H32_ready')
    and data.get('H64_ready')
    and data.get('M128_ready')
    and data.get('M512_ready')
    and data.get('valid_point_ratio', 0.0) >= 0.4
)
dst.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
lines=['# STWM TraceAnything Hardbench Cache V25','']
for k in ['processed_clip_count','complete_m128_m512_pair_count','file_count_total','point_count','valid_point_ratio','H32_ready','H64_ready','M128_ready','M512_ready','traceanything_hardbench_cache_ready','exact_blocker']:
    lines.append(f'- {k}: `{data.get(k)}`')
doc.write_text('\n'.join(lines) + '\n')
print(dst)
PY
