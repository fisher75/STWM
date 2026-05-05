#!/usr/bin/env bash
set -euo pipefail

ROOT=/raid/chen034/workspace/stwm
export PYTHONPATH="$ROOT/code:$ROOT/third_party/TraceAnything:${PYTHONPATH:-}"

/home/chen034/miniconda3/envs/stwm/bin/python - <<'PY'
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.run_traceanything_object_trajectory_teacher_v25_20260502 import (
    DEFAULT_BENCHMARK_PATH,
    _predecode_index,
    _select_candidates,
)

report_dir = ROOT / "reports/stwm_traceanything_hardbench_v25_shards"
manifest_path = ROOT / "reports/stwm_traceanything_hardbench_launch_manifest_v25_20260502.json"
watch_path = ROOT / "reports/stwm_traceanything_hardbench_watcher_v25_20260502.json"
final_report = ROOT / "reports/stwm_traceanything_hardbench_cache_v25_20260502.json"
final_doc = ROOT / "docs/STWM_TRACEANYTHING_HARDBENCH_CACHE_V25_20260502.md"
out_root = ROOT / "outputs/cache/stwm_traceanything_hardbench_v25"

manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"launch_manifest": []}
watch = json.loads(watch_path.read_text()) if watch_path.exists() else {"rows": []}
benchmark = json.loads(Path(DEFAULT_BENCHMARK_PATH).read_text())
predecode = _predecode_index()
expected_h32, stats_h32 = _select_candidates(benchmark, predecode, obs_len=8, selection_horizon=32, max_clips=300, allowed_item_keys=None)
expected_h64, stats_h64 = _select_candidates(benchmark, predecode, obs_len=8, selection_horizon=64, max_clips=300, allowed_item_keys=None)

shard_reports = sorted(report_dir.glob("stwm_ta_v25_h*_s*.json"))
combo_rows = defaultdict(list)
all_failures = []
all_skipped = []
for rp in shard_reports:
    data = json.loads(rp.read_text())
    combo_rows[f"H{int(data['horizon'])}"].extend(data.get("rows", []))
    all_failures.extend(data.get("failures", []))
    all_skipped.extend(data.get("skipped", []))

combo_cache_stats = {}
unique_by_combo = {}
all_unique_items = set()
point_count_total = 0
valid_point_values = []
subset_counter = Counter()
per_dataset = Counter()
for combo in ["M128_H32", "M512_H32", "M128_H64", "M512_H64"]:
    files = sorted((out_root / combo).glob("*/*.npz"))
    vis_cov = []
    valid = []
    same_frac = []
    traj_var = []
    items = set()
    for f in files:
        z = np.load(f, allow_pickle=True)
        item_key = str(np.asarray(z["item_key"]).item())
        items.add(item_key)
        all_unique_items.add(item_key)
        per_dataset[str(np.asarray(z["dataset"]).item())] += 1
        vis = np.asarray(z["visibility"]).astype(bool)
        tr = np.asarray(z["tracks_xy"], dtype=np.float32)
        valid_ratio = float(vis.mean()) if vis.size else 0.0
        valid.append(valid_ratio)
        valid_point_values.append(valid_ratio)
        vis_cov.append(valid_ratio)
        if "same_trajectory_fraction" in z.files:
            same_frac.append(float(np.asarray(z["same_trajectory_fraction"]).item()))
        traj_var.append(float(np.var(tr, axis=2).mean()))
        point_count_total += int(np.asarray(z["point_id"]).size)
    combo_cache_stats[combo] = {
        "cache_dir": str((out_root / combo).relative_to(ROOT)),
        "file_count": len(files),
        "point_count": int(point_count_total if False else sum(int(np.asarray(np.load(f, allow_pickle=True)["point_id"]).size) for f in files)),
        "valid_point_ratio": float(np.mean(valid)) if valid else 0.0,
        "same_trajectory_fraction": float(np.mean(same_frac)) if same_frac else None,
        "trajectory_variance": float(np.mean(traj_var)) if traj_var else None,
        "estimated_visibility_coverage": float(np.mean(vis_cov)) if vis_cov else 0.0,
        "total_size_bytes": int(sum(f.stat().st_size for f in files)),
        "sample_checksums": {str(f.relative_to(ROOT)): hashlib.md5(f.read_bytes()).hexdigest() for f in files[:10]},
    }
    unique_by_combo[combo] = items

row_lookup = {}
for rows in combo_rows.values():
    for row in rows:
        row_lookup[(row["item_key"], row["dataset"], row["split"])] = row
        for tag in row.get("reason_tags", []):
            subset_counter[tag] += 1

combo_summary = {}
for horizon_key, rows in combo_rows.items():
    out_rows = []
    for row in rows:
        cache_path = row.get("cache_paths_by_m", {}).get("M128") or next(iter(row.get("cache_paths_by_m", {}).values()), None)
        cot = None
        for comp in row.get("comparison_to_cotracker_by_m", {}).values():
            if isinstance(comp, dict) and comp.get("cotracker_cache_path"):
                cot = comp["cotracker_cache_path"]
                break
        out_rows.append(
            {
                "item_key": row["item_key"],
                "dataset": row["dataset"],
                "split": row["split"],
                "reason_tags": row.get("reason_tags", []),
                "cache_path": cache_path,
                "matching_cotracker_h16_path": cot,
            }
        )
    combo_summary[horizon_key] = {
        "processed_clip_count": len(rows),
        "rows": out_rows,
    }

expected_counts = {"H32": len(expected_h32), "H64": len(expected_h64)}
paired_h32 = len(unique_by_combo.get("M128_H32", set()) & unique_by_combo.get("M512_H32", set()))
paired_h64 = len(unique_by_combo.get("M128_H64", set()) & unique_by_combo.get("M512_H64", set()))
h32_ready = paired_h32 >= expected_counts["H32"] and expected_counts["H32"] > 0
h64_ready = paired_h64 >= expected_counts["H64"] and expected_counts["H64"] > 0
m128_ready = len(unique_by_combo.get("M128_H32", set())) >= expected_counts["H32"] and len(unique_by_combo.get("M128_H64", set())) >= expected_counts["H64"]
m512_ready = len(unique_by_combo.get("M512_H32", set())) >= expected_counts["H32"] and len(unique_by_combo.get("M512_H64", set())) >= expected_counts["H64"]
valid_point_ratio = float(np.mean(valid_point_values)) if valid_point_values else 0.0

vipseg_h64_available = int(stats_h64["per_dataset_candidate_counts"].get("VIPSEG", 0))
exact_blocker = None
if not h64_ready or not h32_ready or not m128_ready or not m512_ready:
    exact_blocker = "shard_completion_or_runtime_incomplete"
elif expected_counts["H64"] < 300:
    exact_blocker = f"H64 actual-frame-feasible hardbench clips are {expected_counts['H64']} total with VIPSEG={vipseg_h64_available}; benchmark-level H64 feasibility overestimates contiguous obs+future(72-frame) availability"

payload = {
    "audit_name": "stwm_traceanything_hardbench_cache_v25",
    "generated_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    "processed_clip_count": len(all_unique_items),
    "unique_item_count": len(all_unique_items),
    "failed_clip_count": len(all_failures),
    "skipped_clip_count": len(all_skipped),
    "point_count": int(point_count_total),
    "valid_point_ratio": valid_point_ratio,
    "H32_ready": h32_ready,
    "H64_ready": h64_ready,
    "M128_ready": m128_ready,
    "M512_ready": m512_ready,
    "expected_clip_count_by_horizon": expected_counts,
    "processed_clip_count_by_horizon": {"H32": paired_h32, "H64": paired_h64},
    "candidate_stats_by_horizon": {"H32": stats_h32, "H64": stats_h64},
    "hard_subset_counts": dict(subset_counter),
    "per_dataset_counts": dict(per_dataset),
    "combo_summary": combo_summary,
    "cache_paths_size_checksums": combo_cache_stats,
    "shard_log_paths": [row["log_path"] for row in manifest.get("launch_manifest", [])],
    "shard_report_paths": [str(p.relative_to(ROOT)) for p in shard_reports],
    "watcher_summary_path": str(watch_path.relative_to(ROOT)),
    "failed_clip_reasons": dict(Counter(f.get("reason", "unknown") for f in all_failures)),
    "traceanything_hardbench_cache_ready": bool(
        len(all_unique_items) >= 300
        and h32_ready
        and h64_ready
        and m128_ready
        and m512_ready
        and valid_point_ratio >= 0.4
    ),
    "exact_blocker": exact_blocker,
    "notes": [
        "teacher_target_uses_full_clip=true but model_input_observed_only=true for all saved cache items",
        "H64 feasibility is bounded by actual contiguous 72-frame availability, not benchmark tag alone",
    ],
}
final_report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
lines = [
    "# STWM TraceAnything Hardbench Cache V25",
    "",
]
for key in [
    "processed_clip_count",
    "failed_clip_count",
    "point_count",
    "valid_point_ratio",
    "H32_ready",
    "H64_ready",
    "M128_ready",
    "M512_ready",
    "traceanything_hardbench_cache_ready",
    "exact_blocker",
]:
    lines.append(f"- {key}: `{payload.get(key)}`")
lines.append(f"- expected_H32: `{expected_counts['H32']}`")
lines.append(f"- expected_H64: `{expected_counts['H64']}`")
final_doc.write_text("\n".join(lines).rstrip() + "\n")
print(final_report.relative_to(ROOT))
PY
