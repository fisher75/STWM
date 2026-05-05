#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc


REPORT_PATH = ROOT / "reports/stwm_external_point_benchmark_audit_v2_20260502.json"
DOC_PATH = ROOT / "docs/STWM_EXTERNAL_POINT_BENCHMARK_AUDIT_V2_20260502.md"


def _list_paths(root: Path, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(str(p.relative_to(ROOT)) for p in root.glob(pattern))
    return sorted(set(found))


def _npz_summary(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    out = {"path": str(path.relative_to(ROOT)), "keys": sorted(z.files)}
    for key in z.files:
        try:
            out[f"{key}_shape"] = list(np.asarray(z[key]).shape)
        except Exception:
            pass
    return out


def main() -> int:
    data_root = ROOT / "data"
    code_root = ROOT / "code/stwm/tracewm/datasets"
    stage1_cache = ROOT / "outputs/stage1_minisplit_cache/tapvid_20260408"
    pointodyssey_data = _list_paths(data_root, ["**/*PointOdyssey*", "**/*pointodyssey*"])
    tapvid_data = _list_paths(data_root, ["**/*tapvid*", "**/*TAP-Vid*"])
    tapir_data = _list_paths(data_root, ["**/*tapir*", "**/*TAPIR*"])
    wrappers = {
        "pointodyssey_wrapper": str((code_root / "stage1_pointodyssey.py").relative_to(ROOT)) if (code_root / "stage1_pointodyssey.py").exists() else None,
        "tapvid_wrapper": str((code_root / "stage1_tapvid.py").relative_to(ROOT)) if (code_root / "stage1_tapvid.py").exists() else None,
        "tapvid3d_wrapper": str((code_root / "stage1_tapvid3d.py").relative_to(ROOT)) if (code_root / "stage1_tapvid3d.py").exists() else None,
    }
    tapvid_cache_paths = sorted(stage1_cache.glob("*.npz"))
    minimal_samples = [_npz_summary(p) for p in tapvid_cache_paths[:20]]
    payload = {
        "audit_name": "stwm_external_point_benchmark_audit_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pointodyssey_dataset_available_under_data_root": bool(pointodyssey_data),
        "pointodyssey_dataset_paths": pointodyssey_data[:50],
        "tapvid_dataset_available_under_data_root": bool(tapvid_data),
        "tapvid_dataset_paths": tapvid_data[:50],
        "tapir_cache_or_repo_available_under_data_root": bool(tapir_data),
        "tapir_paths": tapir_data[:50],
        "dataset_wrappers": wrappers,
        "tapvid_minicache_available": bool(tapvid_cache_paths),
        "tapvid_minicache_clip_count": len(tapvid_cache_paths),
        "tapvid_minicache_examples": minimal_samples,
        "external_point_benchmark_ready": bool(pointodyssey_data or len(tapvid_cache_paths) >= 20),
        "exact_blockers": [
            *([] if pointodyssey_data else ["PointOdyssey GT dataset not materialized under /raid/chen034/workspace/stwm/data"]),
            *([] if tapvid_data else ["No full TAP-Vid dataset payload found under /raid/chen034/workspace/stwm/data"]),
            *([] if tapir_data else ["No TAPIR official repo/cache/weights found under /raid/chen034/workspace/stwm/data"]),
            *([] if len(tapvid_cache_paths) >= 20 else [f"Only {len(tapvid_cache_paths)} local TAP-Vid mini-cache clips are available; cannot build the requested 20-clip OSTF-compatible GT sample from current live files"]),
        ],
        "recommended_next_step": "download_pointodyssey_tapvid" if not (pointodyssey_data or len(tapvid_cache_paths) >= 20) else "build_20clip_gt_point_sample",
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM External Point Benchmark Audit V2",
        payload,
        [
            "pointodyssey_dataset_available_under_data_root",
            "tapvid_dataset_available_under_data_root",
            "tapir_cache_or_repo_available_under_data_root",
            "tapvid_minicache_clip_count",
            "external_point_benchmark_ready",
            "exact_blockers",
            "recommended_next_step",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
