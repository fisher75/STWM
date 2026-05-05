#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc


REPORT_PATH = ROOT / "reports/stwm_traceanything_visibility_extraction_audit_v25_20260502.json"
DOC_PATH = ROOT / "docs/STWM_TRACEANYTHING_VISIBILITY_EXTRACTION_AUDIT_V25_20260502.md"
V25_CACHE_REPORT = ROOT / "reports/stwm_traceanything_hardbench_cache_v25_20260502.json"


def _scalar(x: Any) -> Any:
    a = np.asarray(x)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def main() -> int:
    cache_report = json.loads(V25_CACHE_REPORT.read_text(encoding="utf-8"))
    cache_root = ROOT / "outputs/cache/stwm_traceanything_hardbench_v24"
    combo_stats: dict[str, dict[str, Any]] = {}
    target_side_flags = []
    native_visibility_flags = []
    query_frame_only_eval_count = 0
    for combo in ["M128_H32", "M512_H32", "M128_H64", "M512_H64"]:
        files = sorted((cache_root / combo).glob("*/*.npz"))
        if not files:
            combo_stats[combo] = {
                "file_count": 0,
                "estimated_visibility_coverage": None,
                "valid_point_ratio": None,
                "same_trajectory_fraction": None,
                "trajectory_variance": None,
                "target_side_box_search_used_ratio": None,
                "native_visibility_available": None,
            }
            continue
        vis_cov = []
        valid_ratios = []
        same_fracs = []
        variances = []
        for f in files:
            z = np.load(f, allow_pickle=True)
            vis = np.asarray(z["visibility"]).astype(bool)
            tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
            vis_cov.append(float(vis.mean()) if vis.size else 0.0)
            valid_ratios.append(float(vis.mean()) if vis.size else 0.0)
            if "same_trajectory_fraction" in z.files:
                same_fracs.append(float(_scalar(z["same_trajectory_fraction"])))
            variances.append(float(np.var(tracks, axis=2).mean()))
            if "target_side_object_box_search_used" in z.files:
                target_side_flags.append(bool(_scalar(z["target_side_object_box_search_used"])))
            if "native_visibility_available" in z.files:
                native_visibility_flags.append(bool(_scalar(z["native_visibility_available"])))
        combo_stats[combo] = {
            "file_count": len(files),
            "estimated_visibility_coverage": float(np.mean(vis_cov)),
            "valid_point_ratio": float(np.mean(valid_ratios)),
            "same_trajectory_fraction": float(np.mean(same_fracs)) if same_fracs else None,
            "trajectory_variance": float(np.mean(variances)),
            "target_side_box_search_used_ratio": float(np.mean(target_side_flags[-len(files) :])) if target_side_flags else None,
            "native_visibility_available": bool(np.mean(native_visibility_flags[-len(files) :]) > 0.5) if native_visibility_flags else False,
        }

    overall_valid = [
        stats["valid_point_ratio"]
        for stats in combo_stats.values()
        if stats["valid_point_ratio"] is not None
    ]
    overall_vis = [
        stats["estimated_visibility_coverage"]
        for stats in combo_stats.values()
        if stats["estimated_visibility_coverage"] is not None
    ]
    overall_var = [
        stats["trajectory_variance"]
        for stats in combo_stats.values()
        if stats["trajectory_variance"] is not None
    ]
    payload = {
        "audit_name": "stwm_traceanything_visibility_extraction_audit_v25",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "native_visibility_available": False,
        "estimated_visibility_method": "traceanything_confidence_plus_trajectory_consistency_teacher_only",
        "estimated_visibility_coverage_by_combo": {
            combo: stats["estimated_visibility_coverage"] for combo, stats in combo_stats.items()
        },
        "valid_point_ratio_by_combo": {
            combo: stats["valid_point_ratio"] for combo, stats in combo_stats.items()
        },
        "same_trajectory_fraction_by_combo": {
            combo: stats["same_trajectory_fraction"] for combo, stats in combo_stats.items()
        },
        "trajectory_variance_by_combo": {
            combo: stats["trajectory_variance"] for combo, stats in combo_stats.items()
        },
        "target_side_box_search_used_ratio": float(np.mean(target_side_flags)) if target_side_flags else 1.0,
        "query_frame_only_extraction_alternative_clip_count": query_frame_only_eval_count,
        "query_frame_only_extraction_alternative_exact_blocker": "not_recomputed_for_20_clips_in_v25_partial_freeze; current cache carries teacher-only target-side-box extraction path",
        "comparison_target_box_vs_query_frame_only": "missing_for_full_20_clip_v25_audit",
        "teacher_target_construction_acceptable_for_training": bool(
            cache_report.get("valid_point_ratio", 0.0) >= 0.4 and cache_report.get("processed_clip_count", 0) > 0
        ),
        "visibility_quality_acceptable": bool(np.mean(overall_vis) >= 0.4 if overall_vis else False),
        "extraction_quality_acceptable": False,
        "overall_valid_point_ratio": float(np.mean(overall_valid)) if overall_valid else 0.0,
        "overall_estimated_visibility_coverage": float(np.mean(overall_vis)) if overall_vis else 0.0,
        "overall_trajectory_variance": float(np.mean(overall_var)) if overall_var else 0.0,
        "cache_report_path": str(V25_CACHE_REPORT.relative_to(ROOT)),
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM TraceAnything Visibility Extraction Audit V25",
        payload,
        [
            "native_visibility_available",
            "estimated_visibility_method",
            "overall_valid_point_ratio",
            "overall_estimated_visibility_coverage",
            "target_side_box_search_used_ratio",
            "query_frame_only_extraction_alternative_clip_count",
            "teacher_target_construction_acceptable_for_training",
            "visibility_quality_acceptable",
            "extraction_quality_acceptable",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
