#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))
from stwm_v15b_forensic_utils_20260502 import (  # noqa: E402
    ROOT,
    dump_json,
    is_fallback_source,
    load_cache,
    now_utc,
    summarize_sources_by_m,
    teacher_source_distribution,
    write_doc,
)


def _exists_any(patterns: list[str]) -> list[str]:
    out: list[str] = []
    for pattern in patterns:
        out.extend(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern))
    return sorted(set(out))


def main() -> int:
    m_values = [1, 128, 512]
    per_m: dict[str, Any] = {}
    total_points = 0
    fallback_points = 0
    for m in m_values:
        z = load_cache(m)
        row = teacher_source_distribution(z)
        source = str(row["source"])
        row["cache_path"] = str((ROOT / f"outputs/cache/stwm_object_dense_trace_v15/M{m}/object_dense_trace_cache.npz").relative_to(ROOT))
        row["teacher_source_is_fallback_or_pseudo"] = is_fallback_source(source)
        per_m[f"M{m}"] = row
        total_points += int(row["point_count"])
        fallback_points += int(row["point_count"]) if row["teacher_source_is_fallback_or_pseudo"] else 0
    fallback_ratio = float(fallback_points / max(total_points, 1))
    teacher_availability = {
        "TraceAnything_path_exists": bool((ROOT / "third_party/TraceAnything").exists()),
        "TraceAnything_paths": _exists_any(["third_party/TraceAnything", "third_party/TraceAnything/**/README*"])[:20],
        "CoTracker_repo_exists": bool((ROOT / "baselines/repos/co-tracker/cotracker").exists()),
        "CoTracker_checkpoint_paths": _exists_any(["baselines/checkpoints/cotracker/**/*", "baselines/checkpoints/*cotracker*"]),
        "TAPIR_paths": _exists_any(["third_party/**/tapir*", "baselines/**/tapir*"]),
        "PointOdyssey_paths": _exists_any(["data/external/**/PointOdyssey*", "data/external/**/pointodyssey*"]),
        "internal_stage1_trace_cache_exists": bool((ROOT / "data/processed/stage2_tusb_v3_predecode_cache_20260418").exists()),
    }
    inference_logs = _exists_any(
        [
            "logs/**/*cotracker*.log",
            "logs/**/*TraceAnything*.log",
            "logs/**/*tapir*.log",
            "logs/**/*object_dense*teacher*.log",
            "logs/**/*dense_trace*teacher*.log",
        ]
    )
    payload: dict[str, Any] = {
        "audit_name": "stwm_v15_teacher_source_forensics",
        "generated_at_utc": now_utc(),
        "m_values_checked": m_values,
        "per_m_teacher_source_distribution": per_m,
        "teacher_source_distribution_summary": summarize_sources_by_m(m_values),
        "real_teacher_tracks_exist": False,
        "official_teacher_used_in_v15_cache": False,
        "used_TraceAnything": False,
        "used_CoTracker_or_CoTracker3": False,
        "used_TAPIR": False,
        "used_PointOdyssey_GT": False,
        "used_internal_Stage1": False,
        "fallback_or_bbox_grid_synthetic_track_ratio_by_point": fallback_ratio,
        "teacher_availability_on_disk": teacher_availability,
        "official_checkpoint_path_or_command_recorded_for_v15": False,
        "gpu_teacher_inference_log_exists": bool(inference_logs),
        "gpu_teacher_inference_log_candidates": inference_logs[:50],
        "object_dense_trace_claim_allowed": bool(fallback_ratio <= 0.20 and False),
        "claim_blocker": (
            "V15 object-dense cache teacher_source is mask_bbox_relative_pseudo_track for all checked M values; "
            "no CoTracker/TraceAnything/TAPIR/PointOdyssey teacher command, checkpoint provenance, or GPU inference log is recorded."
        ),
    }
    out = ROOT / "reports/stwm_v15_teacher_source_forensics_20260502.json"
    doc = ROOT / "docs/STWM_V15_TEACHER_SOURCE_FORENSICS_20260502.md"
    dump_json(out, payload)
    write_doc(
        doc,
        "STWM V15 Teacher Source Forensics",
        payload,
        [
            "real_teacher_tracks_exist",
            "official_teacher_used_in_v15_cache",
            "fallback_or_bbox_grid_synthetic_track_ratio_by_point",
            "gpu_teacher_inference_log_exists",
            "object_dense_trace_claim_allowed",
            "claim_blocker",
        ],
    )
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
