#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def _load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM OSTF V15C Real Teacher Decision", ""]
    for key in [
        "real_teacher_tracks_exist",
        "persistent_point_identity_valid",
        "teacher_source",
        "average_points_per_object",
        "average_points_per_scene",
        "visualization_ready",
        "OSTF_real_teacher_pilot_metrics",
        "next_step_choice",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    teacher = _load("reports/stwm_cotracker_object_dense_teacher_v15c_20260502.json")
    audit = _load("reports/stwm_real_object_dense_teacher_cache_audit_v15c_20260502.json")
    viz = _load("reports/stwm_real_teacher_object_dense_visualization_v15c_20260502.json")
    pilot = _load("reports/stwm_ostf_real_teacher_pilot_v15c_20260502.json")
    real = bool(audit.get("real_teacher_tracks_exist", False))
    persistent = bool(audit.get("persistent_point_identity_valid", False))
    viz_ready = bool(viz.get("visualization_ready", False))
    if not real or not persistent:
        next_step = "fix_cotracker_integration"
    elif not viz_ready:
        next_step = "fix_cotracker_integration"
    else:
        next_step = "proceed_to_full_real_teacher_cache"
    payload = {
        "audit_name": "stwm_ostf_v15c_real_teacher_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cotracker_teacher_runner_path": "reports/stwm_cotracker_object_dense_teacher_v15c_20260502.json",
        "cache_audit_path": "reports/stwm_real_object_dense_teacher_cache_audit_v15c_20260502.json",
        "visualization_path": "reports/stwm_real_teacher_object_dense_visualization_v15c_20260502.json",
        "real_teacher_pilot_path": "reports/stwm_ostf_real_teacher_pilot_v15c_20260502.json",
        "real_teacher_tracks_exist": real,
        "persistent_point_identity_valid": persistent,
        "teacher_source": "cotracker_official" if real else teacher.get("teacher_source", "unknown"),
        "average_points_per_object": audit.get("average_points_per_object"),
        "average_points_per_scene": audit.get("average_points_per_scene"),
        "visualization_ready": viz_ready,
        "OSTF_real_teacher_pilot_metrics": {
            "M1": pilot.get("M1_metrics"),
            "M128": pilot.get("M128_metrics"),
            "metric_note": pilot.get("metric_note"),
        },
        "processed_clip_count": teacher.get("processed_clip_count"),
        "split_counts": audit.get("split_counts"),
        "valid_point_ratio": audit.get("valid_point_ratio"),
        "object_dense_physical_teacher_cache_ready": bool(real and persistent),
        "pilot_interpretation": (
            "CoTracker teacher integration is successful. M128 pilot is not yet stronger than M1 under the tiny 1000-step head, "
            "so this is a cache/teacher success plus model-capacity follow-up, not a full OSTF claim."
        ),
        "next_step_choice": next_step,
    }
    out = ROOT / "reports/stwm_ostf_v15c_real_teacher_decision_20260502.json"
    doc = ROOT / "docs/STWM_OSTF_V15C_REAL_TEACHER_DECISION_20260502.md"
    _dump(out, payload)
    _write_doc(doc, payload)
    print("reports/stwm_ostf_v15c_real_teacher_decision_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
