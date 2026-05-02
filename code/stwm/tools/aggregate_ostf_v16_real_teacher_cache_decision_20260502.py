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


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM OSTF V16 Real Teacher Cache Decision", ""]
    for key in [
        "full_real_teacher_cache_ready",
        "M128_ready",
        "M512_ready",
        "H8_ready",
        "H16_ready",
        "processed_clip_count",
        "valid_point_ratio",
        "visualization_ready",
        "proceed_to",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    audit = _load("reports/stwm_real_object_dense_teacher_cache_audit_v16_20260502.json")
    viz = _load("reports/stwm_real_teacher_object_dense_visualization_v16_20260502.json")
    manifest = _load("reports/stwm_real_teacher_cache_artifact_manifest_v16_20260502.json")
    per = audit.get("per_combo", {})
    m128_ready = bool(per.get("M128_H8", {}).get("success_gate_passed")) and bool(per.get("M128_H16", {}).get("success_gate_passed"))
    m512_ready = bool(per.get("M512_H8", {}).get("success_gate_passed")) and bool(per.get("M512_H16", {}).get("success_gate_passed"))
    h8_ready = bool(per.get("M128_H8", {}).get("success_gate_passed")) and bool(per.get("M512_H8", {}).get("success_gate_passed"))
    h16_ready = bool(per.get("M128_H16", {}).get("success_gate_passed")) and bool(per.get("M512_H16", {}).get("success_gate_passed"))
    full_ready = bool(audit.get("success_gate_passed")) and bool(viz.get("visualization_ready"))
    if full_ready:
        proceed = "train_full_ostf_multitrace_model"
    elif not h16_ready:
        proceed = "fix_window_selection"
    elif audit.get("processed_clip_count", 0) < 2000:
        proceed = "reduce_scale_due_to_runtime"
    else:
        proceed = "block_due_to_teacher_failure"
    payload = {
        "audit_name": "stwm_ostf_v16_real_teacher_cache_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_cache_report_path": "reports/stwm_cotracker_object_dense_teacher_v16_20260502.json",
        "cache_audit_path": "reports/stwm_real_object_dense_teacher_cache_audit_v16_20260502.json",
        "visualization_path": "reports/stwm_real_teacher_object_dense_visualization_v16_20260502.json",
        "artifact_manifest_path": "reports/stwm_real_teacher_cache_artifact_manifest_v16_20260502.json",
        "full_real_teacher_cache_ready": full_ready,
        "M128_ready": m128_ready,
        "M512_ready": m512_ready,
        "H8_ready": h8_ready,
        "H16_ready": h16_ready,
        "processed_clip_count": audit.get("processed_clip_count", 0),
        "point_count": audit.get("point_count", 0),
        "M_H_completed": [k for k, v in per.items() if v.get("success_gate_passed")],
        "valid_point_ratio": audit.get("valid_point_ratio", 0.0),
        "visualization_ready": bool(viz.get("visualization_ready")),
        "artifact_pack_path": manifest.get("artifact_pack_path"),
        "proceed_to": proceed,
    }
    _dump(ROOT / "reports/stwm_ostf_v16_real_teacher_cache_decision_20260502.json", payload)
    _write_doc(ROOT / "docs/STWM_OSTF_V16_REAL_TEACHER_CACHE_DECISION_20260502.md", payload)
    print("reports/stwm_ostf_v16_real_teacher_cache_decision_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
