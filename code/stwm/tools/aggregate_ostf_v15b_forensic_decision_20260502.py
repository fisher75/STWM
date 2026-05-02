#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent))
from stwm_v15b_forensic_utils_20260502 import ROOT, dump_json, load_json, now_utc, write_doc  # noqa: E402


def main() -> int:
    teacher = load_json(ROOT / "reports/stwm_v15_teacher_source_forensics_20260502.json")
    identity = load_json(ROOT / "reports/stwm_v15_persistent_point_identity_audit_20260502.json")
    metric = load_json(ROOT / "reports/stwm_v15_multitrace_metric_fairness_20260502.json")
    budget = load_json(ROOT / "reports/stwm_v15_training_budget_audit_20260502.json")
    viz = load_json(ROOT / "reports/stwm_v15_forensic_visualization_20260502.json")
    real_teacher = bool(teacher.get("real_teacher_tracks_exist", False))
    persistent = bool(identity.get("persistent_point_identity_valid", False))
    fake_dense = bool(identity.get("fake_dense_or_anchor_copied", True))
    comparable = bool(metric.get("M1_L1_vs_M512_point_L1_comparable", False))
    budget_short = bool(budget.get("training_time_too_short_for_M512_conclusion", True))
    if not real_teacher:
        interp = "weak_teacher"
        next_step = "fix_teacher_cache"
    elif not persistent or fake_dense:
        interp = "invalid_cache"
        next_step = "fix_teacher_cache"
    elif not comparable:
        interp = "unfair_metric"
        next_step = "rerun_multitrace_with_real_teacher"
    elif budget_short:
        interp = "insufficient_training"
        next_step = "increase_training_capacity"
    else:
        interp = "real_dense_failure"
        next_step = "redesign_dense_to_unit_encoder"
    payload: dict[str, Any] = {
        "audit_name": "stwm_ostf_v15b_forensic_decision",
        "generated_at_utc": now_utc(),
        "teacher_forensic_path": "reports/stwm_v15_teacher_source_forensics_20260502.json",
        "persistent_identity_audit_path": "reports/stwm_v15_persistent_point_identity_audit_20260502.json",
        "metric_fairness_path": "reports/stwm_v15_multitrace_metric_fairness_20260502.json",
        "training_budget_audit_path": "reports/stwm_v15_training_budget_audit_20260502.json",
        "forensic_visualization_path": "reports/stwm_v15_forensic_visualization_20260502.json",
        "real_teacher_tracks_exist": real_teacher,
        "persistent_point_identity_valid": persistent,
        "fake_dense_or_anchor_copied": fake_dense,
        "future_leakage_detected": bool(identity.get("future_leakage_detected", False)),
        "M1_vs_M512_metric_comparable": comparable,
        "M512_failure_interpretation": interp,
        "object_dense_trace_claim_allowed": False,
        "object_dense_trace_claim_allowed_reason": (
            "V15B found non-empty object-internal point tensors and forensic GIFs, but teacher_source is fallback "
            "mask_bbox_relative_pseudo_track, persistent identity is index-based rather than physical tracking, "
            "and M1/M512 losses are not a fair final dense metric."
        ),
        "visualization_gif_count": viz.get("gif_count", 0),
        "proceed_to": next_step,
    }
    out = ROOT / "reports/stwm_ostf_v15b_forensic_decision_20260502.json"
    doc = ROOT / "docs/STWM_OSTF_V15B_FORENSIC_DECISION_20260502.md"
    dump_json(out, payload)
    write_doc(
        doc,
        "STWM OSTF V15B Forensic Decision",
        payload,
        [
            "real_teacher_tracks_exist",
            "persistent_point_identity_valid",
            "fake_dense_or_anchor_copied",
            "future_leakage_detected",
            "M1_vs_M512_metric_comparable",
            "M512_failure_interpretation",
            "object_dense_trace_claim_allowed",
            "proceed_to",
        ],
    )
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
