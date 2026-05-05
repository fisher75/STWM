#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, load_json, write_doc


REPORT_PATH = ROOT / "reports/stwm_ostf_v2_redesign_decision_20260502.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V2_REDESIGN_DECISION_20260502.md"


def main() -> int:
    teacher = load_json(ROOT / "reports/stwm_traceanything_teacher_pilot_v2_20260502.json")
    hard = load_json(ROOT / "reports/stwm_ostf_hard_benchmark_v2_20260502.json")
    ext = load_json(ROOT / "reports/stwm_external_point_benchmark_audit_v2_20260502.json")
    bridge = load_json(ROOT / "reports/stwm_ostf_semantic_identity_bridge_v2_20260502.json")
    trace_run = bool(teacher.get("traceanything_teacher_runnable", False))
    trace_valid = bool(teacher.get("traceanything_object_tracks_valid", False))
    hard_ready = bool(hard.get("hard_benchmark_ready", False))
    h32_h64 = bool(hard.get("h32_h64_feasible", False))
    ext_ready = bool(ext.get("external_point_benchmark_ready", False))
    bridge_ready = bool(bridge.get("semantic_identity_bridge_ready", False))
    if trace_run and trace_valid and hard_ready and bridge_ready:
        next_step = "train_ostf_v2_on_traceanything_hardbench"
    elif not trace_run or not trace_valid:
        next_step = "fix_traceanything_adapter"
    elif not h32_h64 and hard_ready:
        next_step = "build_h32_hardbench"
    elif not ext_ready:
        next_step = "download_pointodyssey_tapvid"
    else:
        next_step = "pause_ostf_and_continue_FSTF_only"
    payload = {
        "audit_name": "stwm_ostf_v2_redesign_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "traceanything_teacher_runnable": trace_run,
        "traceanything_object_tracks_valid": trace_valid,
        "cotracker_vs_traceanything_comparison_summary": teacher.get("comparison_to_cotracker_same_clips", {}),
        "hard_benchmark_ready": hard_ready,
        "H32_H64_feasible": h32_h64,
        "external_point_benchmark_ready": ext_ready,
        "semantic_identity_bridge_ready": bridge_ready,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V2 Redesign Decision",
        payload,
        [
            "traceanything_teacher_runnable",
            "traceanything_object_tracks_valid",
            "hard_benchmark_ready",
            "H32_H64_feasible",
            "external_point_benchmark_ready",
            "semantic_identity_bridge_ready",
            "recommended_next_step",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
