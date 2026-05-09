#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_semantic_identity_target_construction_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_SEMANTIC_IDENTITY_TARGET_CONSTRUCTION_DECISION_20260509.md"


def load(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    truth = load("reports/stwm_ostf_v33_latest_repo_truth_audit_20260509.json")
    freeze = load("reports/stwm_ostf_v33_trajectory_backbone_freeze_audit_20260509.json")
    source = load("reports/stwm_ostf_v33_pointodyssey_semantic_identity_source_audit_20260509.json")
    build = load("reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json")
    visual = load("reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json")
    coverage = load("reports/stwm_ostf_v33_dense_field_target_coverage_20260509.json")
    smoke = load("reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json")
    point_ok = bool(build.get("point_identity_coverage_ratio", 0.0) >= 0.95)
    inst_ok = bool(build.get("instance_identity_coverage_ratio", 0.0) >= 0.5)
    class_ok = bool(source.get("class_semantic_available"))
    visual_ok = bool(visual.get("visual_teacher_semantic_targets_available"))
    leakage_safe = bool(build.get("leakage_safe"))
    smoke_passed = bool(smoke.get("smoke_passed")) if smoke else False
    trajectory_degraded = bool(smoke.get("whether_trajectory_degraded")) if smoke else None
    if inst_ok and leakage_safe and smoke_passed and trajectory_degraded is False:
        next_step = "run_v33_identity_seed42_pilot"
    elif point_ok and not inst_ok:
        next_step = "build_visual_teacher_semantic_prototypes"
    elif not leakage_safe or build.get("assignment_confidence_mean", 0.0) < 0.5:
        next_step = "fix_identity_target_mapping"
    else:
        next_step = "stop_and_return_to_data_source_for_semantic_labels"
    payload = {
        "generated_at_utc": utc_now(),
        "v30_backbone_frozen": freeze.get("official_current_trajectory_backbone") == "V30_M128",
        "v33_target_construction_missing_before_this_round": bool(truth.get("v33_target_construction_missing", True)),
        "point_identity_targets_available": point_ok,
        "instance_identity_targets_available": inst_ok,
        "class_semantic_targets_available": class_ok,
        "visual_teacher_semantic_targets_available": visual_ok,
        "leakage_safe": leakage_safe,
        "m128_target_coverage": coverage.get("M128", {}),
        "m512_target_coverage": coverage.get("M512", {}),
        "m1024_target_coverage": coverage.get("M1024", {}),
        "smoke_passed": smoke_passed if smoke else "not_run",
        "trajectory_degraded": trajectory_degraded if smoke else "not_run",
        "semantic_identity_field_claim_preliminary": bool(inst_ok and smoke_passed and trajectory_degraded is False),
        "recommended_next_step": next_step,
        "source_audit_path": "reports/stwm_ostf_v33_pointodyssey_semantic_identity_source_audit_20260509.json",
        "target_build_path": "reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json",
        "smoke_summary_path": "reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json" if smoke else None,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 Semantic Identity Target Construction Decision", payload, [
        "v30_backbone_frozen",
        "point_identity_targets_available",
        "instance_identity_targets_available",
        "class_semantic_targets_available",
        "visual_teacher_semantic_targets_available",
        "leakage_safe",
        "smoke_passed",
        "trajectory_degraded",
        "semantic_identity_field_claim_preliminary",
        "recommended_next_step",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
