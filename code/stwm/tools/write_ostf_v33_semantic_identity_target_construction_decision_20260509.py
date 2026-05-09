#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_semantic_identity_target_construction_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_SEMANTIC_IDENTITY_TARGET_CONSTRUCTION_DECISION_20260509.md"
CLAIM_FIX_REPORT = ROOT / "reports/stwm_ostf_v33_claim_boundary_fix_20260509.json"
CLAIM_FIX_DOC = ROOT / "docs/STWM_OSTF_V33_CLAIM_BOUNDARY_FIX_20260509.md"


def load(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    truth = load("reports/stwm_ostf_v33_latest_repo_truth_refresh_20260509.json") or load("reports/stwm_ostf_v33_latest_repo_truth_audit_20260509.json")
    freeze = load("reports/stwm_ostf_v33_trajectory_backbone_freeze_audit_20260509.json")
    source = load("reports/stwm_ostf_v33_pointodyssey_semantic_identity_source_audit_20260509.json")
    build = load("reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json")
    visual = load("reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json")
    coverage = load("reports/stwm_ostf_v33_dense_field_target_coverage_20260509.json")
    contract = load("reports/stwm_ostf_v33_semantic_identity_code_contract_audit_20260509.json")
    smoke = load("reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json")
    point_ok = bool(build.get("point_identity_coverage_ratio", 0.0) >= 0.95)
    inst_ok = bool(build.get("instance_identity_coverage_ratio", 0.0) >= 0.5)
    class_ok = bool(source.get("class_semantic_available"))
    visual_ok = bool(visual.get("visual_teacher_semantic_targets_available"))
    leakage_safe = bool(build.get("leakage_safe"))
    smoke_passed: bool | str = bool(smoke.get("smoke_passed")) if smoke else "not_run"
    smoke_level = str(smoke.get("smoke_level", "not_run")) if smoke else "not_run"
    trajectory_degraded = smoke.get("whether_trajectory_degraded") if smoke else "not_run"
    v30_consumed = bool(smoke.get("v30_checkpoint_consumed_in_smoke"))
    real_frozen_v30_head = bool(smoke.get("integrated_v30_backbone_used")) and trajectory_degraded is False
    head_only_passed = bool(smoke_passed is True and not real_frozen_v30_head)
    integrated_passed = bool(smoke_passed is True and real_frozen_v30_head and trajectory_degraded is False)
    identity_world_model_claim_allowed = integrated_passed
    semantic_world_model_claim_allowed = False
    if inst_ok and leakage_safe and integrated_passed:
        next_step = "run_v33_identity_seed42_pilot"
    elif smoke_passed is True and not real_frozen_v30_head:
        next_step = "integrate_sidecar_into_v30_dataset_and_trainer"
    elif point_ok and not inst_ok:
        next_step = "build_visual_teacher_semantic_prototypes"
    elif not leakage_safe or build.get("assignment_confidence_mean", 0.0) < 0.5:
        next_step = "fix_identity_target_mapping"
    else:
        next_step = "stop_and_return_to_data_source_for_semantic_labels"
    payload = {
        "generated_at_utc": utc_now(),
        "trajectory_backbone_frozen_as_v30_m128": freeze.get("official_current_trajectory_backbone") == "V30_M128",
        "v30_backbone_frozen": freeze.get("official_current_trajectory_backbone") == "V30_M128",
        "v33_existing_code_was_partial": bool(truth.get("eval_stub_detected", False) or truth.get("v30_checkpoint_not_consumed", False)),
        "v33_eval_stub_detected": bool(truth.get("eval_stub_detected", False)),
        "v30_checkpoint_consumed_in_smoke": v30_consumed,
        "v33_target_construction_missing_before_this_round": bool(truth.get("v33_target_construction_missing", True)),
        "pointodyssey_identity_target_available": point_ok,
        "point_identity_targets_available": point_ok,
        "pointodyssey_instance_identity_available": inst_ok,
        "instance_identity_targets_available": inst_ok,
        "pointodyssey_class_semantic_available": class_ok,
        "class_semantic_targets_available": class_ok,
        "visual_teacher_semantic_needed": bool(visual.get("visual_teacher_semantic_needed", not class_ok)),
        "visual_teacher_preflight_passed": bool(
            visual.get("rgb_crop_extraction_available")
            and (visual.get("DINOv2_available") or visual.get("CLIP_open_clip_available") or visual.get("SigLIP_available"))
        ),
        "visual_teacher_semantic_targets_available": visual_ok,
        "identity_target_coverage": build.get("point_identity_coverage_ratio", 0.0),
        "instance_identity_coverage": build.get("instance_identity_coverage_ratio", 0.0),
        "leakage_safe": leakage_safe,
        "m128_target_coverage": coverage.get("M128", {}),
        "m512_target_coverage": coverage.get("M512", {}),
        "m1024_target_coverage": coverage.get("M1024", {}),
        "smoke_passed": smoke_passed if smoke else "not_run",
        "identity_smoke_passed": smoke_passed if smoke else "not_run",
        "smoke_level": smoke_level,
        "trajectory_degraded": trajectory_degraded if smoke else "not_run",
        "semantic_identity_code_ready": not bool(contract.get("v33_eval_is_stub", True)) and bool(contract.get("v33_head_trainer_loads_v30_checkpoint")),
        "head_only_target_learnability_passed": head_only_passed,
        "integrated_v30_semantic_identity_passed": integrated_passed,
        "identity_world_model_claim_allowed": identity_world_model_claim_allowed,
        "semantic_world_model_claim_allowed": semantic_world_model_claim_allowed,
        "semantic_identity_field_claim_preliminary": bool(identity_world_model_claim_allowed and semantic_world_model_claim_allowed),
        "recommended_next_step": next_step,
        "source_audit_path": "reports/stwm_ostf_v33_pointodyssey_semantic_identity_source_audit_20260509.json",
        "target_build_path": "reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json",
        "smoke_summary_path": "reports/stwm_ostf_v33_semantic_identity_smoke_summary_20260509.json" if smoke else None,
    }
    dump_json(REPORT, payload)
    claim_payload = {
        "generated_at_utc": utc_now(),
        "head_only_target_learnability_passed": head_only_passed,
        "integrated_v30_semantic_identity_passed": integrated_passed,
        "identity_world_model_claim_allowed": identity_world_model_claim_allowed,
        "semantic_world_model_claim_allowed": semantic_world_model_claim_allowed,
        "semantic_identity_field_claim_preliminary": bool(identity_world_model_claim_allowed and semantic_world_model_claim_allowed),
        "integrated_v30_backbone_used": bool(smoke.get("integrated_v30_backbone_used", False)) if smoke else False,
        "trajectory_degraded": trajectory_degraded if smoke else "not_run",
        "exact_reason": "Head-only target learnability is not an integrated V30 world-model claim; class/visual semantic targets are still absent.",
    }
    dump_json(CLAIM_FIX_REPORT, claim_payload)
    write_doc(DOC, "STWM OSTF V33 Semantic Identity Target Construction Decision", payload, [
        "trajectory_backbone_frozen_as_v30_m128",
        "v33_existing_code_was_partial",
        "v33_eval_stub_detected",
        "v30_checkpoint_consumed_in_smoke",
        "pointodyssey_identity_target_available",
        "pointodyssey_instance_identity_available",
        "pointodyssey_class_semantic_available",
        "visual_teacher_semantic_needed",
        "visual_teacher_preflight_passed",
        "leakage_safe",
        "identity_smoke_passed",
        "smoke_level",
        "trajectory_degraded",
        "semantic_identity_field_claim_preliminary",
        "recommended_next_step",
    ])
    write_doc(CLAIM_FIX_DOC, "STWM OSTF V33 Claim Boundary Fix", claim_payload, [
        "head_only_target_learnability_passed",
        "integrated_v30_semantic_identity_passed",
        "identity_world_model_claim_allowed",
        "semantic_world_model_claim_allowed",
        "semantic_identity_field_claim_preliminary",
        "integrated_v30_backbone_used",
        "trajectory_degraded",
        "exact_reason",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
