#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_1_integrated_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_1_INTEGRATED_DECISION_20260509.md"


def load(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    artifact = load("reports/stwm_ostf_v33_1_artifact_truth_audit_20260509.json")
    contract = load("reports/stwm_ostf_v33_1_sidecar_dataset_contract_20260509.json")
    smoke = load("reports/stwm_ostf_v33_1_integrated_smoke_summary_20260509.json")
    vis = load("reports/stwm_ostf_v33_1_identity_field_visualization_manifest_20260509.json")
    visual_teacher = load("reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json")
    metrics = smoke.get("test_metrics", {})
    identity_auc = metrics.get("identity_ROC_AUC")
    identity_pr = metrics.get("identity_PR_AUC")
    trivial = bool(smoke.get("trivial_prior_beaten", False))
    horizon_dep = bool(smoke.get("horizon_dependent_identity_logits", False))
    traj_deg = bool(smoke.get("trajectory_degraded", True))
    integrated = bool(smoke.get("integrated_v30_backbone_used", False))
    viz_ready = bool(vis.get("visualization_ready", False))
    identity_allowed = bool(integrated and not traj_deg and horizon_dep and trivial and viz_ready and identity_auc is not None)
    if identity_allowed:
        next_step = "run_v33_1_integrated_h64_h96_smoke"
    elif not bool(contract.get("sidecar_dataset_integrated", False)):
        next_step = "fix_sidecar_dataset_integration"
    elif not horizon_dep:
        next_step = "fix_identity_head_horizon_dependence"
    elif traj_deg:
        next_step = "fix_trajectory_preservation"
    elif bool(visual_teacher.get("visual_teacher_semantic_needed", True)):
        next_step = "build_visual_teacher_semantic_prototypes"
    else:
        next_step = "stop_and_return_to_target_mapping"
    payload = {
        "generated_at_utc": utc_now(),
        "artifact_truth_ok": bool(artifact.get("artifact_truth_ok", False)),
        "sidecar_dataset_integrated": bool(contract.get("sidecar_dataset_integrated", False)),
        "v30_checkpoint_loaded": bool(smoke.get("v30_checkpoint_loaded", False)),
        "v30_backbone_frozen": bool(smoke.get("v30_backbone_frozen", False)),
        "integrated_v30_backbone_used": integrated,
        "observed_instance_context_used": bool(smoke.get("observed_instance_context_used", False)),
        "horizon_dependent_identity_logits": horizon_dep,
        "true_auroc_used": bool(metrics.get("true_auroc_used", False)),
        "identity_target_coverage": metrics.get("identity_target_coverage"),
        "same_instance_accuracy": metrics.get("same_instance_accuracy"),
        "same_instance_balanced_accuracy": metrics.get("same_instance_balanced_accuracy"),
        "identity_ROC_AUC": identity_auc,
        "identity_PR_AUC": identity_pr,
        "trivial_prior_beaten": trivial,
        "trajectory_minFDE_delta_vs_frozen_V30": metrics.get("trajectory_minFDE_delta_vs_frozen_V30"),
        "trajectory_degraded": traj_deg,
        "visualization_ready": viz_ready,
        "class_semantic_targets_available": False,
        "visual_teacher_semantic_needed": bool(visual_teacher.get("visual_teacher_semantic_needed", True)),
        "visual_teacher_semantic_targets_available": bool(visual_teacher.get("visual_teacher_semantic_targets_available", False)),
        "integrated_identity_field_claim_allowed": identity_allowed,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.1 Integrated Decision", payload, ["artifact_truth_ok", "sidecar_dataset_integrated", "v30_checkpoint_loaded", "v30_backbone_frozen", "integrated_v30_backbone_used", "observed_instance_context_used", "horizon_dependent_identity_logits", "true_auroc_used", "identity_ROC_AUC", "identity_PR_AUC", "trivial_prior_beaten", "trajectory_degraded", "visualization_ready", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
