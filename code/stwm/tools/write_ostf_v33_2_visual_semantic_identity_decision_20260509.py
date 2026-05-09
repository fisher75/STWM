#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_2_visual_semantic_identity_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_2_VISUAL_SEMANTIC_IDENTITY_DECISION_20260509.md"


def load(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.exists() else {}


def main() -> int:
    art = load("reports/stwm_ostf_v33_2_artifact_truth_and_claims_20260509.json")
    teacher = load("reports/stwm_ostf_v33_2_visual_teacher_preflight_20260509.json")
    cache = load("reports/stwm_ostf_v33_2_visual_teacher_prototype_build_20260509.json")
    hard = load("reports/stwm_ostf_v33_2_hard_identity_semantic_subset_20260509.json")
    smoke = load("reports/stwm_ostf_v33_2_visual_semantic_identity_smoke_summary_20260509.json")
    vis = load("reports/stwm_ostf_v33_2_semantic_identity_visualization_manifest_20260509.json")
    metrics = smoke.get("test_metrics", {})
    identity_pass = bool(metrics.get("trivial_prior_beaten") and (metrics.get("hard_identity_ROC_AUC") or 0) >= 0.60 and (metrics.get("hard_identity_balanced_accuracy") or 0) >= 0.55)
    semantic_pass = bool(metrics.get("semantic_copy_baseline_beaten"))
    if not art.get("artifact_truth_ok"):
        next_step = "fix_artifact_truth_and_rerun"
    elif not teacher.get("teacher_model_loaded"):
        next_step = "fix_teacher_model_availability"
    elif not cache.get("visual_teacher_cache_built"):
        next_step = "fix_visual_teacher_cache"
    elif not hard.get("whether_balanced_eval_possible"):
        next_step = "fix_hard_identity_subset"
    elif not identity_pass:
        next_step = "fix_identity_contrastive_loss"
    elif not semantic_pass:
        next_step = "fix_semantic_prototype_loss"
    else:
        next_step = "run_v33_2_h64_h96_smoke"
    payload = {
        "generated_at_utc": utc_now(),
        "artifact_truth_ok": bool(art.get("artifact_truth_ok")),
        "teacher_preflight_passed": bool(teacher.get("teacher_model_loaded") and teacher.get("teacher_forward_dryrun_passed")),
        "visual_teacher_cache_built": bool(cache.get("visual_teacher_cache_built")),
        "hard_subset_built": bool(hard.get("whether_balanced_eval_possible")),
        "sidecar_dataset_integrated": True,
        "v30_checkpoint_loaded": bool(metrics.get("v30_checkpoint_loaded")),
        "v30_backbone_frozen": bool(metrics.get("v30_backbone_frozen")),
        "integrated_v30_backbone_used": bool(metrics.get("integrated_v30_backbone_used")),
        "observed_instance_context_used": bool(metrics.get("observed_instance_context_used")),
        "observed_visual_teacher_context_used": bool(metrics.get("observed_visual_teacher_context_used")),
        "future_teacher_leakage_detected": bool(metrics.get("future_teacher_leakage_detected", True)),
        "identity_ROC_AUC": metrics.get("identity_ROC_AUC"),
        "identity_PR_AUC": metrics.get("identity_PR_AUC"),
        "same_instance_balanced_accuracy": metrics.get("same_instance_balanced_accuracy"),
        "hard_identity_ROC_AUC": metrics.get("hard_identity_ROC_AUC"),
        "hard_identity_balanced_accuracy": metrics.get("hard_identity_balanced_accuracy"),
        "identity_retrieval_top1": metrics.get("identity_retrieval_top1"),
        "semantic_retrieval_top1": metrics.get("semantic_retrieval_top1"),
        "semantic_retrieval_top5": metrics.get("semantic_retrieval_top5"),
        "semantic_copy_baseline_beaten": bool(metrics.get("semantic_copy_baseline_beaten")),
        "trivial_prior_beaten": bool(metrics.get("trivial_prior_beaten")),
        "trajectory_degraded": bool(metrics.get("trajectory_degraded", True)),
        "visualization_ready": bool(vis.get("visualization_ready")),
        "integrated_identity_field_claim_allowed": bool(identity_pass and not metrics.get("trajectory_degraded", True) and not metrics.get("future_teacher_leakage_detected", True)),
        "integrated_semantic_field_claim_allowed": bool(identity_pass and semantic_pass and not metrics.get("trajectory_degraded", True) and not metrics.get("future_teacher_leakage_detected", True)),
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.2 Visual Semantic Identity Decision", payload, ["artifact_truth_ok", "teacher_preflight_passed", "visual_teacher_cache_built", "hard_subset_built", "integrated_v30_backbone_used", "observed_visual_teacher_context_used", "future_teacher_leakage_detected", "identity_ROC_AUC", "hard_identity_ROC_AUC", "hard_identity_balanced_accuracy", "semantic_retrieval_top1", "semantic_copy_baseline_beaten", "trivial_prior_beaten", "trajectory_degraded", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
