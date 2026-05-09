#!/usr/bin/env python3
from __future__ import annotations

import json

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_3_structured_semantic_identity_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_3_STRUCTURED_SEMANTIC_IDENTITY_DECISION_20260509.md"


def load(rel: str) -> dict:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    artifact = load("reports/stwm_ostf_v33_3_artifact_truth_20260509.json")
    vocab = load("reports/stwm_ostf_v33_3_semantic_prototype_vocab_20260509.json")
    targets = load("reports/stwm_ostf_v33_3_semantic_prototype_targets_20260509.json")
    subset = load("reports/stwm_ostf_v33_3_balanced_hard_subset_20260509.json")
    summary = load("reports/stwm_ostf_v33_3_structured_semantic_identity_smoke_summary_20260509.json")
    smoke_decision = load("reports/stwm_ostf_v33_3_structured_semantic_identity_smoke_decision_20260509.json")
    vis = load("reports/stwm_ostf_v33_3_structured_semantic_visualization_manifest_20260509.json")
    test = summary.get("test_metrics", {}) if isinstance(summary.get("test_metrics", {}), dict) else {}
    val_test_gap = test.get("val_test_gap")
    split_shift = bool(test.get("split_shift_suspected", False))
    artifact_ok = bool(artifact.get("artifact_truth_ok", False))
    claim_contra = bool(artifact.get("claim_contradiction_detected", False))
    proto_ok = bool(vocab.get("prototype_vocab_built", False))
    target_ok = bool(targets.get("prototype_targets_built", False))
    subset_ok = bool(subset.get("balanced_hard_subset_built", False))
    identity_ok = bool((test.get("hard_identity_ROC_AUC") or 0.0) >= 0.60 and (test.get("hard_identity_balanced_accuracy") or 0.0) >= 0.55 and (test.get("identity_embedding_retrieval_top1") or 0.0) > float(test.get("identity_retrieval_prior_top1") or 0.0))
    semantic_ok = bool(test.get("semantic_copy_baseline_beaten", False))
    if not artifact_ok:
        next_step = "fix_artifact_truth_and_rerun"
    elif not proto_ok or not target_ok:
        next_step = "fix_semantic_prototype_vocab"
    elif not subset_ok:
        next_step = "fix_balanced_hard_subset"
    elif not identity_ok:
        next_step = "fix_identity_contrastive_loss"
    elif not semantic_ok:
        next_step = "fix_semantic_prototype_loss"
    elif split_shift or claim_contra:
        next_step = "fix_split_shift_or_eval_protocol"
    else:
        next_step = "run_v33_3_h64_h96_smoke"
    payload = {
        "generated_at_utc": utc_now(),
        "artifact_truth_ok": artifact_ok,
        "claim_contradiction_detected": claim_contra,
        "prototype_vocab_built": proto_ok,
        "prototype_targets_built": target_ok,
        "balanced_hard_subset_built": subset_ok,
        "v30_checkpoint_loaded": bool(test.get("v30_checkpoint_loaded", False)),
        "v30_backbone_frozen": bool(test.get("v30_backbone_frozen", False)),
        "integrated_v30_backbone_used": bool(test.get("integrated_v30_backbone_used", False)),
        "observed_visual_teacher_context_used": bool(test.get("observed_visual_teacher_context_used", False)),
        "future_teacher_leakage_detected": bool(test.get("future_teacher_leakage_detected", True)),
        "identity_contrastive_loss_active": bool(test.get("identity_contrastive_loss_active", False)),
        "semantic_proto_loss_active": bool(test.get("semantic_proto_loss_active", False)),
        "identity_embedding_retrieval_top1": test.get("identity_embedding_retrieval_top1"),
        "identity_embedding_retrieval_top5": test.get("identity_embedding_retrieval_top5"),
        "hard_identity_ROC_AUC": test.get("hard_identity_ROC_AUC"),
        "hard_identity_balanced_accuracy": test.get("hard_identity_balanced_accuracy"),
        "semantic_proto_top1": test.get("semantic_proto_top1"),
        "semantic_proto_top5": test.get("semantic_proto_top5"),
        "semantic_copy_baseline_beaten": bool(test.get("semantic_copy_baseline_beaten", False)),
        "val_test_gap": val_test_gap,
        "split_shift_suspected": split_shift,
        "trajectory_degraded": bool(test.get("trajectory_degraded", True)),
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "integrated_identity_field_claim_allowed": bool(smoke_decision.get("integrated_identity_field_claim_allowed", False) and not claim_contra and not split_shift),
        "integrated_semantic_field_claim_allowed": bool(smoke_decision.get("integrated_semantic_field_claim_allowed", False) and not claim_contra and not split_shift),
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.3 Structured Semantic Identity Decision", payload, ["artifact_truth_ok", "claim_contradiction_detected", "prototype_vocab_built", "prototype_targets_built", "balanced_hard_subset_built", "hard_identity_ROC_AUC", "hard_identity_balanced_accuracy", "identity_embedding_retrieval_top1", "semantic_proto_top1", "semantic_proto_top5", "semantic_copy_baseline_beaten", "val_test_gap", "split_shift_suspected", "trajectory_degraded", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
