#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


FORENSICS = ROOT / "reports/stwm_ostf_v33_7_identity_training_forensics_20260509.json"
COVERAGE = ROOT / "reports/stwm_ostf_v33_7_h32_m128_complete_target_coverage_20260509.json"
MASKS = ROOT / "reports/stwm_ostf_v33_7_hard_identity_train_mask_build_20260509.json"
TRAIN = ROOT / "reports/stwm_ostf_v33_7_identity_belief_train_summary_20260509.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v33_7_identity_belief_eval_decision_20260509.json"
REPORT = ROOT / "reports/stwm_ostf_v33_7_identity_belief_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_7_IDENTITY_BELIEF_DECISION_20260509.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    forensics = load(FORENSICS)
    coverage = load(COVERAGE)
    masks = load(MASKS)
    train = load(TRAIN)
    eval_dec = load(EVAL_DECISION)
    complete_train = int(coverage.get("complete_train_sample_count", train.get("complete_train_sample_count", 0)) or 0)
    coverage_expanded = bool(coverage.get("coverage_expanded_vs_v33_6", False))
    hard_bal = bool(masks.get("identity_hard_train_balanced", False))
    identity_ok = bool(eval_dec.get("pass_gate", False))
    semantic_top5 = bool(eval_dec.get("semantic_top5_copy_beaten", False))
    trajectory_degraded = bool(eval_dec.get("trajectory_degraded", True))
    if complete_train <= 47:
        next_step = "fix_h32_m128_target_coverage"
    elif not hard_bal:
        next_step = "fix_identity_hard_train_masks"
    elif not identity_ok:
        next_step = "fix_identity_belief_calibration"
    elif not semantic_top5:
        next_step = "fix_semantic_prototype_loss"
    else:
        next_step = "run_v33_7_h32_full_data_smoke"
    payload = {
        "generated_at_utc": utc_now(),
        "training_coverage_bottleneck_detected": bool(forensics.get("training_coverage_bottleneck_detected", complete_train < 200)),
        "complete_train_sample_count": complete_train,
        "coverage_expanded_vs_v33_6": coverage_expanded,
        "identity_hard_train_balanced": hard_bal,
        "same_instance_hard_bce_active": bool(train.get("same_instance_hard_bce_active", False)),
        "embedding_similarity_logits_active": bool(train.get("embedding_similarity_logits_active", False)),
        "fused_same_instance_logits_active": bool(train.get("fused_same_instance_logits_active", False)),
        "global_identity_labels_used": bool(train.get("global_identity_labels_used", False)),
        "sample_local_collision_prevented": bool(train.get("sample_local_collision_prevented", False)),
        "v30_checkpoint_loaded": bool(train.get("v30_checkpoint_loaded", False)),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen", False)),
        "integrated_v30_backbone_used": bool(train.get("integrated_v30_backbone_used", False)),
        "observed_instance_context_used": bool(train.get("observed_instance_context_used", False)),
        "observed_visual_teacher_context_used": bool(train.get("observed_visual_teacher_context_used", False)),
        "future_teacher_leakage_detected": bool(eval_dec.get("future_teacher_leakage_detected", False)),
        "hard_identity_ROC_AUC_fused_val": eval_dec.get("hard_identity_ROC_AUC_fused_val"),
        "hard_identity_ROC_AUC_fused_test": eval_dec.get("hard_identity_ROC_AUC_fused_test"),
        "val_calibrated_balanced_accuracy_val": eval_dec.get("val_calibrated_balanced_accuracy_val"),
        "val_calibrated_balanced_accuracy_test": eval_dec.get("val_calibrated_balanced_accuracy_test"),
        "identity_retrieval_exclude_same_point_top1_val": eval_dec.get("identity_retrieval_exclude_same_point_top1_val"),
        "identity_retrieval_exclude_same_point_top1_test": eval_dec.get("identity_retrieval_exclude_same_point_top1_test"),
        "identity_retrieval_same_frame_top1_val": eval_dec.get("identity_retrieval_same_frame_top1_val"),
        "identity_retrieval_same_frame_top1_test": eval_dec.get("identity_retrieval_same_frame_top1_test"),
        "semantic_proto_top1_val": eval_dec.get("semantic_proto_top1_val"),
        "semantic_proto_top1_test": eval_dec.get("semantic_proto_top1_test"),
        "semantic_proto_top5_val": eval_dec.get("semantic_proto_top5_val"),
        "semantic_proto_top5_test": eval_dec.get("semantic_proto_top5_test"),
        "semantic_top1_copy_beaten": bool(eval_dec.get("semantic_top1_copy_beaten", False)),
        "semantic_top5_copy_beaten": semantic_top5,
        "trajectory_degraded": trajectory_degraded,
        "identity_signal_stable": identity_ok,
        "semantic_ranking_signal_stable": bool(eval_dec.get("semantic_ranking_signal_stable", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.7 Identity Belief Decision",
        payload,
        ["training_coverage_bottleneck_detected", "complete_train_sample_count", "coverage_expanded_vs_v33_6", "identity_hard_train_balanced", "hard_identity_ROC_AUC_fused_val", "hard_identity_ROC_AUC_fused_test", "val_calibrated_balanced_accuracy_val", "val_calibrated_balanced_accuracy_test", "semantic_top5_copy_beaten", "trajectory_degraded", "identity_signal_stable", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
