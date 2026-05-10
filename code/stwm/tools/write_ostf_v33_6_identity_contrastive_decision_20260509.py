#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v33_6_identity_label_namespace_audit_20260509.json"
BUILD = ROOT / "reports/stwm_ostf_v33_6_global_identity_label_build_20260509.json"
CONTRACT = ROOT / "reports/stwm_ostf_v33_6_global_identity_dataset_contract_20260509.json"
TRAIN = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_train_summary_20260509.json"
EVAL = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_eval_summary_20260509.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_eval_decision_20260509.json"
ABLATION = ROOT / "reports/stwm_ostf_v33_6_identity_label_ablation_summary_20260509.json"
REPORT = ROOT / "reports/stwm_ostf_v33_6_identity_contrastive_decision_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_6_IDENTITY_CONTRASTIVE_DECISION_20260509.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric(decision: dict[str, Any], name: str, split: str) -> Any:
    return decision.get(f"{name}_{split}")


def main() -> int:
    audit = load(AUDIT)
    build = load(BUILD)
    contract = load(CONTRACT)
    train = load(TRAIN)
    eval_summary = load(EVAL)
    eval_decision = load(EVAL_DECISION)
    namespace_safe = bool(build.get("global_identity_labels_built")) and bool(contract.get("dataset_contract_ok"))
    global_used = bool(eval_decision.get("global_identity_label_used"))
    collision_fixed = namespace_safe and global_used and bool(eval_decision.get("sample_local_collision_prevented"))
    identity_ok = bool(eval_decision.get("identity_signal_stable"))
    semantic_ok = bool(eval_decision.get("semantic_ranking_signal_stable"))
    trajectory_degraded = bool(eval_decision.get("trajectory_degraded", True))
    semantic_top5 = bool(eval_decision.get("semantic_top5_copy_beaten"))
    if not namespace_safe:
        next_step = "fix_identity_label_namespace"
    elif not identity_ok:
        next_step = "fix_identity_contrastive_loss"
    elif not semantic_top5:
        next_step = "fix_semantic_prototype_loss"
    elif not trajectory_degraded and semantic_ok:
        next_step = "run_v33_6_h32_full_data_smoke"
    else:
        next_step = "fix_identity_contrastive_loss"
    payload = {
        "generated_at_utc": utc_now(),
        "identity_label_namespace_safe": namespace_safe,
        "fut_instance_id_global_unique": bool(audit.get("fut_instance_id_global_unique", False)),
        "cross_sample_label_collision_detected": bool(audit.get("cross_sample_label_collision_detected", False)),
        "global_identity_labels_built": bool(build.get("global_identity_labels_built", False)),
        "global_identity_labels_used_in_training": bool(train.get("global_identity_labels_used_in_training", False)),
        "cross_sample_label_collision_fixed": collision_fixed,
        "sample_local_collision_prevented": bool(eval_decision.get("sample_local_collision_prevented", False)),
        "v30_checkpoint_loaded": bool(train.get("v30_checkpoint_loaded", False)),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen", False)),
        "integrated_v30_backbone_used": bool(train.get("integrated_v30_backbone_used", False)),
        "observed_instance_context_used": bool(train.get("observed_instance_context_used", False)),
        "observed_visual_teacher_context_used": bool(train.get("observed_visual_teacher_context_used", False)),
        "future_teacher_leakage_detected": bool(eval_decision.get("future_teacher_leakage_detected", False)),
        "hard_identity_ROC_AUC_val": metric(eval_decision, "hard_identity_ROC_AUC", "val"),
        "hard_identity_ROC_AUC_test": metric(eval_decision, "hard_identity_ROC_AUC", "test"),
        "hard_identity_balanced_accuracy_val": metric(eval_decision, "hard_identity_balanced_accuracy", "val"),
        "hard_identity_balanced_accuracy_test": metric(eval_decision, "hard_identity_balanced_accuracy", "test"),
        "identity_retrieval_exclude_same_point_top1_val": metric(eval_decision, "identity_retrieval_exclude_same_point_top1", "val"),
        "identity_retrieval_exclude_same_point_top1_test": metric(eval_decision, "identity_retrieval_exclude_same_point_top1", "test"),
        "identity_retrieval_same_frame_top1_val": metric(eval_decision, "identity_retrieval_same_frame_top1", "val"),
        "identity_retrieval_same_frame_top1_test": metric(eval_decision, "identity_retrieval_same_frame_top1", "test"),
        "identity_retrieval_instance_pooled_top1_val": metric(eval_decision, "identity_retrieval_instance_pooled_top1", "val"),
        "identity_retrieval_instance_pooled_top1_test": metric(eval_decision, "identity_retrieval_instance_pooled_top1", "test"),
        "semantic_proto_top1_val": metric(eval_decision, "semantic_proto_top1", "val"),
        "semantic_proto_top1_test": metric(eval_decision, "semantic_proto_top1", "test"),
        "semantic_proto_top5_val": metric(eval_decision, "semantic_proto_top5", "val"),
        "semantic_proto_top5_test": metric(eval_decision, "semantic_proto_top5", "test"),
        "semantic_top1_copy_beaten": bool(eval_decision.get("semantic_top1_copy_beaten", False)),
        "semantic_top5_copy_beaten": semantic_top5,
        "trajectory_degraded": trajectory_degraded,
        "identity_signal_stable": identity_ok,
        "semantic_ranking_signal_stable": semantic_ok,
        "integrated_identity_field_claim_allowed": bool(identity_ok and semantic_ok and not trajectory_degraded and False),
        "integrated_semantic_field_claim_allowed": False,
        "eval_summary_path": str(EVAL.relative_to(ROOT)) if eval_summary else None,
        "identity_label_ablation_summary_path": str(ABLATION.relative_to(ROOT)) if ABLATION.exists() else None,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.6 Identity Contrastive Decision",
        payload,
        ["identity_label_namespace_safe", "cross_sample_label_collision_detected", "global_identity_labels_built", "global_identity_labels_used_in_training", "sample_local_collision_prevented", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "hard_identity_balanced_accuracy_val", "hard_identity_balanced_accuracy_test", "semantic_top5_copy_beaten", "trajectory_degraded", "identity_signal_stable", "semantic_ranking_signal_stable", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
