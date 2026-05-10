#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRUTH = ROOT / "reports/stwm_ostf_v33_9_v33_8_training_truth_audit_20260510.json"
TRAIN = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_train_summary_20260510.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_eval_decision_20260510.json"
VIZ = ROOT / "reports/stwm_ostf_v33_9_world_model_diagnostic_visualization_manifest_20260510.json"
REPORT = ROOT / "reports/stwm_ostf_v33_9_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_9_DECISION_20260510.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    truth = load(TRUTH)
    train = load(TRAIN)
    evald = load(EVAL_DECISION)
    viz = load(VIZ)
    fresh_done = bool(train.get("fresh_training") and train.get("completed_candidate_count") == train.get("candidate_count") and train.get("skipped_existing_candidate_count", 99) == 0)
    identity_ok = bool(evald.get("identity_signal_stable"))
    semantic_weak = bool(evald.get("semantic_weak_gate_passed"))
    semantic_strong = bool(evald.get("semantic_strong_gate_passed"))
    trajectory_degraded = bool(evald.get("trajectory_degraded", True))
    visualization_ready = bool(viz.get("visualization_ready"))
    if not fresh_done:
        next_step = "force_fresh_training_fix"
    elif not identity_ok:
        next_step = "fix_identity_training_loss"
    elif not semantic_weak:
        next_step = "fix_semantic_prototype_loss_or_target_space"
    elif not trajectory_degraded and visualization_ready:
        next_step = "run_v33_9_seed123_replication"
    else:
        next_step = "fix_semantic_prototype_loss_or_target_space"
    payload = {
        "generated_at_utc": utc_now(),
        "v33_8_training_not_fresh": bool(truth.get("v33_8_training_not_fresh", False)),
        "fresh_training_completed": fresh_done,
        "skipped_existing_candidate_count": int(train.get("skipped_existing_candidate_count", 0)) if train else None,
        "complete_train_sample_count": train.get("complete_train_sample_count"),
        "actual_train_sample_count": train.get("actual_train_sample_count_by_candidate"),
        "best_candidate_by_val": evald.get("best_candidate_by_val"),
        "best_candidate_test_confirmed": bool(evald.get("best_candidate_test_confirmed", False)),
        "hard_identity_ROC_AUC_val": evald.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": evald.get("hard_identity_ROC_AUC_test"),
        "val_calibrated_balanced_accuracy_val": evald.get("val_calibrated_balanced_accuracy_val"),
        "val_calibrated_balanced_accuracy_test": evald.get("val_calibrated_balanced_accuracy_test"),
        "identity_retrieval_exclude_same_point_top1_val": evald.get("identity_retrieval_exclude_same_point_top1_val"),
        "identity_retrieval_exclude_same_point_top1_test": evald.get("identity_retrieval_exclude_same_point_top1_test"),
        "identity_retrieval_same_frame_top1_val": evald.get("identity_retrieval_same_frame_top1_val"),
        "identity_retrieval_same_frame_top1_test": evald.get("identity_retrieval_same_frame_top1_test"),
        "global_semantic_top1_copy_beaten": bool(evald.get("global_semantic_top1_copy_beaten", False)),
        "global_semantic_top5_copy_beaten": bool(evald.get("global_semantic_top5_copy_beaten", False)),
        "stable_preservation_not_degraded": bool(evald.get("stable_preservation_not_degraded", False)),
        "changed_top1_beats_copy": bool(evald.get("changed_top1_beats_copy", False)),
        "changed_top5_beats_copy": bool(evald.get("changed_top5_beats_copy", False)),
        "semantic_hard_top1_beats_copy": bool(evald.get("semantic_hard_top1_beats_copy", False)),
        "semantic_hard_top5_beats_copy": bool(evald.get("semantic_hard_top5_beats_copy", False)),
        "semantic_strong_gate_passed": semantic_strong,
        "semantic_weak_gate_passed": semantic_weak,
        "trajectory_degraded": trajectory_degraded,
        "visualization_ready": visualization_ready,
        "identity_signal_stable": identity_ok,
        "semantic_ranking_signal_stable": semantic_weak,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.9 Decision",
        payload,
        ["v33_8_training_not_fresh", "fresh_training_completed", "best_candidate_by_val", "hard_identity_ROC_AUC_val", "val_calibrated_balanced_accuracy_val", "semantic_strong_gate_passed", "semantic_weak_gate_passed", "trajectory_degraded", "visualization_ready", "identity_signal_stable", "semantic_ranking_signal_stable", "recommended_next_step"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
