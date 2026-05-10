#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_8_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_DECISION_20260510.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    coverage = read_json(ROOT / "reports/stwm_ostf_v33_8_complete_h32_m128_target_coverage_20260510.json")
    eval_decision = read_json(ROOT / "reports/stwm_ostf_v33_8_ablation_safe_eval_decision_20260510.json")
    viz = read_json(ROOT / "reports/stwm_ostf_v33_8_visualization_manifest_20260510.json")
    target_pass = bool(coverage.get("target_coverage_pass"))
    visualization_ready = bool(viz.get("visualization_ready"))
    identity_ok = bool(eval_decision.get("identity_signal_stable"))
    semantic_top5 = bool(eval_decision.get("semantic_top5_copy_beaten"))
    trajectory_degraded = bool(eval_decision.get("trajectory_degraded", False))
    if not target_pass:
        # Coverage is the declared V33.8 root cause; distinguish visual vs
        # prototype failures when the component counts are available.
        ratios = coverage.get("complete_coverage_ratio_by_split", {})
        recommended = "fix_visual_teacher_target_coverage" if any(float(v or 0.0) < 0.90 for v in ratios.values()) else "fix_semantic_prototype_targets"
    elif not identity_ok:
        recommended = "fix_identity_belief_calibration"
    elif not semantic_top5:
        recommended = "fix_semantic_prototype_loss"
    elif not visualization_ready:
        recommended = "fix_semantic_prototype_targets"
    else:
        recommended = "run_v33_8_h32_full_reachable_seed123_replication"
    payload = {
        "generated_at_utc": utc_now(),
        "target_coverage_pass": target_pass,
        "complete_train_sample_count": coverage.get("complete_train_sample_count"),
        "complete_coverage_ratio_by_split": coverage.get("complete_coverage_ratio_by_split", {}),
        "coverage_expanded_vs_v33_7": bool(coverage.get("coverage_expanded_vs_v33_7")),
        "best_candidate_by_val": eval_decision.get("best_candidate_by_val"),
        "best_candidate_test_confirmed": bool(eval_decision.get("best_candidate_test_confirmed")),
        "hard_identity_ROC_AUC_val": eval_decision.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": eval_decision.get("hard_identity_ROC_AUC_test"),
        "val_calibrated_balanced_accuracy_val": eval_decision.get("val_calibrated_balanced_accuracy_val"),
        "val_calibrated_balanced_accuracy_test": eval_decision.get("val_calibrated_balanced_accuracy_test"),
        "identity_retrieval_exclude_same_point_top1_val": eval_decision.get("identity_retrieval_exclude_same_point_top1_val"),
        "identity_retrieval_exclude_same_point_top1_test": eval_decision.get("identity_retrieval_exclude_same_point_top1_test"),
        "identity_retrieval_same_frame_top1_val": eval_decision.get("identity_retrieval_same_frame_top1_val"),
        "identity_retrieval_same_frame_top1_test": eval_decision.get("identity_retrieval_same_frame_top1_test"),
        "semantic_proto_top1_val": eval_decision.get("semantic_proto_top1_val"),
        "semantic_proto_top1_test": eval_decision.get("semantic_proto_top1_test"),
        "semantic_proto_top5_val": eval_decision.get("semantic_proto_top5_val"),
        "semantic_proto_top5_test": eval_decision.get("semantic_proto_top5_test"),
        "semantic_top1_copy_beaten": bool(eval_decision.get("semantic_top1_copy_beaten")),
        "semantic_top5_copy_beaten": semantic_top5,
        "trajectory_degraded": trajectory_degraded,
        "visualization_ready": visualization_ready,
        "identity_signal_stable": identity_ok,
        "semantic_ranking_signal_stable": bool(eval_decision.get("semantic_ranking_signal_stable")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": recommended,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.8 Decision",
        payload,
        [
            "target_coverage_pass",
            "complete_train_sample_count",
            "coverage_expanded_vs_v33_7",
            "best_candidate_by_val",
            "hard_identity_ROC_AUC_val",
            "val_calibrated_balanced_accuracy_val",
            "semantic_top5_copy_beaten",
            "trajectory_degraded",
            "visualization_ready",
            "recommended_next_step",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
