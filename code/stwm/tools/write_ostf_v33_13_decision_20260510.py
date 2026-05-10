#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_13_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_DECISION_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    forensic = load("reports/stwm_ostf_v33_13_v33_12_gate_and_target_forensics_20260510.json")
    train = load("reports/stwm_ostf_v33_13_gate_repaired_train_summary_20260510.json")
    gate = load("reports/stwm_ostf_v33_13_gate_calibration_sweep_20260510.json")
    probe = load("reports/stwm_ostf_v33_13_semantic_target_space_probe_decision_20260510.json")
    teacher = load("reports/stwm_ostf_v33_13_real_stronger_teacher_preflight_20260510.json")
    eval_dec = load("reports/stwm_ostf_v33_13_gate_repaired_eval_decision_20260510.json")
    viz = load("reports/stwm_ostf_v33_13_gate_and_target_visualization_manifest_20260510.json")

    protocol_ok = bool(train.get("gate_protocol_repaired") and train.get("double_sigmoid_bug_fixed") and train.get("train_eval_gate_consistent"))
    probe_done = bool(probe.get("target_space_probe_done"))
    probe_passed = bool(probe.get("target_space_learnability_passed"))
    stable_ok = bool(eval_dec.get("stable_preservation_not_degraded_top5"))
    changed_ok = bool(eval_dec.get("changed_top5_beats_strongest_baseline"))
    hard_ok = bool(eval_dec.get("semantic_hard_top5_beats_strongest_baseline"))
    identity_regressed = bool(eval_dec.get("identity_regressed_vs_v33_9"))
    trajectory_degraded = bool(eval_dec.get("trajectory_degraded"))
    gate_collapse = bool(eval_dec.get("gate_collapse_detected"))
    viz_ready = bool(viz.get("visualization_ready"))
    if not protocol_ok:
        next_step = "fix_gate_repaired_model_loss"
    elif probe_done and not probe_passed:
        next_step = "build_real_stronger_teacher_targets"
    elif probe_passed and not (stable_ok and changed_ok and hard_ok and not gate_collapse and not identity_regressed and not trajectory_degraded):
        next_step = "fix_gate_repaired_model_loss"
    elif stable_ok and changed_ok and hard_ok and not identity_regressed and not trajectory_degraded and viz_ready:
        next_step = "run_v33_13_seed123_replication"
    else:
        next_step = "fix_gate_repaired_model_loss"
    payload = {
        "generated_at_utc": utc_now(),
        "gate_protocol_repaired": bool(train.get("gate_protocol_repaired")),
        "double_sigmoid_bug_fixed": bool(train.get("double_sigmoid_bug_fixed")),
        "train_eval_gate_consistent": bool(train.get("train_eval_gate_consistent")),
        "target_space_probe_done": probe_done,
        "target_space_learnability_passed": probe_passed,
        "stronger_teacher_preflight_done": bool(teacher.get("stronger_teacher_preflight_done")),
        "only_clip_b32_available": bool(teacher.get("only_clip_b32_available", forensic.get("only_clip_b32_available"))),
        "hard_identity_ROC_AUC_val": eval_dec.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": eval_dec.get("hard_identity_ROC_AUC_test"),
        "val_calibrated_balanced_accuracy_val": eval_dec.get("val_calibrated_balanced_accuracy_val"),
        "val_calibrated_balanced_accuracy_test": eval_dec.get("val_calibrated_balanced_accuracy_test"),
        "stable_preservation_not_degraded_top5": stable_ok,
        "stable_wrong_update_rate": eval_dec.get("stable_wrong_update_rate"),
        "changed_top5_beats_strongest_baseline": changed_ok,
        "semantic_hard_top5_beats_strongest_baseline": hard_ok,
        "semantic_change_AUROC": eval_dec.get("semantic_change_AUROC"),
        "gate_collapse_detected": gate_collapse,
        "identity_regressed_vs_v33_9": identity_regressed,
        "trajectory_degraded": trajectory_degraded,
        "visualization_ready": viz_ready,
        "gate_threshold_bottleneck": bool(gate.get("gate_threshold_bottleneck")),
        "residual_classifier_bottleneck": bool(gate.get("residual_classifier_bottleneck")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.13 Decision",
        payload,
        [
            "gate_protocol_repaired",
            "double_sigmoid_bug_fixed",
            "train_eval_gate_consistent",
            "target_space_probe_done",
            "target_space_learnability_passed",
            "stronger_teacher_preflight_done",
            "only_clip_b32_available",
            "hard_identity_ROC_AUC_val",
            "hard_identity_ROC_AUC_test",
            "stable_preservation_not_degraded_top5",
            "changed_top5_beats_strongest_baseline",
            "semantic_hard_top5_beats_strongest_baseline",
            "semantic_change_AUROC",
            "gate_collapse_detected",
            "identity_regressed_vs_v33_9",
            "trajectory_degraded",
            "visualization_ready",
            "integrated_identity_field_claim_allowed",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
