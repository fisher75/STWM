#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v33_10_semantic_gate_failure_audit_20260510.json"
BASE = ROOT / "reports/stwm_ostf_v33_10_semantic_nontrivial_baselines_20260510.json"
TARGETS = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_target_build_20260510.json"
TRAIN = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_train_summary_20260510.json"
EVAL = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_eval_decision_20260510.json"
VIZ = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_visualization_manifest_20260510.json"
REPORT = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_10_COPY_RESIDUAL_SEMANTIC_DECISION_20260510.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    audit = load(AUDIT)
    base = load(BASE)
    targets = load(TARGETS)
    train = load(TRAIN)
    evald = load(EVAL)
    viz = load(VIZ)
    stable_ok = bool(evald.get("stable_preservation_not_degraded_top5", False))
    changed_ok = bool(evald.get("changed_top5_beats_nontrivial_baseline", False))
    hard_ok = bool(evald.get("semantic_hard_top5_beats_nontrivial_baseline", False))
    identity_regressed = bool(evald.get("identity_regressed", False))
    trajectory_degraded = bool(evald.get("trajectory_degraded", True))
    gate_overclaim_fixed = bool(audit.get("semantic_strong_gate_overclaims", True) and not evald.get("semantic_strong_gate_overclaims", False))
    if bool(audit.get("semantic_strong_gate_overclaims", False)) and not gate_overclaim_fixed:
        next_step = "fix_semantic_gate_protocol"
    elif not stable_ok:
        next_step = "fix_semantic_copy_residual_loss"
    elif not (changed_ok and hard_ok):
        next_step = "fix_semantic_target_space"
    elif identity_regressed:
        next_step = "fix_identity_regression_after_semantic"
    elif not trajectory_degraded and bool(viz.get("visualization_ready")):
        next_step = "run_v33_10_seed123_replication"
    else:
        next_step = "fix_semantic_copy_residual_loss"
    payload = {
        "generated_at_utc": utc_now(),
        "semantic_gate_overclaim_fixed": gate_overclaim_fixed,
        "nontrivial_semantic_baselines_built": bool(base.get("nontrivial_semantic_baselines_built", False)),
        "copy_residual_targets_built": bool(targets.get("copy_residual_targets_built", False)),
        "copy_prior_active": bool(evald.get("copy_prior_active", False)),
        "semantic_change_gate_active": bool(evald.get("semantic_change_gate_active", False)),
        "fresh_training_completed": bool(train.get("fresh_training_completed", False)),
        "complete_train_sample_count": train.get("complete_train_sample_count"),
        "best_candidate": evald.get("candidate"),
        "hard_identity_ROC_AUC_val": evald.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": evald.get("hard_identity_ROC_AUC_test"),
        "val_calibrated_balanced_accuracy_val": evald.get("val_calibrated_balanced_accuracy_val"),
        "val_calibrated_balanced_accuracy_test": evald.get("val_calibrated_balanced_accuracy_test"),
        "stable_preservation_not_degraded_top5": stable_ok,
        "changed_top5_beats_nontrivial_baseline": changed_ok,
        "semantic_hard_top5_beats_nontrivial_baseline": hard_ok,
        "semantic_change_AUROC": evald.get("semantic_change_AUROC"),
        "gate_collapse_detected": bool(evald.get("gate_collapse_detected", True)),
        "identity_regressed": identity_regressed,
        "trajectory_degraded": trajectory_degraded,
        "real_visualizations_ready": bool(viz.get("visualization_ready", False) and viz.get("real_images_rendered", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.10 Copy Residual Semantic Decision", payload, ["semantic_gate_overclaim_fixed", "nontrivial_semantic_baselines_built", "copy_residual_targets_built", "copy_prior_active", "semantic_change_gate_active", "fresh_training_completed", "stable_preservation_not_degraded_top5", "changed_top5_beats_nontrivial_baseline", "semantic_hard_top5_beats_nontrivial_baseline", "semantic_change_AUROC", "gate_collapse_detected", "identity_regressed", "trajectory_degraded", "real_visualizations_ready", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
