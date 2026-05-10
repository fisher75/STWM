#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_11_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_DECISION_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    forensic = load("reports/stwm_ostf_v33_11_v33_10_forensics_20260510.json")
    hard = load("reports/stwm_ostf_v33_11_true_semantic_hard_protocol_20260510.json")
    bank = load("reports/stwm_ostf_v33_11_semantic_baseline_bank_20260510.json")
    oracle = load("reports/stwm_ostf_v33_11_oracle_gate_upper_bound_20260510.json")
    train = load("reports/stwm_ostf_v33_11_identity_preserving_copy_residual_train_summary_20260510.json")
    eval_dec = load("reports/stwm_ostf_v33_11_identity_preserving_copy_residual_eval_decision_20260510.json")
    viz = load("reports/stwm_ostf_v33_11_semantic_identity_visualization_manifest_20260510.json")
    semantic_hard_fixed = bool(hard.get("semantic_hard_seed_independent") and hard.get("exact_blockers") == [])
    baseline_fixed = bool(bank.get("baseline_bank_ready") and bank.get("strongest_baseline_by_subset_selected_on_val"))
    identity_regressed = bool(eval_dec.get("identity_regressed_vs_v33_9"))
    stable_ok = bool(eval_dec.get("stable_preservation_not_degraded_top5"))
    changed_ok = bool(eval_dec.get("changed_top5_beats_strongest_baseline"))
    hard_ok = bool(eval_dec.get("semantic_hard_top5_beats_strongest_baseline"))
    trajectory_degraded = bool(eval_dec.get("trajectory_degraded"))
    viz_ready = bool(viz.get("visualization_ready") and viz.get("case_mining_used"))
    if not semantic_hard_fixed:
        next_step = "fix_semantic_hard_protocol"
    elif not baseline_fixed:
        next_step = "fix_nontrivial_baseline_protocol"
    elif identity_regressed:
        next_step = "fix_identity_preservation"
    elif not stable_ok:
        next_step = "fix_semantic_copy_residual_loss"
    elif not (changed_ok and hard_ok):
        next_step = "fix_semantic_target_space"
    elif trajectory_degraded or not viz_ready:
        next_step = "fix_semantic_copy_residual_loss"
    else:
        next_step = "run_v33_11_seed123_replication"
    payload = {
        "generated_at_utc": utc_now(),
        "v33_10_forensics_done": bool(forensic),
        "semantic_hard_seed_locked_fixed": semantic_hard_fixed,
        "nontrivial_baseline_mismatch_fixed": baseline_fixed,
        "baseline_selected_on_val": bool(bank.get("val_selection_only")),
        "oracle_gate_upper_bound_done": bool(oracle.get("oracle_gate_upper_bound_done")),
        "identity_preserving_model_built": (ROOT / "code/stwm/modules/ostf_v33_11_identity_preserving_copy_residual_semantic_world_model.py").exists(),
        "fresh_training_completed": bool(train.get("fresh_training_completed")),
        "identity_path_frozen_or_distilled": bool(train.get("identity_path_frozen_or_distilled")),
        "complete_train_sample_count": train.get("complete_train_sample_count"),
        "hard_identity_ROC_AUC_val": eval_dec.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": eval_dec.get("hard_identity_ROC_AUC_test"),
        "val_calibrated_balanced_accuracy_val": eval_dec.get("val_calibrated_balanced_accuracy_val"),
        "val_calibrated_balanced_accuracy_test": eval_dec.get("val_calibrated_balanced_accuracy_test"),
        "identity_regressed_vs_v33_9": identity_regressed,
        "stable_preservation_not_degraded_top5": stable_ok,
        "stable_wrong_update_rate": eval_dec.get("stable_wrong_update_rate"),
        "changed_top5_beats_strongest_baseline": changed_ok,
        "semantic_hard_top5_beats_strongest_baseline": hard_ok,
        "semantic_change_AUROC": eval_dec.get("semantic_change_AUROC"),
        "changed_update_gate_recall": eval_dec.get("changed_update_gate_recall"),
        "gate_collapse_detected": bool(eval_dec.get("gate_collapse_detected")),
        "trajectory_degraded": trajectory_degraded,
        "case_mined_visualizations_ready": viz_ready,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.11 Decision", payload, ["semantic_hard_seed_locked_fixed", "nontrivial_baseline_mismatch_fixed", "baseline_selected_on_val", "oracle_gate_upper_bound_done", "fresh_training_completed", "identity_path_frozen_or_distilled", "complete_train_sample_count", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "identity_regressed_vs_v33_9", "stable_preservation_not_degraded_top5", "changed_top5_beats_strongest_baseline", "semantic_hard_top5_beats_strongest_baseline", "semantic_change_AUROC", "changed_update_gate_recall", "gate_collapse_detected", "trajectory_degraded", "case_mined_visualizations_ready", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
