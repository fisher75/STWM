#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_11_v33_10_forensics_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_V33_10_FORENSICS_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def mean_metric(eval_summary: dict[str, Any], key: str, split: str) -> Any:
    return eval_summary.get("candidates", [{}])[0].get("metrics", {}).get(key, {}).get(split, {}).get("mean")


def per_seed_values(eval_summary: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    row = eval_summary.get("candidates", [{}])[0]
    out: dict[str, Any] = {}
    for seed, by_split in row.get("per_seed", {}).items():
        out[seed] = {}
        for split, metrics in by_split.items():
            out[seed][split] = {k: metrics.get(k) for k in keys}
    return out


def main() -> int:
    gate = load("reports/stwm_ostf_v33_10_semantic_gate_failure_audit_20260510.json")
    baseline = load("reports/stwm_ostf_v33_10_semantic_nontrivial_baselines_20260510.json")
    train = load("reports/stwm_ostf_v33_10_copy_residual_semantic_train_summary_20260510.json")
    eval_summary = load("reports/stwm_ostf_v33_10_copy_residual_semantic_eval_summary_20260510.json")
    eval_decision = load("reports/stwm_ostf_v33_10_copy_residual_semantic_eval_decision_20260510.json")
    final_decision = load("reports/stwm_ostf_v33_10_copy_residual_semantic_decision_20260510.json")
    v33_9 = load("reports/stwm_ostf_v33_9_decision_20260510.json")
    target_code = (ROOT / "code/stwm/tools/build_ostf_v33_10_copy_residual_semantic_targets_20260510.py").read_text(encoding="utf-8")
    eval_code = (ROOT / "code/stwm/tools/eval_ostf_v33_10_copy_residual_semantic_20260510.py").read_text(encoding="utf-8")
    model_code = (ROOT / "code/stwm/modules/ostf_v33_10_copy_residual_semantic_world_model.py").read_text(encoding="utf-8")
    render_code = (ROOT / "code/stwm/tools/render_ostf_v33_10_copy_residual_semantic_diagnostics_20260510.py").read_text(encoding="utf-8")
    semantic_hard_seed_locked = "H32_M128_seed42.json" in target_code and "semantic_hard" in target_code
    strongest = baseline.get("which_baseline_is_strongest_by_subset", {})
    nontrivial_baseline_mismatch = (
        strongest.get("changed") == "sample_level_prototype_frequency"
        and "observed_frequency_prior_distribution" in eval_code
        and "sample_level" not in eval_code
    )
    identity_semantic_trunk_coupling = "hidden = self.identity_trunk" in model_code and "semantic_change_head(hidden)" in model_code
    visualization_random = "i % obs.shape[0]" in render_code or "first batch" in render_code.lower()
    id9 = v33_9.get("hard_identity_ROC_AUC_test", {}).get("mean")
    id10 = eval_decision.get("hard_identity_ROC_AUC_test", {}).get("mean")
    cal9 = v33_9.get("val_calibrated_balanced_accuracy_test", {}).get("mean")
    cal10 = eval_decision.get("val_calibrated_balanced_accuracy_test", {}).get("mean")
    residual_signal = bool(eval_decision.get("changed_top5_beats_nontrivial_baseline") and eval_decision.get("semantic_hard_top5_beats_nontrivial_baseline"))
    gate_auc = eval_decision.get("semantic_change_AUROC", {})
    gate_weak = bool((gate_auc.get("val", {}).get("mean") or 0.0) < 0.60 and (gate_auc.get("test", {}).get("mean") or 0.0) < 0.60)
    payload = {
        "generated_at_utc": utc_now(),
        "stable_preservation_metrics": per_seed_values(eval_summary, ["stable_copy_top1", "stable_copy_top5", "stable_model_top1", "stable_model_top5", "stable_update_gate_mean", "stable_wrong_update_rate"]),
        "changed_hard_residual_metrics": per_seed_values(eval_summary, ["changed_model_top1", "changed_model_top5", "changed_nontrivial_baseline_top1", "changed_nontrivial_baseline_top5", "semantic_hard_model_top1", "semantic_hard_model_top5", "semantic_hard_nontrivial_baseline_top1", "semantic_hard_nontrivial_baseline_top5"]),
        "semantic_gate_metrics": per_seed_values(eval_summary, ["semantic_change_AUROC", "semantic_change_balanced_accuracy", "changed_update_gate_recall", "gate_positive_ratio", "gate_collapse_detected"]),
        "identity_regression": {
            "v33_9_hard_identity_ROC_AUC_test": id9,
            "v33_10_hard_identity_ROC_AUC_test": id10,
            "hard_identity_ROC_AUC_delta_test": None if id9 is None or id10 is None else float(id10 - id9),
            "v33_9_val_calibrated_balanced_accuracy_test": cal9,
            "v33_10_val_calibrated_balanced_accuracy_test": cal10,
            "val_calibrated_balanced_accuracy_delta_test": None if cal9 is None or cal10 is None else float(cal10 - cal9),
        },
        "semantic_hard_seed_locked": semantic_hard_seed_locked,
        "nontrivial_baseline_mismatch": nontrivial_baseline_mismatch,
        "identity_semantic_trunk_coupling_detected": identity_semantic_trunk_coupling,
        "stable_preservation_failed": bool(final_decision.get("stable_preservation_not_degraded_top5") is False or gate.get("stable_preservation_failed")),
        "identity_regressed": bool(eval_decision.get("identity_regressed")),
        "semantic_residual_signal_positive": residual_signal,
        "semantic_gate_weak": gate_weak,
        "visualization_case_selection_random": visualization_random,
        "recommended_fix": "fix_semantic_hard_protocol_and_baseline_bank_then_freeze_identity_path",
        "source_train_summary": train.get("checkpoint_path"),
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.11 V33.10 Forensics", payload, ["semantic_hard_seed_locked", "nontrivial_baseline_mismatch", "identity_semantic_trunk_coupling_detected", "stable_preservation_failed", "identity_regressed", "semantic_residual_signal_positive", "semantic_gate_weak", "visualization_case_selection_random", "recommended_fix"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
