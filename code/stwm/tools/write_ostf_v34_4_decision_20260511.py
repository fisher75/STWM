#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_4_v34_3_residual_failure_audit_20260511.json"
TARGETS = ROOT / "reports/stwm_ostf_v34_4_residual_utility_target_build_20260511.json"
ORACLE_TRAIN = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_train_summary_20260511.json"
ORACLE_DECISION = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_decision_20260511.json"
GATE_TRAIN = ROOT / "reports/stwm_ostf_v34_4_supervised_residual_gate_train_summary_20260511.json"
GATE_DECISION = ROOT / "reports/stwm_ostf_v34_4_supervised_residual_gate_eval_decision_20260511.json"
VIS = ROOT / "reports/stwm_ostf_v34_4_residual_utility_visualization_manifest_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_4_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_4_DECISION_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def choose(payload: dict[str, Any]) -> str:
    if not payload.get("residual_utility_target_ready", False):
        return "fix_residual_utility_targets"
    if not payload.get("oracle_residual_probe_passed", False):
        return "fix_unit_residual_content"
    if payload.get("oracle_residual_probe_passed") and not payload.get("supervised_residual_gate_training_ran", False):
        return "fix_supervised_residual_gate"
    if payload.get("supervised_residual_gate_training_ran") and not payload.get("semantic_gate_order_ok", False):
        return "fix_supervised_residual_gate"
    if payload.get("residual_improves_over_pointwise_on_hard") and payload.get("residual_does_not_degrade_stable") and not payload.get("trajectory_degraded") and payload.get("visualization_ready"):
        return "run_v34_4_seed123_replication"
    return "fix_unit_residual_content"


def main() -> int:
    audit = load(AUDIT)
    targets = load(TARGETS)
    oracle_train = load(ORACLE_TRAIN)
    oracle = load(ORACLE_DECISION)
    gate_train = load(GATE_TRAIN)
    gate = load(GATE_DECISION)
    vis = load(VIS)
    source = gate if gate.get("supervised_residual_gate_training_ran") else oracle
    payload = {
        "generated_at_utc": utc_now(),
        "v34_3_residual_failure_audit_done": bool(audit),
        "residual_utility_targets_built": bool(targets),
        "residual_utility_target_ready": bool(targets.get("residual_utility_target_ready", False)),
        "oracle_residual_probe_ran": bool(oracle.get("oracle_residual_probe_ran", False)),
        "oracle_residual_probe_passed": bool(oracle.get("oracle_residual_probe_passed", False)),
        "supervised_residual_gate_training_ran": bool(gate_train.get("supervised_residual_gate_training_ran", False)),
        "v30_backbone_frozen": bool(source.get("v30_backbone_frozen", oracle_train.get("v30_backbone_frozen", True))),
        "future_leakage_detected": bool(source.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(source.get("trajectory_degraded", False)),
        "hard_identity_ROC_AUC_val": source.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": source.get("hard_identity_ROC_AUC_test"),
        "semantic_hard_signal": source.get("semantic_hard_signal", {"val": False, "test": False}),
        "changed_semantic_signal": source.get("changed_semantic_signal", {"val": False, "test": False}),
        "stable_preservation": source.get("stable_preservation", {"val": False, "test": False}),
        "pointwise_baseline_dominates": bool(source.get("pointwise_baseline_dominates", True)),
        "residual_improves_over_pointwise_on_hard": bool(source.get("residual_improves_over_pointwise_on_hard", False)),
        "residual_does_not_degrade_stable": bool(source.get("residual_does_not_degrade_stable", False)),
        "semantic_gate_order_ok": bool(source.get("semantic_gate_order_ok", False)),
        "semantic_residual_gate_mean_stable": source.get("semantic_residual_gate_mean_stable"),
        "semantic_residual_gate_mean_changed": source.get("semantic_residual_gate_mean_changed"),
        "semantic_residual_gate_mean_hard": source.get("semantic_residual_gate_mean_hard"),
        "residual_utility_subset_gain": source.get("residual_utility_subset_gain"),
        "effective_units": source.get("effective_units"),
        "unit_dominant_instance_purity": source.get("unit_dominant_instance_purity"),
        "unit_semantic_purity": source.get("unit_semantic_purity"),
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "gate_collapsed": bool(audit.get("gate_collapsed", False)),
        "gate_order_wrong": bool(audit.get("gate_order_wrong", False)),
        "force_gate_one_hurts": bool(audit.get("force_gate_one_hurts", False)),
        "residual_content_not_proven": bool(audit.get("residual_content_not_proven", True)),
        "direct_residual_supervision_missing": bool(audit.get("direct_residual_supervision_missing", True)),
    }
    payload["recommended_next_step"] = choose(payload)
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.4 Decision",
        payload,
        [
            "v34_3_residual_failure_audit_done",
            "residual_utility_targets_built",
            "residual_utility_target_ready",
            "oracle_residual_probe_ran",
            "oracle_residual_probe_passed",
            "supervised_residual_gate_training_ran",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "hard_identity_ROC_AUC_val",
            "hard_identity_ROC_AUC_test",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "pointwise_baseline_dominates",
            "residual_improves_over_pointwise_on_hard",
            "residual_does_not_degrade_stable",
            "semantic_gate_order_ok",
            "residual_utility_subset_gain",
            "effective_units",
            "unit_dominant_instance_purity",
            "unit_semantic_purity",
            "visualization_ready",
            "integrated_identity_field_claim_allowed",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
