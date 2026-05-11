#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_5_v34_4_residual_objective_audit_20260511.json"
TARGETS = ROOT / "reports/stwm_ostf_v34_5_strict_residual_utility_target_build_20260511.json"
DELTA_TRAIN = ROOT / "reports/stwm_ostf_v34_5_delta_residual_probe_train_summary_20260511.json"
DELTA = ROOT / "reports/stwm_ostf_v34_5_delta_residual_probe_decision_20260511.json"
ABLATION = ROOT / "reports/stwm_ostf_v34_5_residual_content_ablation_20260511.json"
GATE_TRAIN = ROOT / "reports/stwm_ostf_v34_5_delta_residual_gate_train_summary_20260511.json"
GATE_DECISION = ROOT / "reports/stwm_ostf_v34_5_delta_residual_gate_eval_decision_20260511.json"
VIS = ROOT / "reports/stwm_ostf_v34_5_delta_residual_visualization_manifest_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_5_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_5_DECISION_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def choose(payload: dict[str, Any]) -> str:
    if not payload.get("strict_residual_utility_targets_built") or not payload.get("strict_residual_utility_target_ready"):
        return "fix_strict_residual_targets"
    if payload.get("delta_residual_probe_ran") and not payload.get("delta_objective_beats_standalone_objective", False):
        return "fix_delta_residual_objective"
    if payload.get("delta_residual_probe_ran") and not payload.get("delta_residual_probe_passed", False):
        return "fix_unit_memory_residual_content"
    if payload.get("delta_residual_probe_passed") and not payload.get("learned_gate_training_ran", False):
        return "fix_delta_residual_gate"
    if payload.get("learned_gate_training_ran") and not payload.get("semantic_gate_order_ok", False):
        return "fix_delta_residual_gate"
    if (
        payload.get("residual_improves_over_pointwise_on_hard")
        and payload.get("residual_does_not_degrade_stable")
        and not payload.get("trajectory_degraded")
        and payload.get("visualization_ready")
    ):
        return "run_v34_5_seed123_replication"
    return "fix_unit_memory_residual_content"


def main() -> int:
    audit = load(AUDIT)
    targets = load(TARGETS)
    delta_train = load(DELTA_TRAIN)
    delta = load(DELTA)
    ablation = load(ABLATION)
    gate_train = load(GATE_TRAIN)
    gate = load(GATE_DECISION)
    vis = load(VIS)
    source = gate if gate.get("learned_gate_training_ran") else delta
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "v34_4_residual_objective_audit_done": bool(audit),
        "strict_residual_utility_targets_built": bool(targets),
        "strict_residual_utility_target_ready": bool(targets.get("strict_residual_utility_target_ready", False)),
        "delta_residual_probe_ran": bool(delta.get("delta_residual_probe_ran", False)),
        "delta_residual_probe_passed": bool(delta.get("delta_residual_probe_passed", False)),
        "delta_objective_beats_standalone_objective": bool(ablation.get("whether_delta_objective_beats_standalone_objective", delta.get("delta_objective_beats_standalone_objective", False))),
        "learned_gate_training_ran": bool(gate_train.get("learned_gate_training_ran", False)),
        "v30_backbone_frozen": bool(source.get("v30_backbone_frozen", delta_train.get("v30_backbone_frozen", True))),
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
        "semantic_gate_order_ok": source.get("semantic_gate_order_ok", "not_run"),
        "strict_residual_subset_gain": source.get("strict_residual_subset_gain"),
        "delta_vs_standalone_gain": ablation.get("delta_vs_standalone_gain", source.get("delta_vs_standalone_gain")),
        "effective_units": source.get("effective_units"),
        "unit_dominant_instance_purity": source.get("unit_dominant_instance_purity"),
        "unit_semantic_purity": source.get("unit_semantic_purity"),
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "residual_supervised_as_standalone_target": bool(audit.get("residual_supervised_as_standalone_target", False)),
        "delta_residual_objective_missing": bool(audit.get("delta_residual_objective_missing", False)),
        "force_gate_one_hurts_due_residual_content": bool(audit.get("force_gate_one_hurts_due_residual_content", False)),
        "oracle_fail_is_borderline": bool(audit.get("oracle_fail_is_borderline", False)),
        "strict_residual_semantic_positive_ratio_by_split": targets.get("strict_residual_semantic_positive_ratio_by_split"),
    }
    payload["recommended_next_step"] = choose(payload)
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.5 Decision",
        payload,
        [
            "v34_4_residual_objective_audit_done",
            "strict_residual_utility_targets_built",
            "strict_residual_utility_target_ready",
            "delta_residual_probe_ran",
            "delta_residual_probe_passed",
            "delta_objective_beats_standalone_objective",
            "learned_gate_training_ran",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "pointwise_baseline_dominates",
            "residual_improves_over_pointwise_on_hard",
            "residual_does_not_degrade_stable",
            "semantic_gate_order_ok",
            "strict_residual_subset_gain",
            "delta_vs_standalone_gain",
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
