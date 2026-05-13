#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_13_v34_12_artifact_and_selector_truth_audit_20260513.json"
ART = ROOT / "reports/stwm_ostf_v34_13_artifact_rematerialization_20260513.json"
SEL_TRAIN = ROOT / "reports/stwm_ostf_v34_13_nonoracle_measurement_selector_train_summary_20260513.json"
SEL = ROOT / "reports/stwm_ostf_v34_13_nonoracle_measurement_selector_decision_20260513.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_13_selector_conditioned_oracle_residual_probe_decision_20260513.json"
GATE = ROOT / "reports/stwm_ostf_v34_13_selector_conditioned_residual_gate_decision_20260513.json"
VIS = ROOT / "reports/stwm_ostf_v34_13_selector_conditioned_visualization_manifest_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_13_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_13_DECISION_20260513.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    audit = load(AUDIT)
    art = load(ART)
    sel_train = load(SEL_TRAIN)
    sel = load(SEL)
    oracle = load(ORACLE)
    gate = load(GATE)
    vis = load(VIS)
    artifact_fixed = bool(art.get("artifact_packaging_fixed") and not art.get("missing_after", {}).get("nonoracle_selector", False))
    if not artifact_fixed:
        rec = "fix_artifact_packaging"
    elif not sel.get("measurement_selector_nonoracle_passed", False):
        rec = "fix_nonoracle_measurement_selector"
    elif oracle.get("oracle_residual_probe_ran") and not oracle.get("semantic_measurements_load_bearing_on_residual", False):
        rec = "fix_selector_conditioned_local_evidence_encoder"
    elif oracle.get("semantic_measurements_load_bearing_on_residual") and not oracle.get("assignment_load_bearing_on_residual", False):
        rec = "fix_assignment_bound_residual_model"
    elif oracle.get("oracle_residual_probe_passed") and gate.get("learned_gate_passed") is not True:
        rec = "fix_residual_gate"
    elif oracle.get("oracle_residual_probe_passed") and gate.get("learned_gate_passed") is True:
        rec = "run_v34_13_seed123_replication"
    else:
        rec = "fix_selector_conditioned_local_evidence_encoder"
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.13 完成 artifact/source truth 审计、训练式 non-oracle selector、selector-conditioned local evidence oracle probe 与可视化；仍不声明 semantic field success。",
        "artifact_truth_audit_done": AUDIT.exists(),
        "artifact_packaging_fixed": artifact_fixed,
        "nonoracle_selector_built": bool(sel_train.get("nonoracle_selector_built", False)),
        "nonoracle_selector_passed": bool(sel.get("measurement_selector_nonoracle_passed", False)),
        "selector_conditioned_encoder_built": True,
        "oracle_residual_probe_ran": bool(oracle.get("oracle_residual_probe_ran", False)),
        "oracle_residual_probe_passed": bool(oracle.get("oracle_residual_probe_passed", False)),
        "learned_gate_training_ran": bool(gate.get("learned_gate_training_ran", False)),
        "learned_gate_passed": gate.get("learned_gate_passed", "not_run"),
        "v30_backbone_frozen": bool(oracle.get("v30_backbone_frozen", True)),
        "future_leakage_detected": bool(oracle.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(oracle.get("trajectory_degraded", False)),
        "semantic_hard_signal": oracle.get("semantic_hard_signal", {"val": False, "test": False}),
        "changed_semantic_signal": oracle.get("changed_semantic_signal", {"val": False, "test": False}),
        "stable_preservation": oracle.get("stable_preservation", {"val": None, "test": None}),
        "pointwise_baseline_dominates": bool(oracle.get("pointwise_baseline_dominates", False)),
        "causal_assignment_subset_gain": oracle.get("causal_assignment_subset_gain", {"val": None, "test": None}),
        "strict_residual_subset_gain": oracle.get("strict_residual_subset_gain", {"val": None, "test": None}),
        "unit_memory_load_bearing_on_residual": bool(oracle.get("unit_memory_load_bearing_on_residual", False)),
        "semantic_measurements_load_bearing_on_residual": bool(oracle.get("semantic_measurements_load_bearing_on_residual", False)),
        "assignment_load_bearing_on_residual": bool(oracle.get("assignment_load_bearing_on_residual", False)),
        "zero_semantic_measurements_metric_delta": oracle.get("zero_semantic_measurements_metric_delta", {"val": None, "test": None}),
        "shuffle_semantic_measurements_metric_delta": oracle.get("shuffle_semantic_measurements_metric_delta", {"val": None, "test": None}),
        "selector_ablation_delta": oracle.get("selector_ablation_delta", {"val": None, "test": None}),
        "effective_units": oracle.get("effective_units", {"val": None, "test": None}),
        "unit_dominant_instance_purity": oracle.get("unit_dominant_instance_purity", {"val": None, "test": None}),
        "unit_semantic_purity": oracle.get("unit_semantic_purity", {"val": None, "test": None}),
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": rec,
        "audit_flags": {
            "nonoracle_selector_json_missing": audit.get("nonoracle_selector_json_missing"),
            "measurement_teacher_name_inconsistent": audit.get("measurement_teacher_name_inconsistent"),
            "selector_is_fixed_heuristic_v34_12": audit.get("selector_is_fixed_heuristic"),
            "forward_gate_zero_by_default_v34_12": audit.get("forward_gate_zero_by_default"),
        },
        "selector_decision": {
            "selector_was_trained": sel.get("selector_was_trained"),
            "selector_beats_random": sel.get("selector_beats_random"),
            "selector_beats_pointwise_on_hard": sel.get("selector_beats_pointwise_on_hard"),
            "selector_beats_pointwise_on_changed": sel.get("selector_beats_pointwise_on_changed"),
            "oracle_gap_to_selector_hard": sel.get("oracle_gap_to_selector_hard"),
            "oracle_gap_to_selector_changed": sel.get("oracle_gap_to_selector_changed"),
        },
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.13 final decision 中文报告",
        payload,
        ["中文结论", "artifact_truth_audit_done", "artifact_packaging_fixed", "nonoracle_selector_built", "nonoracle_selector_passed", "selector_conditioned_encoder_built", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "learned_gate_training_ran", "learned_gate_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "causal_assignment_subset_gain", "strict_residual_subset_gain", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "visualization_ready", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"],
    )
    print(f"已写出 V34.13 final decision: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
