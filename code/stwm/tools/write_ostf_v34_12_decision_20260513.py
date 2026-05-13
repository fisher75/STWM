#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_12_v34_11_measurement_truth_audit_20260513.json"
ART = ROOT / "reports/stwm_ostf_v34_12_artifact_rematerialization_20260513.json"
SELECTOR = ROOT / "reports/stwm_ostf_v34_12_nonoracle_measurement_selector_probe_20260513.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_12_local_evidence_oracle_residual_probe_decision_20260513.json"
GATE = ROOT / "reports/stwm_ostf_v34_12_local_evidence_residual_gate_decision_20260513.json"
VIS = ROOT / "reports/stwm_ostf_v34_12_local_evidence_visualization_manifest_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_12_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_12_DECISION_20260513.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    audit = load(AUDIT)
    art = load(ART)
    selector = load(SELECTOR)
    oracle = load(ORACLE)
    gate = load(GATE)
    vis = load(VIS)
    artifact_fixed = bool(art.get("artifact_packaging_fixed") and not audit.get("quality_probe_json_missing") and not audit.get("visualization_json_missing"))
    if not artifact_fixed:
        rec = "fix_artifact_packaging"
    elif not selector.get("measurement_selector_nonoracle_passed", False):
        rec = "fix_nonoracle_measurement_selector"
    elif oracle.get("oracle_residual_probe_ran") and not oracle.get("semantic_measurements_load_bearing_on_residual", False):
        rec = "fix_local_semantic_evidence_encoder"
    elif oracle.get("semantic_measurements_load_bearing_on_residual") and not oracle.get("assignment_load_bearing_on_residual", False):
        rec = "fix_assignment_bound_residual_model"
    elif oracle.get("oracle_residual_probe_passed") and gate.get("learned_gate_passed") is not True:
        rec = "fix_residual_gate"
    else:
        rec = "fix_local_semantic_evidence_encoder"
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 完成 artifact/source truth、non-oracle selector 与 local evidence oracle probe；不声明 semantic field success，不训练 learned gate 除非 oracle 通过。",
        "measurement_truth_audit_done": bool(AUDIT.exists()),
        "artifact_packaging_fixed": artifact_fixed,
        "nonoracle_measurement_selector_probe_done": bool(selector.get("nonoracle_measurement_selector_probe_done", False)),
        "measurement_quality_overestimated_by_oracle": bool(selector.get("measurement_quality_overestimated_by_oracle", False) or audit.get("quality_probe_uses_oracle_best_measurement", False)),
        "local_evidence_encoder_built": True,
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
        "attention_uses_nontrivial_timesteps": bool(oracle.get("attention_uses_nontrivial_timesteps", False)),
        "zero_semantic_measurements_metric_delta": oracle.get("zero_semantic_measurements_metric_delta", {"val": None, "test": None}),
        "shuffle_semantic_measurements_metric_delta": oracle.get("shuffle_semantic_measurements_metric_delta", {"val": None, "test": None}),
        "effective_units": oracle.get("effective_units", {"val": None, "test": None}),
        "unit_dominant_instance_purity": oracle.get("unit_dominant_instance_purity", {"val": None, "test": None}),
        "unit_semantic_purity": oracle.get("unit_semantic_purity", {"val": None, "test": None}),
        "quality_probe_json_missing": bool(audit.get("quality_probe_json_missing", False)),
        "visualization_json_missing": bool(audit.get("visualization_json_missing", False)),
        "quality_probe_uses_oracle_best_measurement": bool(audit.get("quality_probe_uses_oracle_best_measurement", False)),
        "measurement_teacher_name_inconsistent": bool(audit.get("measurement_teacher_name_inconsistent", False)),
        "measurement_selector_nonoracle_passed": bool(selector.get("measurement_selector_nonoracle_passed", False)),
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": rec,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.12 final decision 中文报告", payload, ["中文结论", "measurement_truth_audit_done", "artifact_packaging_fixed", "nonoracle_measurement_selector_probe_done", "measurement_quality_overestimated_by_oracle", "local_evidence_encoder_built", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "learned_gate_training_ran", "learned_gate_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "causal_assignment_subset_gain", "strict_residual_subset_gain", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "attention_uses_nontrivial_timesteps", "visualization_ready", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(f"已写出 V34.12 final decision: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
