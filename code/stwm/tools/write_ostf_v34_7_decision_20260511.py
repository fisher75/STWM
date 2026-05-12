#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_7_v34_6_assignment_and_artifact_audit_20260511.json"
REMAT = ROOT / "reports/stwm_ostf_v34_7_artifact_rematerialization_20260511.json"
TARGET = ROOT / "reports/stwm_ostf_v34_7_assignment_aware_residual_target_build_20260511.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_7_assignment_oracle_residual_probe_decision_20260511.json"
GATE = ROOT / "reports/stwm_ostf_v34_7_assignment_residual_gate_decision_20260511.json"
VIS = ROOT / "reports/stwm_ostf_v34_7_assignment_residual_visualization_manifest_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_7_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_DECISION_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    audit, remat, target, oracle, gate, vis = (load(p) for p in [AUDIT, REMAT, TARGET, ORACLE, GATE, VIS])
    artifact_fixed = bool(remat.get("artifact_packaging_fixed")) and not audit.get("artifact_packaging_fixed_required", False)
    target_ready = bool(target.get("assignment_aware_target_ready"))
    oracle_pass = bool(oracle.get("assignment_oracle_residual_probe_passed"))
    gate_ran = bool(gate.get("learned_gate_training_ran"))
    gate_pass = gate.get("learned_gate_passed", "not_run")
    if not artifact_fixed:
        rec = "fix_artifact_packaging"
    elif not target_ready:
        rec = "fix_assignment_aware_targets"
    elif not oracle_pass or oracle.get("assignment_load_bearing_on_residual") is not True:
        rec = "fix_assignment_bound_residual_model"
    elif gate_ran and gate_pass is not True:
        rec = "fix_residual_gate"
    elif oracle_pass and gate_pass is True and (oracle.get("semantic_hard_signal", {}).get("val") or oracle.get("changed_semantic_signal", {}).get("val")):
        rec = "run_v34_7_seed123_replication"
    else:
        rec = "fix_residual_gate"
    payload = {
        "generated_at_utc": utc_now(),
        "artifact_packaging_fixed": artifact_fixed,
        "assignment_aware_targets_built": bool(target.get("assignment_aware_targets_built")),
        "assignment_aware_target_ready": target_ready,
        "assignment_bound_model_built": True,
        "assignment_oracle_residual_probe_ran": bool(oracle.get("assignment_oracle_residual_probe_ran")),
        "assignment_oracle_residual_probe_passed": oracle_pass,
        "learned_gate_training_ran": gate_ran,
        "learned_gate_passed": gate_pass,
        "v30_backbone_frozen": bool(oracle.get("v30_backbone_frozen")),
        "future_leakage_detected": bool(oracle.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(oracle.get("trajectory_degraded", False)),
        "semantic_hard_signal": oracle.get("semantic_hard_signal"),
        "changed_semantic_signal": oracle.get("changed_semantic_signal"),
        "stable_preservation": oracle.get("stable_preservation"),
        "pointwise_baseline_dominates": bool(oracle.get("pointwise_baseline_dominates", True)),
        "assignment_aware_subset_gain": oracle.get("assignment_aware_subset_gain"),
        "strict_residual_subset_gain": oracle.get("strict_residual_subset_gain"),
        "unit_memory_load_bearing_on_residual": oracle.get("unit_memory_load_bearing_on_residual"),
        "semantic_measurements_load_bearing_on_residual": oracle.get("semantic_measurements_load_bearing_on_residual"),
        "assignment_load_bearing_on_residual": oracle.get("assignment_load_bearing_on_residual"),
        "semantic_gate_order_ok": gate.get("semantic_gate_order_ok", "not_run") if gate_ran else "not_run",
        "effective_units": oracle.get("effective_units"),
        "unit_dominant_instance_purity": oracle.get("unit_dominant_instance_purity"),
        "unit_semantic_purity": oracle.get("unit_semantic_purity"),
        "visualization_ready": bool(vis.get("visualization_ready")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": rec,
    }
    dump_json(OUT, payload)
    write_doc(DOC, "STWM OSTF V34.7 Decision", payload, ["artifact_packaging_fixed", "assignment_aware_targets_built", "assignment_aware_target_ready", "assignment_bound_model_built", "assignment_oracle_residual_probe_ran", "assignment_oracle_residual_probe_passed", "learned_gate_training_ran", "learned_gate_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "assignment_aware_subset_gain", "strict_residual_subset_gain", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "semantic_gate_order_ok", "visualization_ready", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
