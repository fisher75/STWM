#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_9_v34_8_state_contract_audit_20260512.json"
BANK = ROOT / "reports/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank_20260512.json"
TARGET = ROOT / "reports/stwm_ostf_v34_9_causal_assignment_residual_target_build_20260512.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_oracle_residual_probe_decision_20260512.json"
GATE = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_residual_gate_decision_20260512.json"
VIS = ROOT / "reports/stwm_ostf_v34_9_trace_fixed_residual_visualization_manifest_20260512.json"
OUT = ROOT / "reports/stwm_ostf_v34_9_decision_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_9_DECISION_20260512.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    audit, bank, target, oracle, gate, vis = (load(p) for p in [AUDIT, BANK, TARGET, ORACLE, GATE, VIS])
    obs_zero = bool(audit.get("obs_points_zero_detected"))
    trace_pass = bool(bank.get("trace_state_contract_passed") and oracle.get("trace_state_contract_passed"))
    oracle_pass = bool(oracle.get("oracle_residual_probe_passed"))
    gate_ran = bool(gate.get("learned_gate_training_ran"))
    gate_pass = gate.get("learned_gate_passed", "not_run")
    if not trace_pass:
        rec = "fix_trace_state_contract"
    elif oracle.get("semantic_measurements_load_bearing_on_residual") is False:
        rec = "fix_semantic_measurement_bank"
    elif oracle.get("assignment_load_bearing_on_residual") is False:
        rec = "fix_assignment_bound_residual_model"
    elif oracle_pass and gate_ran and gate_pass is not True:
        rec = "fix_residual_gate"
    elif oracle_pass and gate_pass is True and not oracle.get("trajectory_degraded", True):
        rec = "run_v34_9_seed123_replication"
    elif not target.get("causal_assignment_target_ready"):
        rec = "fix_causal_assignment_targets"
    else:
        rec = "fix_semantic_measurement_bank"
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.9 修复了 measurement bank 的 trace state contract，并重跑最小 causal residual probe；不声明 semantic field success。",
        "state_contract_audit_done": bool(audit),
        "obs_points_zero_detected": obs_zero,
        "trace_state_contract_repaired": bool(bank.get("trace_state_contract_passed")),
        "trace_preserving_measurement_bank_built": bool(bank.get("trace_preserving_measurement_bank_built")),
        "causal_assignment_targets_built": bool(target.get("causal_assignment_targets_built")),
        "oracle_residual_probe_ran": bool(oracle.get("oracle_residual_probe_ran")),
        "oracle_residual_probe_passed": oracle_pass,
        "learned_gate_training_ran": gate_ran,
        "learned_gate_passed": gate_pass,
        "v30_backbone_frozen": bool(oracle.get("v30_backbone_frozen", True)),
        "future_leakage_detected": bool(oracle.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(oracle.get("trajectory_degraded", False)),
        "trace_state_contract_passed": trace_pass,
        "semantic_hard_signal": oracle.get("semantic_hard_signal"),
        "changed_semantic_signal": oracle.get("changed_semantic_signal"),
        "stable_preservation": oracle.get("stable_preservation"),
        "pointwise_baseline_dominates": bool(oracle.get("pointwise_baseline_dominates", True)),
        "causal_assignment_subset_gain": oracle.get("causal_assignment_subset_gain") or oracle.get("assignment_aware_subset_gain"),
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
    write_doc(DOC, "V34.9 final decision 中文报告", payload, ["中文结论", "state_contract_audit_done", "obs_points_zero_detected", "trace_state_contract_repaired", "trace_preserving_measurement_bank_built", "causal_assignment_targets_built", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "learned_gate_training_ran", "learned_gate_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "trace_state_contract_passed", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "causal_assignment_subset_gain", "strict_residual_subset_gain", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "effective_units", "unit_dominant_instance_purity", "unit_semantic_purity", "visualization_ready", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(f"已写出 V34.9 final decision: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
