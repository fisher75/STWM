#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_10_v34_9_trace_contract_and_semantic_path_audit_20260512.json"
ORACLE = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_decision_20260512.json"
TRAIN = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_train_summary_20260512.json"
GATE = ROOT / "reports/stwm_ostf_v34_10_trace_contract_residual_gate_decision_20260512.json"
VIS = ROOT / "reports/stwm_ostf_v34_10_trace_contract_residual_visualization_manifest_20260512.json"
OUT = ROOT / "reports/stwm_ostf_v34_10_decision_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_10_DECISION_20260512.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    audit, oracle, train, gate, vis = (load(p) for p in [AUDIT, ORACLE, TRAIN, GATE, VIS])
    trace_full = bool(oracle.get("trace_state_contract_fully_passed") and train.get("train_dataset_uses_real_obs_conf"))
    usage = bool(oracle.get("semantic_usage_loss_active") and train.get("semantic_usage_loss_active"))
    assign_loss = bool(oracle.get("assignment_contrast_loss_active") and train.get("assignment_contrast_loss_active"))
    gate_ran = bool(gate.get("learned_gate_training_ran"))
    gate_pass = gate.get("learned_gate_passed", "not_run")
    if not trace_full:
        rec = "fix_trace_state_contract"
    elif not usage or not assign_loss:
        rec = "fix_usage_and_assignment_losses"
    elif oracle.get("semantic_measurements_load_bearing_on_residual") is False:
        rec = "fix_semantic_measurement_bank"
    elif oracle.get("assignment_load_bearing_on_residual") is False:
        rec = "fix_assignment_bound_residual_model"
    elif oracle.get("oracle_residual_probe_passed") and gate_ran and gate_pass is not True:
        rec = "fix_residual_gate"
    elif oracle.get("oracle_residual_probe_passed") and gate_pass is True and not oracle.get("trajectory_degraded", True):
        rec = "run_v34_10_seed123_replication"
    else:
        rec = "fix_semantic_measurement_bank"
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.10 完成 trace-state contract 二次修复和 usage/assignment loss 激活后重跑 oracle probe；不声明 semantic field success。",
        "trace_contract_audit_done": bool(audit),
        "trace_state_contract_fully_passed": trace_full,
        "train_dataset_uses_real_obs_conf": bool(train.get("train_dataset_uses_real_obs_conf")),
        "semantic_usage_loss_active": usage,
        "assignment_contrast_loss_active": assign_loss,
        "oracle_residual_probe_ran": bool(oracle.get("oracle_residual_probe_ran")),
        "oracle_residual_probe_passed": bool(oracle.get("oracle_residual_probe_passed")),
        "learned_gate_training_ran": gate_ran,
        "learned_gate_passed": gate_pass,
        "v30_backbone_frozen": bool(oracle.get("v30_backbone_frozen", True)),
        "future_leakage_detected": bool(oracle.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(oracle.get("trajectory_degraded", False)),
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
    write_doc(DOC, "V34.10 final decision 中文报告", payload, ["中文结论", "trace_contract_audit_done", "trace_state_contract_fully_passed", "train_dataset_uses_real_obs_conf", "semantic_usage_loss_active", "assignment_contrast_loss_active", "oracle_residual_probe_ran", "oracle_residual_probe_passed", "learned_gate_training_ran", "learned_gate_passed", "v30_backbone_frozen", "future_leakage_detected", "trajectory_degraded", "semantic_hard_signal", "changed_semantic_signal", "stable_preservation", "causal_assignment_subset_gain", "strict_residual_subset_gain", "unit_memory_load_bearing_on_residual", "semantic_measurements_load_bearing_on_residual", "assignment_load_bearing_on_residual", "effective_units", "unit_dominant_instance_purity", "unit_semantic_purity", "visualization_ready", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(f"已写出 V34.10 final decision: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
