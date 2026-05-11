#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v34_semantic_trace_unit_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V34_SEMANTIC_TRACE_UNIT_DECISION_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    audit = load("reports/stwm_ostf_v34_no_drift_route_audit_20260510.json")
    bank = load("reports/stwm_ostf_v34_semantic_measurement_bank_20260510.json")
    train = load("reports/stwm_ostf_v34_semantic_trace_units_train_summary_20260510.json")
    eval_dec = load("reports/stwm_ostf_v34_semantic_trace_units_eval_decision_20260510.json")
    base = load("reports/stwm_ostf_v34_against_teacher_prototype_baselines_20260510.json")
    viz = load("reports/stwm_ostf_v34_semantic_trace_unit_visualization_manifest_20260510.json")
    route_ok = bool(audit.get("teacher_target_route_diagnostic_only"))
    bank_ok = bool(bank.get("semantic_measurement_bank_built"))
    trace_ok = bool(train.get("trace_conditioned_semantic_units_built"))
    escaped = bool(base.get("does_v34_escape_teacher_only_path"))
    if not route_ok:
        nxt = "fix_semantic_measurement_bank"
    elif not bank_ok:
        nxt = "fix_semantic_measurement_bank"
    elif not trace_ok:
        nxt = "fix_trace_unit_binding"
    elif not escaped or not base.get("does_v34_improve_trace_conditioned_semantic_belief"):
        nxt = "fix_semantic_trace_unit_losses"
    elif bool(eval_dec.get("pass_gate")) and viz.get("visualization_ready"):
        nxt = "run_v34_seed123_replication"
    else:
        nxt = "fix_semantic_trace_unit_losses"
    payload = {
        "generated_at_utc": utc_now(),
        "teacher_target_route_diagnostic_only": route_ok,
        "semantic_measurement_bank_built": bank_ok,
        "trace_conditioned_semantic_units_built": trace_ok,
        "fresh_training_completed": bool(train.get("fresh_training_completed")),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": bool(eval_dec.get("future_leakage_detected", False)),
        "teacher_as_method": bool(eval_dec.get("teacher_as_method", True)),
        "outputs_future_trace_field": bool(eval_dec.get("outputs_future_trace_field")),
        "outputs_future_semantic_field": bool(eval_dec.get("outputs_future_semantic_field")),
        "hard_identity_ROC_AUC_val": eval_dec.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": eval_dec.get("hard_identity_ROC_AUC_test"),
        "semantic_belief_consistency": eval_dec.get("semantic_belief_consistency"),
        "stable_preservation": eval_dec.get("stable_preservation"),
        "changed_semantic_signal": eval_dec.get("changed_semantic_signal"),
        "semantic_uncertainty_quality": eval_dec.get("semantic_uncertainty_quality"),
        "trajectory_degraded": bool(eval_dec.get("trajectory_degraded")),
        "visualization_ready": bool(viz.get("visualization_ready")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": nxt,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34 Semantic Trace Unit Decision", payload, ["teacher_target_route_diagnostic_only", "semantic_measurement_bank_built", "trace_conditioned_semantic_units_built", "fresh_training_completed", "v30_backbone_frozen", "future_leakage_detected", "teacher_as_method", "outputs_future_trace_field", "outputs_future_semantic_field", "hard_identity_ROC_AUC_val", "hard_identity_ROC_AUC_test", "semantic_belief_consistency", "stable_preservation", "changed_semantic_signal", "semantic_uncertainty_quality", "trajectory_degraded", "visualization_ready", "integrated_identity_field_claim_allowed", "integrated_semantic_field_claim_allowed", "recommended_next_step"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
