#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_6_v34_5_residual_math_audit_20260511.json"
SWEEP = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"
ABLATION = ROOT / "reports/stwm_ostf_v34_6_real_residual_content_ablation_20260511.json"
GATE = ROOT / "reports/stwm_ostf_v34_6_best_residual_gate_decision_20260511.json"
VIS = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_visualization_manifest_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_6_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_6_DECISION_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pick_metric(sweep: dict[str, Any], key: str) -> Any:
    return sweep.get(key)


def main() -> int:
    audit = load(AUDIT)
    sweep = load(SWEEP)
    ablation = load(ABLATION)
    gate = load(GATE)
    vis = load(VIS)
    residual_pass = bool(sweep.get("residual_parameterization_passed"))
    ablation_done = bool(ablation.get("real_residual_content_ablation_done"))
    unit_lb = ablation.get("unit_memory_load_bearing_on_residual", "not_run")
    sem_lb = ablation.get("semantic_measurements_load_bearing_on_residual", "not_run")
    assign_lb = ablation.get("assignment_load_bearing_on_residual", "not_run")
    learned_gate_ran = bool(gate.get("learned_gate_training_ran"))
    learned_gate_passed = gate.get("learned_gate_passed", "not_run")
    if not residual_pass:
        recommended = "fix_semantic_measurement_bank" if sem_lb is False else "fix_unit_memory_residual_content"
    elif ablation_done and not (unit_lb is True and sem_lb is True and assign_lb is True):
        recommended = "fix_unit_memory_residual_content" if unit_lb is not True or assign_lb is not True else "fix_semantic_measurement_bank"
    elif learned_gate_ran and learned_gate_passed is not True:
        recommended = "fix_residual_gate"
    elif residual_pass and learned_gate_passed is True:
        recommended = "run_v34_6_seed123_replication"
    else:
        recommended = "fix_residual_gate"
    payload = {
        "generated_at_utc": utc_now(),
        "v34_5_residual_math_audit_done": bool(audit),
        "best_residual_parameterization": sweep.get("best_residual_parameterization"),
        "best_residual_init": sweep.get("best_residual_init"),
        "residual_parameterization_passed": residual_pass,
        "real_residual_content_ablation_done": ablation_done,
        "unit_memory_load_bearing_on_residual": unit_lb,
        "semantic_measurements_load_bearing_on_residual": sem_lb,
        "assignment_load_bearing_on_residual": assign_lb,
        "learned_gate_training_ran": learned_gate_ran,
        "learned_gate_passed": learned_gate_passed,
        "v30_backbone_frozen": bool(sweep.get("v30_backbone_frozen")),
        "future_leakage_detected": bool(sweep.get("future_leakage_detected", False) or gate.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(sweep.get("trajectory_degraded", False) or gate.get("trajectory_degraded", False)),
        "semantic_hard_signal": pick_metric(sweep, "semantic_hard_signal"),
        "changed_semantic_signal": pick_metric(sweep, "changed_semantic_signal"),
        "stable_preservation": pick_metric(sweep, "stable_preservation"),
        "pointwise_baseline_dominates": bool(sweep.get("pointwise_baseline_dominates", True)),
        "residual_improves_over_pointwise_on_hard": bool(sweep.get("residual_improves_over_pointwise_on_hard", False)),
        "residual_does_not_degrade_stable": bool(sweep.get("residual_does_not_degrade_stable", False)),
        "strict_residual_subset_gain": pick_metric(sweep, "strict_residual_subset_gain"),
        "delta_vs_v34_4_standalone_gain": pick_metric(sweep, "delta_vs_v34_4_standalone_gain"),
        "semantic_gate_order_ok": gate.get("semantic_gate_order_ok", "not_run") if learned_gate_ran else "not_run",
        "effective_units": pick_metric(sweep, "effective_units"),
        "unit_dominant_instance_purity": pick_metric(sweep, "unit_dominant_instance_purity"),
        "unit_semantic_purity": pick_metric(sweep, "unit_semantic_purity"),
        "visualization_ready": bool(vis.get("visualization_ready")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": recommended,
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.6 Decision",
        payload,
        [
            "v34_5_residual_math_audit_done",
            "best_residual_parameterization",
            "best_residual_init",
            "residual_parameterization_passed",
            "real_residual_content_ablation_done",
            "unit_memory_load_bearing_on_residual",
            "semantic_measurements_load_bearing_on_residual",
            "assignment_load_bearing_on_residual",
            "learned_gate_training_ran",
            "learned_gate_passed",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "pointwise_baseline_dominates",
            "residual_improves_over_pointwise_on_hard",
            "residual_does_not_degrade_stable",
            "strict_residual_subset_gain",
            "delta_vs_v34_4_standalone_gain",
            "semantic_gate_order_ok",
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
