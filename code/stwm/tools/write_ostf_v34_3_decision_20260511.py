#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


BOTTLENECK = ROOT / "reports/stwm_ostf_v34_3_v34_2_bottleneck_diagnosis_20260511.json"
TRAIN = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_train_summary_20260511.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_eval_decision_20260511.json"
VIS = ROOT / "reports/stwm_ostf_v34_3_pointwise_unit_residual_visualization_manifest_20260511.json"
MODEL = ROOT / "code/stwm/modules/ostf_v34_3_pointwise_unit_residual_world_model.py"
OUT = ROOT / "reports/stwm_ostf_v34_3_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_3_DECISION_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def choose_next(decision: dict[str, Any]) -> str:
    if decision.get("pointwise_baseline_dominates"):
        if decision.get("residual_improves_over_pointwise_on_hard") and not decision.get("residual_does_not_degrade_stable"):
            return "fix_residual_gate"
        return "fix_residual_gate" if not decision.get("semantic_gate_order_ok", False) else "fix_unit_memory_architecture"
    if decision.get("residual_improves_over_pointwise_on_hard") and not decision.get("residual_does_not_degrade_stable"):
        return "fix_residual_gate"
    if not decision.get("semantic_measurements_load_bearing", True):
        return "fix_semantic_measurement_bank"
    identity_auc = decision.get("hard_identity_ROC_AUC_test")
    sem_ok = any((decision.get("semantic_hard_signal") or {}).values()) or any((decision.get("changed_semantic_signal") or {}).values())
    if (identity_auc is not None and identity_auc < 0.55) and sem_ok:
        return "fix_identity_targets"
    if (
        decision.get("trace_units_better_than_pointwise")
        and decision.get("residual_does_not_degrade_stable")
        and not decision.get("trajectory_degraded")
        and decision.get("visualization_ready")
    ):
        return "run_v34_3_seed123_replication"
    return "fix_semantic_trace_unit_losses"


def main() -> int:
    bottleneck = load(BOTTLENECK)
    train = load(TRAIN)
    ev = load(EVAL_DECISION)
    vis = load(VIS)
    payload = {
        "generated_at_utc": utc_now(),
        "v34_2_bottleneck_diagnosis_done": bool(bottleneck),
        "pointwise_unit_residual_model_built": bool(MODEL.exists() and ev.get("pointwise_unit_residual_model_built")),
        "fresh_training_completed": bool(train.get("fresh_training_completed") and ev.get("fresh_training_completed")),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen") and ev.get("v30_backbone_frozen")),
        "future_leakage_detected": bool(ev.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(ev.get("trajectory_degraded", False)),
        "hard_identity_ROC_AUC_val": ev.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": ev.get("hard_identity_ROC_AUC_test"),
        "semantic_hard_signal": ev.get("semantic_hard_signal"),
        "changed_semantic_signal": ev.get("changed_semantic_signal"),
        "stable_preservation": ev.get("stable_preservation"),
        "pointwise_baseline_dominates": bool(ev.get("pointwise_baseline_dominates", True)),
        "residual_improves_over_pointwise_on_hard": bool(ev.get("residual_improves_over_pointwise_on_hard", False)),
        "residual_does_not_degrade_stable": bool(ev.get("residual_does_not_degrade_stable", False)),
        "semantic_residual_gate_mean_stable": ev.get("semantic_residual_gate_mean_stable"),
        "semantic_residual_gate_mean_changed": ev.get("semantic_residual_gate_mean_changed"),
        "semantic_residual_gate_mean_hard": ev.get("semantic_residual_gate_mean_hard"),
        "effective_units": ev.get("effective_units"),
        "unit_dominant_instance_purity": ev.get("unit_dominant_instance_purity"),
        "unit_semantic_purity": ev.get("unit_semantic_purity"),
        "units_load_bearing": bool(ev.get("units_load_bearing", False)),
        "semantic_measurements_load_bearing": bool(ev.get("semantic_measurements_load_bearing", False)),
        "trace_units_better_than_pointwise": bool(ev.get("trace_units_better_than_pointwise", False)),
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
    }
    payload["recommended_next_step"] = choose_next(payload | {"semantic_gate_order_ok": ev.get("semantic_gate_order_ok")})
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.3 Decision",
        payload,
        [
            "v34_2_bottleneck_diagnosis_done",
            "pointwise_unit_residual_model_built",
            "fresh_training_completed",
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
            "semantic_residual_gate_mean_stable",
            "semantic_residual_gate_mean_changed",
            "semantic_residual_gate_mean_hard",
            "effective_units",
            "unit_dominant_instance_purity",
            "unit_semantic_purity",
            "trace_units_better_than_pointwise",
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
