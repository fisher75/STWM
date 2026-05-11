#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


ATTR = ROOT / "reports/stwm_ostf_v34_2_v34_1_failure_attribution_20260511.json"
TRAIN = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_train_summary_20260511.json"
POINT_TRAIN = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_train_summary_20260511.json"
POINT_EVAL = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"
EVAL = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_eval_summary_20260511.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_eval_decision_20260511.json"
VIS = ROOT / "reports/stwm_ostf_v34_2_dual_source_visualization_manifest_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_2_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_2_DECISION_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    attr = load(ATTR)
    train = load(TRAIN)
    point_train = load(POINT_TRAIN)
    point_eval = load(POINT_EVAL)
    eval_dec = load(EVAL_DECISION)
    vis = load(VIS)
    z_dyn_ok = bool(train.get("z_dyn_source_is_trace_dynamics"))
    trace_better = bool(eval_dec.get("trace_units_better_than_pointwise"))
    sem_load = bool(eval_dec.get("semantic_measurements_load_bearing"))
    if not z_dyn_ok:
        rec = "fix_dual_source_unit_architecture"
    elif not trace_better:
        rec = "fix_dual_source_unit_architecture"
    elif not sem_load:
        rec = "fix_semantic_measurement_bank"
    elif not any((eval_dec.get("semantic_hard_signal") or {}).values()) and not any((eval_dec.get("changed_semantic_signal") or {}).values()):
        rec = "fix_semantic_trace_unit_losses"
    elif bool(eval_dec.get("pass_gate")) and bool(vis.get("visualization_ready")):
        rec = "run_v34_2_seed123_replication"
    else:
        rec = "fix_semantic_trace_unit_losses"
    payload = {
        "generated_at_utc": utc_now(),
        "v34_1_failure_attribution_done": bool(attr),
        "dual_source_model_built": bool(train.get("dual_source_model_built")),
        "permutation_aware_binding_active": bool(train.get("permutation_aware_binding_active")),
        "real_pointwise_no_unit_baseline_built": bool(point_train.get("real_pointwise_no_unit_baseline_built") and point_eval.get("real_pointwise_no_unit_baseline_built")),
        "fresh_training_completed": bool(train.get("fresh_training_completed")),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": bool(eval_dec.get("future_leakage_detected", True)),
        "trajectory_degraded": bool(eval_dec.get("trajectory_degraded", False)),
        "z_dyn_source_is_trace_dynamics": z_dyn_ok,
        "z_sem_source_is_semantic_measurement": bool(train.get("z_sem_source_is_semantic_measurement")),
        "z_dyn_z_sem_factorization_real": bool(train.get("z_dyn_z_sem_factorization_real")),
        "hard_identity_ROC_AUC_val": eval_dec.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": eval_dec.get("hard_identity_ROC_AUC_test"),
        "semantic_hard_signal": eval_dec.get("semantic_hard_signal"),
        "changed_semantic_signal": eval_dec.get("changed_semantic_signal"),
        "stable_preservation": eval_dec.get("stable_preservation"),
        "effective_units": eval_dec.get("effective_units"),
        "unit_dominant_instance_purity": eval_dec.get("unit_dominant_instance_purity"),
        "unit_semantic_purity": eval_dec.get("unit_semantic_purity"),
        "units_load_bearing": bool(eval_dec.get("units_load_bearing")),
        "semantic_measurements_load_bearing": sem_load,
        "trace_units_better_than_pointwise": trace_better,
        "visualization_ready": bool(vis.get("visualization_ready")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": rec,
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.2 Decision",
        payload,
        [
            "v34_1_failure_attribution_done",
            "dual_source_model_built",
            "permutation_aware_binding_active",
            "real_pointwise_no_unit_baseline_built",
            "fresh_training_completed",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "z_dyn_source_is_trace_dynamics",
            "z_sem_source_is_semantic_measurement",
            "z_dyn_z_sem_factorization_real",
            "hard_identity_ROC_AUC_val",
            "hard_identity_ROC_AUC_test",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "effective_units",
            "unit_dominant_instance_purity",
            "unit_semantic_purity",
            "units_load_bearing",
            "semantic_measurements_load_bearing",
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
