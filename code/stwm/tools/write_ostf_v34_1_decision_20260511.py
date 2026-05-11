#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


FORENSICS = ROOT / "reports/stwm_ostf_v34_1_semantic_trace_unit_forensics_20260511.json"
PROBE = ROOT / "reports/stwm_ostf_v34_1_unit_intervention_probe_20260511.json"
BIND = ROOT / "reports/stwm_ostf_v34_1_unit_identity_binding_target_build_20260511.json"
TRAIN = ROOT / "reports/stwm_ostf_v34_1_identity_bound_semantic_trace_units_train_summary_20260511.json"
EVAL = ROOT / "reports/stwm_ostf_v34_1_identity_bound_semantic_trace_units_eval_summary_20260511.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v34_1_identity_bound_semantic_trace_units_eval_decision_20260511.json"
VIS = ROOT / "reports/stwm_ostf_v34_1_unit_loadbearing_visualization_manifest_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_1_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_1_DECISION_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    forensics = load(FORENSICS)
    probe = load(PROBE).get("decision", load(PROBE))
    bind = load(BIND)
    train = load(TRAIN)
    eval_dec = load(EVAL_DECISION)
    vis = load(VIS)
    units_load = bool(eval_dec.get("units_load_bearing", probe.get("units_load_bearing", False)))
    sem_load = bool(eval_dec.get("semantic_measurements_load_bearing", probe.get("semantic_measurements_load_bearing", False)))
    trace_better = bool(eval_dec.get("trace_units_better_than_pointwise", probe.get("trace_units_better_than_pointwise", False)))
    bind_ok = bool(bind.get("unit_identity_binding_targets_built") or bind.get("unit_target_feasibility"))
    traj_bad = bool(eval_dec.get("trajectory_degraded", False))
    visual = bool(vis.get("visualization_ready", False))
    if not bind_ok:
        rec = "fix_unit_binding_targets"
    elif not units_load:
        rec = "fix_unit_load_bearing_architecture"
    elif not sem_load:
        rec = "fix_semantic_measurement_bank"
    elif units_load and (not bool(eval_dec.get("pass_gate", False))):
        rec = "fix_semantic_trace_unit_losses"
    elif units_load and sem_load and trace_better and not traj_bad and visual:
        rec = "run_v34_1_seed123_replication"
    else:
        rec = "fix_semantic_trace_unit_losses"
    payload = {
        "generated_at_utc": utc_now(),
        "v34_forensics_done": bool(forensics),
        "units_load_bearing": units_load,
        "semantic_measurements_load_bearing": sem_load,
        "trace_units_better_than_pointwise": trace_better,
        "unit_identity_binding_targets_built": bind_ok,
        "identity_bound_model_built": bool(train.get("identity_bound_model_built")),
        "fresh_training_completed": bool(train.get("fresh_training_completed")),
        "v30_backbone_frozen": bool(train.get("v30_backbone_frozen")),
        "future_leakage_detected": bool(eval_dec.get("future_leakage_detected", train.get("future_leakage_detected", True))),
        "trajectory_degraded": traj_bad,
        "hard_identity_ROC_AUC_val": eval_dec.get("hard_identity_ROC_AUC_val"),
        "hard_identity_ROC_AUC_test": eval_dec.get("hard_identity_ROC_AUC_test"),
        "semantic_hard_signal": eval_dec.get("semantic_hard_signal"),
        "changed_semantic_signal": eval_dec.get("changed_semantic_signal"),
        "stable_preservation": eval_dec.get("stable_preservation"),
        "effective_units": eval_dec.get("effective_units"),
        "unit_dominant_instance_purity": eval_dec.get("unit_dominant_instance_purity"),
        "unit_semantic_purity": eval_dec.get("unit_semantic_purity"),
        "unit_intervention_delta": eval_dec.get("unit_intervention_delta"),
        "pointwise_no_unit_baseline_dominates": bool(eval_dec.get("pointwise_no_unit_baseline_dominates", probe.get("trace_units_not_better_than_pointwise", False))),
        "visualization_ready": visual,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": rec,
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.1 Decision",
        payload,
        [
            "v34_forensics_done",
            "units_load_bearing",
            "semantic_measurements_load_bearing",
            "trace_units_better_than_pointwise",
            "unit_identity_binding_targets_built",
            "identity_bound_model_built",
            "fresh_training_completed",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "hard_identity_ROC_AUC_val",
            "hard_identity_ROC_AUC_test",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
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
