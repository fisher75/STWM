#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


FILES = [
    ROOT / "code/stwm/modules/ostf_v34_semantic_trace_units.py",
    ROOT / "code/stwm/modules/ostf_v34_1_identity_bound_semantic_trace_units.py",
    ROOT / "code/stwm/tools/train_ostf_v34_1_identity_bound_semantic_trace_units_20260511.py",
    ROOT / "code/stwm/tools/eval_ostf_v34_1_identity_bound_semantic_trace_units_20260511.py",
    ROOT / "code/stwm/tools/build_ostf_v34_1_unit_identity_binding_targets_20260511.py",
]
EVAL = ROOT / "reports/stwm_ostf_v34_1_identity_bound_semantic_trace_units_eval_summary_20260511.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v34_1_identity_bound_semantic_trace_units_eval_decision_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_1_decision_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_2_v34_1_failure_attribution_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_2_V34_1_FAILURE_ATTRIBUTION_20260511.md"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    src = {str(p.relative_to(ROOT)): read(p) for p in FILES}
    model = src.get("code/stwm/modules/ostf_v34_1_identity_bound_semantic_trace_units.py", "")
    train = src.get("code/stwm/tools/train_ostf_v34_1_identity_bound_semantic_trace_units_20260511.py", "")
    bind = src.get("code/stwm/tools/build_ostf_v34_1_unit_identity_binding_targets_20260511.py", "")
    decision = load(DECISION)
    eval_dec = load(EVAL_DECISION)
    z_dyn_from_unit_sem = bool(re.search(r"state = self\.factorized_state\(unit_sem\)", model))
    fixed_ce = "F.nll_loss" in train and "point_to_unit_target" in train
    pointwise_reports = list((ROOT / "reports").glob("stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"))
    sem_failed = not any((decision.get("semantic_hard_signal") or {}).values()) and not any((decision.get("changed_semantic_signal") or {}).values())
    if z_dyn_from_unit_sem:
        attr = "unit_architecture"
    elif fixed_ce:
        attr = "loss_design"
    elif sem_failed:
        attr = "semantic_measurement_bank"
    else:
        attr = "unknown"
    payload = {
        "generated_at_utc": utc_now(),
        "z_dyn_source_is_trace_dynamics": False,
        "z_dyn_source_is_semantic_measurement": z_dyn_from_unit_sem,
        "z_sem_source_is_semantic_measurement": True,
        "z_dyn_z_sem_factorization_real": False,
        "identity_key_source": "V34.1 identity_key is derived from unit_sem through FactorizedTraceSemanticState, not from a dual-source z_dyn+z_sem state.",
        "unit_confidence_used_in_loss": "unit_confidence" in train and "uncertainty" in train,
        "point_to_unit_target_is_permutation_invariant": not fixed_ce,
        "real_pointwise_no_unit_baseline_exists": bool(pointwise_reports),
        "trace_units_better_than_pointwise_proven": False,
        "semantic_field_failed_because_units_or_targets_or_losses": attr,
        "exact_code_locations": {
            "z_dyn_from_semantic_state": "code/stwm/modules/ostf_v34_1_identity_bound_semantic_trace_units.py: state = self.factorized_state(unit_sem)",
            "fixed_unit_index_ce": "code/stwm/tools/train_ostf_v34_1_identity_bound_semantic_trace_units_20260511.py: F.nll_loss(assign[valid].log(), target[valid])",
            "binding_target_instance_ordering": "code/stwm/tools/build_ostf_v34_1_unit_identity_binding_targets_20260511.py: instance_to_unit orders instances by count and maps to fixed slots",
        },
        "v34_1_eval_decision": eval_dec,
        "v34_1_final_decision": decision,
        "recommended_fix": "Implement dual-source trace/semantic unit state, replace fixed slot CE with permutation-aware pairwise binding, and train a real pointwise no-unit baseline.",
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.2 V34.1 Failure Attribution",
        payload,
        [
            "z_dyn_source_is_trace_dynamics",
            "z_dyn_source_is_semantic_measurement",
            "z_sem_source_is_semantic_measurement",
            "z_dyn_z_sem_factorization_real",
            "identity_key_source",
            "unit_confidence_used_in_loss",
            "point_to_unit_target_is_permutation_invariant",
            "real_pointwise_no_unit_baseline_exists",
            "trace_units_better_than_pointwise_proven",
            "semantic_field_failed_because_units_or_targets_or_losses",
            "recommended_fix",
        ],
    )
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
