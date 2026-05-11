#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


DUAL_MODEL = ROOT / "code/stwm/modules/ostf_v34_2_dual_source_semantic_trace_units.py"
POINT_MODEL = ROOT / "code/stwm/modules/ostf_v34_2_pointwise_no_unit_baseline.py"
DUAL_EVAL_TOOL = ROOT / "code/stwm/tools/eval_ostf_v34_2_dual_source_semantic_trace_units_20260511.py"
DUAL_SUMMARY = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_eval_summary_20260511.json"
DUAL_DECISION = ROOT / "reports/stwm_ostf_v34_2_dual_source_semantic_trace_units_eval_decision_20260511.json"
POINT_SUMMARY = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"
FINAL_DECISION = ROOT / "reports/stwm_ostf_v34_2_decision_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_3_v34_2_bottleneck_diagnosis_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_3_V34_2_BOTTLENECK_DIAGNOSIS_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def split_delta(dual: dict[str, Any], point: dict[str, Any], key: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for split in ("val", "test"):
        dv = dual.get("per_split", {}).get(split, {}).get(key)
        pv = point.get("per_split", {}).get(split, {}).get(key)
        out[split] = None if dv is None or pv is None else float(dv - pv)
    return out


def main() -> int:
    dual_summary = load(DUAL_SUMMARY)
    dual_decision = load(DUAL_DECISION)
    point = load(POINT_SUMMARY)
    final = load(FINAL_DECISION)
    teacher_delta = split_delta(dual_summary, point, "teacher_agreement_weighted_top5")
    identity_delta = split_delta(dual_summary, point, "hard_identity_ROC_AUC")
    interventions = dual_decision.get("intervention_summary", {})
    def get_intervention(mode: str, metric: str) -> dict[str, Any]:
        return {split: interventions.get(split, {}).get(mode, {}).get("metric_delta_vs_normal", {}).get(metric) for split in ("val", "test")}

    pointwise_dominates = bool(not final.get("trace_units_better_than_pointwise", False))
    payload = {
        "generated_at_utc": utc_now(),
        "checked_files": [str(p.relative_to(ROOT)) for p in [DUAL_MODEL, POINT_MODEL, DUAL_EVAL_TOOL, DUAL_SUMMARY, DUAL_DECISION, POINT_SUMMARY, FINAL_DECISION]],
        "v34_2_units_load_bearing": bool(final.get("units_load_bearing", False)),
        "v34_2_units_predictively_successful": bool(final.get("trace_units_better_than_pointwise", False) and (any((final.get("semantic_hard_signal") or {}).values()) or any((final.get("changed_semantic_signal") or {}).values()))),
        "pointwise_no_unit_dominates": pointwise_dominates,
        "teacher_top5_delta_vs_pointwise": teacher_delta,
        "identity_auc_delta_vs_pointwise": identity_delta,
        "drop_z_dyn_metric_delta": {
            "identity_auc": get_intervention("drop_z_dyn", "identity_auc"),
            "teacher_top5": get_intervention("drop_z_dyn", "teacher_top5"),
        },
        "drop_z_sem_metric_delta": {
            "identity_auc": get_intervention("drop_z_sem", "identity_auc"),
            "teacher_top5": get_intervention("drop_z_sem", "teacher_top5"),
        },
        "permute_assignment_metric_delta": {
            "identity_auc": get_intervention("permute_unit_assignment", "identity_auc"),
            "teacher_top5": get_intervention("permute_unit_assignment", "teacher_top5"),
        },
        "unit_bottleneck_detected": pointwise_dominates,
        "recommended_architecture_fix": "Use pointwise no-unit prediction as the preserved base path and restrict trace units to gated residual memory corrections for hard/changed/confuser cases.",
    }
    dump_json(OUT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.3 V34.2 Bottleneck Diagnosis",
        payload,
        [
            "v34_2_units_load_bearing",
            "v34_2_units_predictively_successful",
            "pointwise_no_unit_dominates",
            "teacher_top5_delta_vs_pointwise",
            "identity_auc_delta_vs_pointwise",
            "drop_z_dyn_metric_delta",
            "drop_z_sem_metric_delta",
            "permute_assignment_metric_delta",
            "unit_bottleneck_detected",
            "recommended_architecture_fix",
        ],
    )
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
