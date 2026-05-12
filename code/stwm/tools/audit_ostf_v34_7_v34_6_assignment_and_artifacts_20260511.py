#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


DECISION = ROOT / "reports/stwm_ostf_v34_6_decision_20260511.json"
SWEEP_DECISION = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_decision_20260511.json"
EVAL_SUMMARY = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_eval_summary_20260511.json"
TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_train_summary_20260511.json"
ABLATION_JSON = ROOT / "reports/stwm_ostf_v34_6_real_residual_content_ablation_20260511.json"
VIS_JSON = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_visualization_manifest_20260511.json"
ABLATION_DOC = ROOT / "docs/STWM_OSTF_V34_6_REAL_RESIDUAL_CONTENT_ABLATION_20260511.md"
VIS_DOC = ROOT / "docs/STWM_OSTF_V34_6_RESIDUAL_PARAMETERIZATION_VISUALIZATION_20260511.md"
ABLATION_CODE = ROOT / "code/stwm/tools/eval_ostf_v34_6_real_residual_content_ablation_20260511.py"
VIS_CODE = ROOT / "code/stwm/tools/render_ostf_v34_6_residual_parameterization_visualizations_20260511.py"
OUT = ROOT / "reports/stwm_ostf_v34_7_v34_6_assignment_and_artifact_audit_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_V34_6_ASSIGNMENT_AND_ARTIFACT_AUDIT_20260511.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    decision = load(DECISION)
    ablation = load(ABLATION_JSON)
    ablation_missing = not ABLATION_JSON.exists()
    vis_missing = not VIS_JSON.exists()
    assign_delta = (ablation.get("strict_residual_subset_gain_delta") or {}).get("assignment_vs_shuffled") or {}
    normal = {s: (((ablation.get("per_split") or {}).get(s) or {}).get("normal") or {}).get("strict_residual_subset_gain") for s in ("val", "test")}
    shuffled = {s: (((ablation.get("per_split") or {}).get(s) or {}).get("residual_with_shuffled_unit_assignment") or {}).get("strict_residual_subset_gain") for s in ("val", "test")}
    assignment_not_lb = decision.get("assignment_load_bearing_on_residual") is False or bool(
        assign_delta and all(v is not None and v < 0.002 for v in assign_delta.values())
    )
    could_without_assignment = bool(assignment_not_lb and all(normal.get(s) is not None and shuffled.get(s) is not None and abs(normal[s] - shuffled[s]) < 0.002 for s in ("val", "test")))
    payload = {
        "generated_at_utc": utc_now(),
        "checked_files": [str(p.relative_to(ROOT)) for p in [DECISION, SWEEP_DECISION, EVAL_SUMMARY, TRAIN_SUMMARY, ABLATION_DOC, VIS_DOC, ABLATION_CODE, VIS_CODE]],
        "ablation_json_missing": ablation_missing,
        "visualization_json_missing": vis_missing,
        "decision_depends_on_missing_ablation_json": bool(ablation_missing and decision.get("assignment_load_bearing_on_residual") is not None),
        "artifact_packaging_fixed_required": bool(ablation_missing or vis_missing),
        "assignment_load_bearing_on_residual_from_v34_6": decision.get("assignment_load_bearing_on_residual"),
        "shuffled_assignment_delta": assign_delta,
        "assignment_vs_normal_residual_gain_delta": {"normal": normal, "shuffled_assignment": shuffled},
        "assignment_not_load_bearing_confirmed": assignment_not_lb,
        "unit_memory_gain_could_be_produced_without_meaningful_assignment": could_without_assignment,
        "unit_memory_not_assignment_bound_suspected": could_without_assignment,
        "current_residual_effectively_global_or_pointwise_suspected": could_without_assignment,
        "exact_evidence": {
            "assignment_vs_shuffled_val": assign_delta.get("val"),
            "assignment_vs_shuffled_test": assign_delta.get("test"),
            "normal_val": normal.get("val"),
            "shuffled_val": shuffled.get("val"),
            "normal_test": normal.get("test"),
            "shuffled_test": shuffled.get("test"),
        },
        "exact_code_locations": {
            "ablation_assignment_intervention": "code/stwm/tools/eval_ostf_v34_6_real_residual_content_ablation_20260511.py: residual_with_shuffled_unit_assignment",
            "v34_6_model_point_residual": "code/stwm/modules/ostf_v34_3_pointwise_unit_residual_world_model.py: semantic_residual_head(unit_hidden)",
            "v34_6_decision_assignment_field": "reports/stwm_ostf_v34_6_decision_20260511.json: assignment_load_bearing_on_residual",
        },
        "recommended_fix": "Build assignment-aware residual targets and force residual readout through unit memories mixed by point_to_unit_assignment before any learned gate training.",
    }
    dump_json(OUT, payload)
    write_doc(DOC, "STWM OSTF V34.7 V34.6 Assignment And Artifact Audit", payload, ["ablation_json_missing", "visualization_json_missing", "artifact_packaging_fixed_required", "assignment_not_load_bearing_confirmed", "shuffled_assignment_delta", "unit_memory_not_assignment_bound_suspected", "recommended_fix"])
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
