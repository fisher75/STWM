#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v34_no_drift_route_audit_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V34_NO_DRIFT_ROUTE_AUDIT_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    v3314 = load("reports/stwm_ostf_v33_14_decision_20260510.json")
    probe = load("reports/stwm_ostf_v33_14_teacher_target_space_probe_sweep_20260510.json")
    v3313 = load("reports/stwm_ostf_v33_13_decision_20260510.json")
    teacher_route_diag = bool(v3314.get("stage2_mainline_remains_trace_conditioned_semantic_trace_units", True))
    ready_training = False
    payload = {
        "generated_at_utc": utc_now(),
        "teacher_target_route_diagnostic_only": teacher_route_diag,
        "teacher_as_stage2_forbidden": True,
        "prototype_only_training_forbidden": True,
        "ready_for_v33_14_model_training": ready_training,
        "must_return_to_semantic_trace_units": True,
        "v30_backbone_must_remain_frozen": True,
        "current_stage2_goal": "trace_conditioned_semantic_trace_units",
        "v33_14_model_training_ran": bool(v3314.get("v33_14_model_training_ran", False)),
        "v33_14_target_space_learnability_passed": bool(v3314.get("target_space_learnability_passed", False) or probe.get("target_space_learnability_passed", False)),
        "v33_13_integrated_semantic_field_claim_allowed": bool(v3313.get("integrated_semantic_field_claim_allowed", False)),
        "exact_blockers": [] if teacher_route_diag else ["v33_14 route does not explicitly mark teacher target route diagnostic-only"],
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34 No-Drift Route Audit", payload, ["teacher_target_route_diagnostic_only", "teacher_as_stage2_forbidden", "prototype_only_training_forbidden", "ready_for_v33_14_model_training", "must_return_to_semantic_trace_units", "v30_backbone_must_remain_frozen", "current_stage2_goal", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
