#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_trajectory_backbone_freeze_audit_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_TRAJECTORY_BACKBONE_FREEZE_AUDIT_20260509.md"


def load(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    v30_r2 = load("reports/stwm_ostf_v30_external_gt_round2_multiseed_decision_v2_20260508.json")
    v30_h96 = load("reports/stwm_ostf_v30_external_gt_h96_multiseed_decision_20260508.json")
    v31 = load("reports/stwm_ostf_v31_field_multiseed_decision_20260508.json")
    v32 = load("reports/stwm_ostf_v32_recurrent_field_pilot_decision_20260509.json")
    density = load("reports/stwm_ostf_v30_density_scaling_decision_20260508.json")
    pooling = load("reports/stwm_ostf_v30_density_pooling_final_decision_20260508.json")
    v30_robust = bool(
        v30_r2.get("h32_positive_seed_count") == 5
        and v30_r2.get("h64_positive_seed_count") == 5
        and v30_h96.get("h96_positive_seed_count") == 5
        and v30_h96.get("trajectory_world_model_remains_robust_through_H96")
    )
    v31_beats = bool(v31.get("v31_overall_beats_v30"))
    v32_beats = bool(v32.get("v32_m128_matches_v30_for_multiseed_gate") or v32.get("recurrent_field_dynamics_positive"))
    density_positive = bool(density.get("density_scaling_positive") or pooling.get("density_scaling_recovered"))
    stop = v32_beats or density_positive
    payload = {
        "generated_at_utc": utc_now(),
        "v30_h32_h64_h96_robust": v30_robust,
        "v31_overall_beats_v30": v31_beats,
        "v32_beats_v30": v32_beats,
        "density_scaling_positive": density_positive,
        "semantic_not_tested_not_failed": bool(
            v30_r2.get("semantic_not_tested_not_failed")
            and v30_h96.get("semantic_not_tested_not_failed")
            and v32.get("semantic_not_tested_not_failed")
        ),
        "official_current_trajectory_backbone": "V30_M128",
        "further_trajectory_architecture_search_should_stop": bool(v30_robust and not v31_beats and not v32_beats and not density_positive),
        "next_focus": "semantic_identity_target_construction",
        "stop_due_to_v32_or_density_override": stop,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 Trajectory Backbone Freeze Audit", payload, [
        "v30_h32_h64_h96_robust",
        "v31_overall_beats_v30",
        "v32_beats_v30",
        "density_scaling_positive",
        "official_current_trajectory_backbone",
        "further_trajectory_architecture_search_should_stop",
        "next_focus",
    ])
    print(REPORT.relative_to(ROOT))
    return 0 if not stop else 2


if __name__ == "__main__":
    raise SystemExit(main())
