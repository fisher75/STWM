#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, load_json, write_doc


def _point_positive(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        float(a["point_L1_px"]) < float(b["point_L1_px"])
        and float(a["endpoint_error_px"]) < float(b["endpoint_error_px"])
        and float(a["PCK_16px"]) > float(b["PCK_16px"])
        and float(a["PCK_32px"]) > float(b["PCK_32px"])
    )


def main() -> int:
    cv = load_json(ROOT / "reports/stwm_ostf_v18_runs/constant_velocity_copy_seed42_h8.json")
    m128 = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m128_seed42_h8.json")
    m512 = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m512_seed42_h8.json")
    decision = load_json(ROOT / "reports/stwm_ostf_v18_decision_20260502.json")

    cvm = cv["test_metrics"]
    m128m = m128["test_metrics"]
    m512m = m512["test_metrics"]
    m128_positive = _point_positive(m128m, cvm)
    m512_positive = _point_positive(m512m, cvm)
    best_name = "V18_M128" if float(m128m["point_L1_px"]) < float(m512m["point_L1_px"]) else "V18_M512"
    best_metrics = m128m if best_name == "V18_M128" else m512m
    best_beats_cv = _point_positive(best_metrics, cvm)

    payload = {
        "audit_name": "stwm_ostf_v18_decision_logic_audit_v19",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "V18_M128_vs_constant_velocity_positive": m128_positive,
        "V18_M512_vs_constant_velocity_positive": m512_positive,
        "current_v18_decision_only_tracks_M512": True,
        "current_v18_decision_reason": "The V18 decision script hard-coded v18_physics_residual_m512_seed42_h8 as the single full-model row and ignored the stronger M128 point-metric result.",
        "best_M128_or_M512": best_name,
        "best_M128_or_M512_beats_constant_velocity": best_beats_cv,
        "semantic_oracle_leakage_exists": True,
        "metrics_allowed_for_claim": [
            "point_L1_px",
            "endpoint_error_px",
            "PCK_4px",
            "PCK_8px",
            "PCK_16px",
            "PCK_32px",
            "visibility_F1",
            "object_extent_iou",
            "corrected_semantic_top1",
            "corrected_semantic_top5",
        ],
        "metrics_must_be_discarded_or_qualified": [
            "old_analytic_semantic_top1",
            "old_analytic_semantic_top5",
        ],
        "v18_m128_test_metrics": m128m,
        "v18_m512_test_metrics": m512m,
        "constant_velocity_test_metrics": cvm,
        "prior_v18_decision": decision,
    }
    out = ROOT / "reports/stwm_ostf_v18_decision_logic_audit_v19_20260502.json"
    dump_json(out, payload)
    write_doc(
        ROOT / "docs/STWM_OSTF_V18_DECISION_LOGIC_AUDIT_V19_20260502.md",
        "STWM OSTF V18 Decision Logic Audit V19",
        payload,
        [
            "V18_M128_vs_constant_velocity_positive",
            "V18_M512_vs_constant_velocity_positive",
            "current_v18_decision_only_tracks_M512",
            "best_M128_or_M512",
            "best_M128_or_M512_beats_constant_velocity",
            "semantic_oracle_leakage_exists",
            "metrics_allowed_for_claim",
            "metrics_must_be_discarded_or_qualified",
        ],
    )
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
