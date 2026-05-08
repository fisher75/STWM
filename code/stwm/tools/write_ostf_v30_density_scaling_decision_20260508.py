#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v30_density_scaling_decision_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_SCALING_DECISION_20260508.md"


def read_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    m512 = read_optional(ROOT / "reports/stwm_ostf_v30_density_m512_pilot_decision_20260508.json") or {}
    m1024_smoke = read_optional(ROOT / "reports/stwm_ostf_v30_density_m1024_smoke_summary_20260508.json") or {}
    m1024_pilot = read_optional(ROOT / "reports/stwm_ostf_v30_density_m1024_pilot_decision_20260508.json") or {}
    m512_beats = bool(m512.get("m512_beats_m128_h32") and m512.get("m512_beats_m128_h64") and m512.get("m512_beats_m128_h96"))
    m512_positive = bool(
        m512.get("m512_h32_positive_vs_prior_seed_count", 0) >= 2
        and m512.get("m512_h64_positive_vs_prior_seed_count", 0) >= 2
        and m512.get("m512_h96_positive_vs_prior_seed_count", 0) >= 2
    )
    m1024_beats = bool(m1024_pilot.get("m1024_beats_m512"))
    if m512_beats and m1024_beats:
        rec = "run_m512_m1024_full_multiseed"
    elif m512_positive and not m512_beats:
        rec = "fix_density_aware_pooling"
    elif m512.get("next_step_choice") == "run_m1024_smoke_then_pilot":
        rec = "run_m512_m1024_full_multiseed"
    else:
        rec = "build_semantic_identity_targets" if m512_positive else "stop_density_scaling_keep_M128_main"
    payload = {
        "decision_name": "stwm_ostf_v30_density_scaling_decision",
        "generated_at_utc": utc_now(),
        "m128_h32_h64_h96_robust": True,
        "m512_pilot_completed": bool(m512),
        "m512_positive_vs_prior": m512_positive,
        "m512_beats_m128": m512_beats,
        "m1024_smoke_passed": bool(m1024_smoke.get("m1024_smoke_passed")),
        "m1024_pilot_completed": bool(m1024_pilot),
        "m1024_beats_m512": m1024_beats,
        "density_scaling_positive": bool(m512_beats and (not m1024_pilot or m1024_beats)),
        "density_scaling_boundary": "M128 remains main if higher density beats priors but fails to improve M128.",
        "semantic_not_tested_not_failed": True,
        "recommended_next_step": rec,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V30 Density Scaling Decision",
        payload,
        [
            "m128_h32_h64_h96_robust",
            "m512_pilot_completed",
            "m512_positive_vs_prior",
            "m512_beats_m128",
            "m1024_smoke_passed",
            "m1024_pilot_completed",
            "m1024_beats_m512",
            "density_scaling_positive",
            "recommended_next_step",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
