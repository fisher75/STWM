#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v30_density_pooling_final_decision_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_POOLING_FINAL_DECISION_20260508.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    forensic = load(ROOT / "reports/stwm_ostf_v30_density_failure_forensic_20260508.json")
    code = load(ROOT / "reports/stwm_ostf_v30_density_pooling_code_audit_20260508.json")
    smoke = load(ROOT / "reports/stwm_ostf_v30_density_pooling_smoke_summary_20260508.json")
    pilot = load(ROOT / "reports/stwm_ostf_v30_density_pooling_pilot_decision_20260508.json")
    pilot_exists = bool(pilot)
    payload = {
        "decision_name": "stwm_ostf_v30_density_pooling_final_decision",
        "generated_at_utc": utc_now(),
        "forensic_audit_path": "reports/stwm_ostf_v30_density_failure_forensic_20260508.json",
        "pooling_code_audit_path": "reports/stwm_ostf_v30_density_pooling_code_audit_20260508.json",
        "smoke_summary_path": "reports/stwm_ostf_v30_density_pooling_smoke_summary_20260508.json",
        "pilot_decision_path": "reports/stwm_ostf_v30_density_pooling_pilot_decision_20260508.json",
        "forensic_cache_bug_detected": bool(forensic.get("cache_manifest_eval_bug_detected")),
        "code_audit_passed": bool(code.get("code_audit_passed")),
        "smoke_passed": bool(smoke.get("smoke_passed")),
        "pilot_completed": pilot_exists,
        "density_scaling_recovered": bool(pilot.get("density_scaling_recovered", False)),
        "m512_beats_lower_density": bool(pilot.get("m512_beats_m128_after_pooling_fix", False)),
        "m1024_beats_lower_density": bool(pilot.get("m1024_beats_m512_after_pooling_fix", False)),
        "semantic_remains_not_tested": bool(pilot.get("semantic_remains_not_tested", True)),
        "recommended_next_step": pilot.get("recommended_next_step", "run_density_pooling_pilot" if smoke.get("smoke_passed") else "fix_density_pooling_smoke"),
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V30 Density Pooling Final Decision",
        payload,
        ["code_audit_passed", "smoke_passed", "pilot_completed", "density_scaling_recovered", "m512_beats_lower_density", "m1024_beats_lower_density", "semantic_remains_not_tested", "recommended_next_step"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
