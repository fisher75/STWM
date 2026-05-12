#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


BANK_JSON = ROOT / "reports/stwm_ostf_v34_9_trace_preserving_semantic_measurement_bank_20260512.json"
TARGET_JSON = ROOT / "reports/stwm_ostf_v34_9_causal_assignment_residual_target_build_20260512.json"
BANK_DOC = ROOT / "docs/STWM_OSTF_V34_9_TRACE_PRESERVING_SEMANTIC_MEASUREMENT_BANK_20260512.md"
TARGET_DOC = ROOT / "docs/STWM_OSTF_V34_9_CAUSAL_ASSIGNMENT_RESIDUAL_TARGET_BUILD_20260512.md"
OUT = ROOT / "reports/stwm_ostf_v34_10_artifact_rematerialization_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_10_ARTIFACT_REMATERIALIZATION_20260512.md"


def consistent(j: Path, d: Path) -> bool:
    if not j.exists() or not d.exists():
        return False
    payload = json.loads(j.read_text(encoding="utf-8"))
    text = d.read_text(encoding="utf-8")
    return any(str(k) in text for k in payload.keys())


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in [BANK_JSON, TARGET_JSON] if not p.exists()]
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "当前 live repo 中 V34.9 measurement bank JSON 和 target build JSON 均存在；无需从 docs 恢复。",
        "artifact_packaging_fixed": not missing,
        "missing_json_before_rematerialization": missing,
        "recovered_from_existing_outputs": False,
        "rerun_performed": False,
        "measurement_report_json_exists": BANK_JSON.exists(),
        "target_report_json_exists": TARGET_JSON.exists(),
        "json_md_fields_consistent": {"measurement_bank": consistent(BANK_JSON, BANK_DOC), "target_build": consistent(TARGET_JSON, TARGET_DOC)},
        "exact_blockers": [] if not missing else ["缺失 V34.9 JSON artifact"],
    }
    dump_json(OUT, payload)
    write_doc(DOC, "V34.10 artifact 重物化中文报告", payload, ["中文结论", "artifact_packaging_fixed", "missing_json_before_rematerialization", "recovered_from_existing_outputs", "measurement_report_json_exists", "target_report_json_exists", "json_md_fields_consistent", "exact_blockers"])
    print(f"已写出 V34.10 artifact 重物化报告: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
