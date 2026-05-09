#!/usr/bin/env python3
from __future__ import annotations

import py_compile
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_latest_repo_truth_audit_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_LATEST_REPO_TRUTH_AUDIT_20260509.md"

KEY_FILES = [
    "code/stwm/modules/ostf_external_gt_world_model_v30.py",
    "code/stwm/tools/train_ostf_external_gt_v30_20260508.py",
    "code/stwm/tools/eval_ostf_external_gt_v30_20260508.py",
    "code/stwm/tools/aggregate_ostf_external_gt_v30_round2_multiseed_20260508.py",
    "code/stwm/modules/ostf_field_preserving_world_model_v31.py",
    "code/stwm/tools/train_ostf_field_preserving_v31_20260508.py",
    "code/stwm/tools/eval_ostf_field_preserving_v31_20260508.py",
    "code/stwm/tools/aggregate_ostf_v31_field_multiseed_20260508.py",
    "code/stwm/modules/ostf_recurrent_field_world_model_v32.py",
    "code/stwm/tools/train_ostf_recurrent_field_v32_20260509.py",
    "code/stwm/tools/eval_ostf_recurrent_field_v32_20260509.py",
    "code/stwm/tools/aggregate_ostf_recurrent_field_v32_pilot_20260509.py",
]

KEY_REPORTS = [
    "reports/stwm_ostf_v30_external_gt_round2_multiseed_decision_v2_20260508.json",
    "reports/stwm_ostf_v30_external_gt_h96_multiseed_decision_20260508.json",
    "reports/stwm_ostf_v31_field_multiseed_decision_20260508.json",
    "reports/stwm_ostf_v32_recurrent_field_pilot_decision_20260509.json",
    "reports/stwm_ostf_v32_semantic_target_route_audit_20260509.json",
]


def _compile(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "py_compile_ok": False, "exact_error": "missing"}
    try:
        py_compile.compile(str(path), doraise=True)
        return {"exists": True, "py_compile_ok": True, "exact_error": None}
    except Exception as exc:
        return {"exists": True, "py_compile_ok": False, "exact_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    files = {rel: _compile(ROOT / rel) for rel in KEY_FILES}
    reports = {rel: {"exists": (ROOT / rel).exists(), "size_bytes": (ROOT / rel).stat().st_size if (ROOT / rel).exists() else 0} for rel in KEY_REPORTS}
    v33_code = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "code/stwm/tools").glob("*v33*20260509.py"))
    v33_reports_before = [
        "reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json",
        "reports/stwm_ostf_v33_semantic_identity_target_construction_decision_20260509.json",
    ]
    target_reports_exist = any((ROOT / p).exists() for p in v33_reports_before)
    payload = {
        "generated_at_utc": utc_now(),
        "files": files,
        "reports": reports,
        "all_key_files_compile": all(v.get("py_compile_ok") for v in files.values()),
        "all_key_reports_exist": all(v.get("exists") for v in reports.values()),
        "v33_code_paths_seen_now": v33_code,
        "v33_target_construction_missing": not target_reports_exist,
        "v33_existing_code_consumes_targets": any("identity" in p or "semantic" in p for p in v33_code),
        "report_only_v33_detected": False,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 Latest Repo Truth Audit", payload, [
        "all_key_files_compile",
        "all_key_reports_exist",
        "v33_target_construction_missing",
        "v33_code_paths_seen_now",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
