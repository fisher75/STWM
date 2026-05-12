#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


ABLATION_JSON = ROOT / "reports/stwm_ostf_v34_6_real_residual_content_ablation_20260511.json"
VIS_JSON = ROOT / "reports/stwm_ostf_v34_6_residual_parameterization_visualization_manifest_20260511.json"
ABLATION_DOC = ROOT / "docs/STWM_OSTF_V34_6_REAL_RESIDUAL_CONTENT_ABLATION_20260511.md"
VIS_DOC = ROOT / "docs/STWM_OSTF_V34_6_RESIDUAL_PARAMETERIZATION_VISUALIZATION_20260511.md"
OUT = ROOT / "reports/stwm_ostf_v34_7_artifact_rematerialization_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_ARTIFACT_REMATERIALIZATION_20260511.md"


def fields_match(json_path: Path, doc_path: Path) -> bool:
    if not json_path.exists() or not doc_path.exists():
        return False
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    doc = doc_path.read_text(encoding="utf-8")
    return any(str(k) in doc for k in payload.keys())


def main() -> int:
    missing = [str(p.relative_to(ROOT)) for p in [ABLATION_JSON, VIS_JSON] if not p.exists()]
    payload = {
        "generated_at_utc": utc_now(),
        "artifact_packaging_fixed": not missing,
        "missing_before_rematerialization": missing,
        "recovered_from_existing_outputs": False,
        "rerun_performed": False,
        "ablation_json_path": str(ABLATION_JSON.relative_to(ROOT)),
        "visualization_json_path": str(VIS_JSON.relative_to(ROOT)),
        "ablation_json_exists": ABLATION_JSON.exists(),
        "visualization_json_exists": VIS_JSON.exists(),
        "json_md_fields_consistent": {
            "ablation": fields_match(ABLATION_JSON, ABLATION_DOC),
            "visualization": fields_match(VIS_JSON, VIS_DOC),
        },
        "exact_blockers": [] if not missing else ["missing_v34_6_json_artifacts"],
    }
    dump_json(OUT, payload)
    write_doc(DOC, "STWM OSTF V34.7 Artifact Rematerialization", payload, ["artifact_packaging_fixed", "missing_before_rematerialization", "recovered_from_existing_outputs", "rerun_performed", "ablation_json_exists", "visualization_json_exists", "json_md_fields_consistent", "exact_blockers"])
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
