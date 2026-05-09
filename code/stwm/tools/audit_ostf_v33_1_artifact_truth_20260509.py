#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_1_artifact_truth_audit_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_1_ARTIFACT_TRUTH_AUDIT_20260509.md"

JSONS = [
    "reports/stwm_ostf_v33_latest_repo_truth_refresh_20260509.json",
    "reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json",
    "reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json",
    "reports/stwm_ostf_v33_dense_field_target_coverage_20260509.json",
    "reports/stwm_ostf_v33_semantic_identity_eval_20260509.json",
    "reports/stwm_ostf_v33_semantic_identity_schema_20260509.json",
]
DOCS = [
    "docs/STWM_OSTF_V33_POINTODYSSEY_IDENTITY_TARGET_BUILD_20260509.md",
    "docs/STWM_OSTF_V33_DENSE_FIELD_TARGET_COVERAGE_20260509.md",
    "docs/STWM_OSTF_V33_VISUAL_TEACHER_PREFLIGHT_20260509.md",
    "docs/STWM_OSTF_V33_LATEST_REPO_TRUTH_REFRESH_20260509.md",
]

REGEN = {
    "reports/stwm_ostf_v33_latest_repo_truth_refresh_20260509.json": "PYTHONPATH=code /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/audit_ostf_v33_latest_repo_truth_refresh_20260509.py",
    "reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json": "PYTHONPATH=code /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/build_ostf_v33_pointodyssey_identity_targets_20260509.py --overwrite",
    "reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json": "PYTHONPATH=code /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/preflight_ostf_v33_visual_teacher_semantic_prototypes_20260509.py",
    "reports/stwm_ostf_v33_dense_field_target_coverage_20260509.json": "PYTHONPATH=code /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/audit_ostf_v33_dense_field_target_coverage_20260509.py",
    "reports/stwm_ostf_v33_semantic_identity_schema_20260509.json": "PYTHONPATH=code /home/chen034/miniconda3/envs/stwm/bin/python code/stwm/tools/ostf_v33_semantic_identity_schema_20260509.py",
}


def status(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    return {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}


def main() -> int:
    regenerated = []
    initial_missing = [rel for rel in JSONS if not (ROOT / rel).exists()]
    for rel in initial_missing:
        if rel in REGEN:
            subprocess.run(REGEN[rel], shell=True, cwd=ROOT, check=False)
            regenerated.append(rel)
    payload = {
        "generated_at_utc": utc_now(),
        "json_artifacts": {rel: status(rel) for rel in JSONS},
        "doc_artifacts": {rel: status(rel) for rel in DOCS},
        "initial_missing_json_artifacts": initial_missing,
        "regenerated_json_artifacts": regenerated,
        "exact_missing_artifacts": [rel for rel in JSONS + DOCS if not (ROOT / rel).exists()],
        "artifact_truth_ok": all((ROOT / rel).exists() for rel in JSONS + DOCS),
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.1 Artifact Truth Audit", payload, ["artifact_truth_ok", "initial_missing_json_artifacts", "regenerated_json_artifacts", "exact_missing_artifacts"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
