#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_3_artifact_truth_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_3_ARTIFACT_TRUTH_20260509.md"

REQUIRED_JSON = [
    "reports/stwm_ostf_v33_2_artifact_truth_and_claims_20260509.json",
    "reports/stwm_ostf_v33_2_visual_teacher_preflight_20260509.json",
    "reports/stwm_ostf_v33_2_visual_teacher_prototype_build_20260509.json",
    "reports/stwm_ostf_v33_2_hard_identity_semantic_subset_20260509.json",
    "reports/stwm_ostf_v33_2_visual_semantic_identity_eval_20260509.json",
    "reports/stwm_ostf_v33_2_semantic_identity_visualization_manifest_20260509.json",
]

RELATED_DOC = {
    "reports/stwm_ostf_v33_2_artifact_truth_and_claims_20260509.json": "docs/STWM_OSTF_V33_2_ARTIFACT_TRUTH_AND_CLAIMS_20260509.md",
    "reports/stwm_ostf_v33_2_visual_teacher_preflight_20260509.json": "docs/STWM_OSTF_V33_2_VISUAL_TEACHER_PREFLIGHT_20260509.md",
    "reports/stwm_ostf_v33_2_visual_teacher_prototype_build_20260509.json": "docs/STWM_OSTF_V33_2_VISUAL_TEACHER_PROTOTYPE_BUILD_20260509.md",
    "reports/stwm_ostf_v33_2_hard_identity_semantic_subset_20260509.json": "docs/STWM_OSTF_V33_2_HARD_IDENTITY_SEMANTIC_SUBSET_20260509.md",
    "reports/stwm_ostf_v33_2_visual_semantic_identity_eval_20260509.json": "docs/STWM_OSTF_V33_2_VISUAL_SEMANTIC_IDENTITY_EVAL_20260509.md",
    "reports/stwm_ostf_v33_2_semantic_identity_visualization_manifest_20260509.json": "docs/STWM_OSTF_V33_2_SEMANTIC_IDENTITY_TRACE_FIELD_VISUALIZATION_20260509.md",
}


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"json_load_error": str(exc)}


def main() -> int:
    artifacts: dict[str, Any] = {}
    missing_json: list[str] = []
    doc_without_json: list[str] = []
    json_load_errors: dict[str, str] = {}
    for rel in REQUIRED_JSON:
        path = ROOT / rel
        doc = ROOT / RELATED_DOC.get(rel, "")
        payload = load(path)
        exists = path.exists()
        if not exists:
            missing_json.append(rel)
            if doc.exists():
                doc_without_json.append(rel)
        if "json_load_error" in payload:
            json_load_errors[rel] = str(payload["json_load_error"])
        artifacts[rel] = {
            "json_exists": exists,
            "doc_exists": doc.exists(),
            "size_bytes": path.stat().st_size if exists else 0,
            "json_loadable": exists and "json_load_error" not in payload,
        }

    final_decision_path = ROOT / "reports/stwm_ostf_v33_2_visual_semantic_identity_decision_20260509.json"
    smoke_summary = load(ROOT / "reports/stwm_ostf_v33_2_visual_semantic_identity_smoke_summary_20260509.json")
    smoke_decision = load(ROOT / "reports/stwm_ostf_v33_2_visual_semantic_identity_smoke_decision_20260509.json")
    final_decision = load(final_decision_path)
    decision_source_missing = bool(missing_json) and final_decision_path.exists()
    source_json_complete = not missing_json and not json_load_errors
    payload = {
        "generated_at_utc": utc_now(),
        "artifact_truth_ok": source_json_complete and not doc_without_json,
        "source_json_complete": source_json_complete,
        "decision_source_missing": decision_source_missing,
        "missing_json": missing_json,
        "doc_without_json": doc_without_json,
        "json_load_errors": json_load_errors,
        "artifacts": artifacts,
        "smoke_passed": bool(smoke_summary.get("smoke_passed", False)),
        "smoke_decision_identity_claim": bool(smoke_decision.get("integrated_identity_field_claim_allowed", False)),
        "final_decision_identity_claim": bool(final_decision.get("integrated_identity_field_claim_allowed", False)),
        "claim_contradiction_detected": bool(
            smoke_decision
            and final_decision
            and bool(smoke_decision.get("integrated_identity_field_claim_allowed", False))
            != bool(final_decision.get("integrated_identity_field_claim_allowed", False))
        ),
        "final_decision_not_source_truth": True,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.3 Artifact Truth",
        payload,
        [
            "artifact_truth_ok",
            "source_json_complete",
            "decision_source_missing",
            "missing_json",
            "doc_without_json",
            "smoke_passed",
            "claim_contradiction_detected",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
