#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now

REPORT = ROOT / "reports/stwm_ostf_v33_2_artifact_truth_and_claims_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_2_ARTIFACT_TRUTH_AND_CLAIMS_20260509.md"

REQUIRED_JSON = [
    "reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json",
    "reports/stwm_ostf_v33_dense_field_target_coverage_20260509.json",
    "reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json",
    "reports/stwm_ostf_v33_semantic_identity_eval_20260509.json",
    "reports/stwm_ostf_v33_semantic_identity_schema_20260509.json",
    "reports/stwm_ostf_v33_1_sidecar_dataset_contract_20260509.json",
    "reports/stwm_ostf_v33_1_identity_field_visualization_manifest_20260509.json",
    "reports/stwm_ostf_v33_1_integrated_eval_20260509.json",
]


def load(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_json_error": f"{type(exc).__name__}: {exc}"}


def status(rel: str) -> dict[str, Any]:
    p = ROOT / rel
    return {"exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0}


def main() -> int:
    json_status = {rel: status(rel) for rel in REQUIRED_JSON}
    missing = [rel for rel, st in json_status.items() if not st["exists"]]
    smoke = load("reports/stwm_ostf_v33_1_integrated_smoke_summary_20260509.json")
    decision = load("reports/stwm_ostf_v33_1_integrated_decision_20260509.json")
    metrics = smoke.get("test_metrics", {})
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "json_artifacts": json_status,
        "exact_missing_artifacts": missing,
        "artifact_truth_ok": len(missing) == 0,
        "run_completed": bool(smoke.get("completed", False)),
        "smoke_training_completed": bool(smoke.get("completed", False) and smoke.get("integrated_v30_backbone_used", False)),
        "identity_load_bearing_passed": bool(decision.get("integrated_identity_field_claim_allowed", False)),
        "integrated_identity_field_claim_allowed": bool(decision.get("integrated_identity_field_claim_allowed", False)),
        "integrated_semantic_field_claim_allowed": bool(decision.get("integrated_semantic_field_claim_allowed", False)),
        "identity_ROC_AUC": metrics.get("identity_ROC_AUC"),
        "same_instance_balanced_accuracy": metrics.get("same_instance_balanced_accuracy"),
        "identity_PR_AUC": metrics.get("identity_PR_AUC"),
        "positive_ratio": metrics.get("positive_ratio"),
        "pr_auc_is_primary_success_metric": False,
        "primary_success_metrics": [
            "ROC_AUC",
            "balanced_accuracy",
            "hard_negative_retrieval_topk",
            "trivial_prior_beaten",
        ],
        "trajectory_degraded_false_means": "frozen V30 trajectory preserved; it does not mean semantic/identity improves trajectory",
        "claim_boundary": {
            "may_claim_integrated_identity_field": False,
            "may_claim_integrated_semantic_field": False,
            "reason": "V33.1 did not beat trivial identity prior; V33.2 must build visual semantic prototypes and hard negatives.",
        },
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.2 Artifact Truth and Claims", payload, [
        "artifact_truth_ok",
        "exact_missing_artifacts",
        "run_completed",
        "smoke_training_completed",
        "identity_load_bearing_passed",
        "integrated_identity_field_claim_allowed",
        "integrated_semantic_field_claim_allowed",
        "pr_auc_is_primary_success_metric",
        "trajectory_degraded_false_means",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
