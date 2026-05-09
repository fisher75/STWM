#!/usr/bin/env python3
from __future__ import annotations

import json

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_3_claim_boundary_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_3_CLAIM_BOUNDARY_20260509.md"


def load(rel: str) -> dict:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    smoke_summary = load("reports/stwm_ostf_v33_2_visual_semantic_identity_smoke_summary_20260509.json")
    smoke_decision = load("reports/stwm_ostf_v33_2_visual_semantic_identity_smoke_decision_20260509.json")
    final_decision = load("reports/stwm_ostf_v33_2_visual_semantic_identity_decision_20260509.json")
    val = smoke_summary.get("val_metrics", {}) if isinstance(smoke_summary.get("val_metrics", {}), dict) else {}
    test = smoke_summary.get("test_metrics", {}) if isinstance(smoke_summary.get("test_metrics", {}), dict) else {}
    claim_contradiction = bool(
        smoke_decision
        and final_decision
        and bool(smoke_decision.get("integrated_identity_field_claim_allowed", False))
        != bool(final_decision.get("integrated_identity_field_claim_allowed", False))
    )
    val_chance = (
        val.get("hard_identity_ROC_AUC") is not None
        and abs(float(val.get("hard_identity_ROC_AUC", 0.5)) - 0.5) < 0.03
        and abs(float(val.get("hard_identity_balanced_accuracy", 0.5)) - 0.5) < 0.03
    )
    test_positive = (
        float(test.get("hard_identity_ROC_AUC", 0.0) or 0.0) >= 0.60
        and float(test.get("hard_identity_balanced_accuracy", 0.0) or 0.0) >= 0.55
    )
    payload = {
        "generated_at_utc": utc_now(),
        "smoke_passed": bool(smoke_summary.get("smoke_passed", False)),
        "claim_contradiction_detected": claim_contradiction,
        "identity_claim_requires_split_audit": bool(val_chance and test_positive),
        "identity_signal_preliminary_positive": bool(test_positive),
        "identity_world_model_claim_allowed": False,
        "semantic_world_model_claim_allowed": False,
        "paper_claim_allowed": False,
        "reason": "V33.2 smoke_passed=false; smoke_decision/final_decision identity claim disagree; validation hard identity is near chance while test is positive; semantic copy baseline is not beaten.",
        "val_hard_identity_ROC_AUC": val.get("hard_identity_ROC_AUC"),
        "val_hard_identity_balanced_accuracy": val.get("hard_identity_balanced_accuracy"),
        "test_hard_identity_ROC_AUC": test.get("hard_identity_ROC_AUC"),
        "test_hard_identity_balanced_accuracy": test.get("hard_identity_balanced_accuracy"),
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.3 Claim Boundary",
        payload,
        [
            "smoke_passed",
            "claim_contradiction_detected",
            "identity_claim_requires_split_audit",
            "identity_signal_preliminary_positive",
            "identity_world_model_claim_allowed",
            "semantic_world_model_claim_allowed",
            "paper_claim_allowed",
            "reason",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
