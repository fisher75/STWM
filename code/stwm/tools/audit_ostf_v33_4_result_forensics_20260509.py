#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_4_result_forensics_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_4_RESULT_FORENSICS_20260509.md"


def load(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    summary = load("reports/stwm_ostf_v33_3_structured_semantic_identity_smoke_summary_20260509.json")
    smoke_decision = load("reports/stwm_ostf_v33_3_structured_semantic_identity_smoke_decision_20260509.json")
    final_decision = load("reports/stwm_ostf_v33_3_structured_semantic_identity_decision_20260509.json")
    test = summary.get("test_metrics", {}) if isinstance(summary.get("test_metrics"), dict) else {}
    val = summary.get("val_metrics", {}) if isinstance(summary.get("val_metrics"), dict) else {}
    semantic_top1 = test.get("semantic_proto_top1")
    semantic_top5 = test.get("semantic_proto_top5")
    copy_top1 = test.get("semantic_proto_copy_top1")
    copy_top5 = test.get("semantic_proto_copy_top5")
    top1_beaten = bool(semantic_top1 is not None and copy_top1 is not None and float(semantic_top1) > float(copy_top1) + 1e-9)
    top5_beaten = bool(semantic_top5 is not None and copy_top5 is not None and float(semantic_top5) > float(copy_top5) + 1e-9)
    payload = {
        "generated_at_utc": utc_now(),
        "smoke_completed": bool(summary.get("completed", False)),
        "smoke_passed": bool(summary.get("smoke_passed", False)),
        "train_loss_decreased": bool(summary.get("train_loss_decreased", False)),
        "val_identity_ROC_AUC": val.get("identity_ROC_AUC"),
        "test_identity_ROC_AUC": test.get("identity_ROC_AUC"),
        "val_test_gap": test.get("val_test_gap"),
        "split_shift_suspected": bool(test.get("split_shift_suspected", False)),
        "hard_identity_ROC_AUC": test.get("hard_identity_ROC_AUC"),
        "hard_identity_balanced_accuracy": test.get("hard_identity_balanced_accuracy"),
        "identity_embedding_retrieval_top1": test.get("identity_embedding_retrieval_top1"),
        "identity_retrieval_prior_top1": test.get("identity_retrieval_prior_top1"),
        "semantic_proto_top1": semantic_top1,
        "semantic_proto_copy_top1": copy_top1,
        "semantic_proto_top5": semantic_top5,
        "semantic_proto_copy_top5": copy_top5,
        "semantic_top1_copy_beaten": top1_beaten,
        "semantic_top5_copy_beaten": top5_beaten,
        "semantic_copy_baseline_beaten_current_definition": bool(test.get("semantic_copy_baseline_beaten", False)),
        "trajectory_degraded": bool(test.get("trajectory_degraded", True)),
        "current_claim_allowed": {
            "smoke_decision_identity": bool(smoke_decision.get("integrated_identity_field_claim_allowed", False)),
            "smoke_decision_semantic": bool(smoke_decision.get("integrated_semantic_field_claim_allowed", False)),
            "final_decision_identity": bool(final_decision.get("integrated_identity_field_claim_allowed", False)),
            "final_decision_semantic": bool(final_decision.get("integrated_semantic_field_claim_allowed", False)),
        },
        "exact_protocol_risks": [
            "V33.3 semantic_copy_baseline_beaten mixed top1/top5; V33.4 splits semantic_top1_copy_beaten and semantic_top5_copy_beaten.",
            "V33.3 hard identity mask mixed identity negatives with semantic prototype changes.",
            "V33.3 raw identity retrieval can be inflated by same point / adjacent horizon near-duplicates.",
            "V33.3 validation identity metrics were near chance while test was positive.",
        ],
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.4 Result Forensics", payload, ["smoke_completed", "smoke_passed", "train_loss_decreased", "val_identity_ROC_AUC", "test_identity_ROC_AUC", "val_test_gap", "split_shift_suspected", "hard_identity_ROC_AUC", "hard_identity_balanced_accuracy", "identity_embedding_retrieval_top1", "identity_retrieval_prior_top1", "semantic_proto_top1", "semantic_proto_copy_top1", "semantic_proto_top5", "semantic_proto_copy_top5", "semantic_top1_copy_beaten", "semantic_top5_copy_beaten", "trajectory_degraded", "exact_protocol_risks"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
