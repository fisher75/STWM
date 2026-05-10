#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_7_identity_belief_ablation_summary_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_7_IDENTITY_BELIEF_ABLATION_20260509.md"
MAIN = ROOT / "reports/stwm_ostf_v33_7_identity_belief_eval_summary_20260509.json"
ABLATIONS = {
    "no_hard_bce": ROOT / "reports/stwm_ostf_v33_7_no_hard_bce_eval_summary_20260509.json",
    "no_embedding_similarity_logits": ROOT / "reports/stwm_ostf_v33_7_no_embedding_similarity_logits_eval_summary_20260509.json",
    "no_fused_logits": ROOT / "reports/stwm_ostf_v33_7_no_fused_logits_eval_summary_20260509.json",
}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def mean(payload: dict[str, Any], metric: str, split: str) -> float | None:
    val = payload.get("metrics", {}).get(metric, {}).get(split, {}).get("mean")
    return float(val) if val is not None else None


def compare(main: dict[str, Any], other: dict[str, Any], metric: str, split: str) -> float | None:
    a = mean(main, metric, split)
    b = mean(other, metric, split)
    return None if a is None or b is None else float(a - b)


def main() -> int:
    main_payload = load(MAIN)
    rows: dict[str, Any] = {}
    for name, path in ABLATIONS.items():
        other = load(path)
        rows[name] = {
            "eval_exists": bool(other),
            "hard_identity_ROC_AUC_fused_delta_val": compare(main_payload, other, "hard_identity_ROC_AUC_fused", "val"),
            "hard_identity_ROC_AUC_fused_delta_test": compare(main_payload, other, "hard_identity_ROC_AUC_fused", "test"),
            "val_calibrated_balanced_accuracy_delta_val": compare(main_payload, other, "val_calibrated_balanced_accuracy", "val"),
            "val_calibrated_balanced_accuracy_delta_test": compare(main_payload, other, "val_calibrated_balanced_accuracy", "test"),
            "strict_retrieval_delta_val": compare(main_payload, other, "identity_retrieval_exclude_same_point_top1", "val"),
            "strict_retrieval_delta_test": compare(main_payload, other, "identity_retrieval_exclude_same_point_top1", "test"),
        }
    payload = {
        "generated_at_utc": utc_now(),
        "main_eval_summary_path": str(MAIN.relative_to(ROOT)),
        "ablation_reports": {k: str(v.relative_to(ROOT)) for k, v in ABLATIONS.items()},
        "ablations": rows,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.7 Identity Belief Ablation", payload, ["main_eval_summary_path", "ablations"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
