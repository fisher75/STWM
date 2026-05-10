#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


MAIN = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_eval_summary_20260510.json"
NO_COPY = ROOT / "reports/stwm_ostf_v33_10_no_copy_prior_eval_summary_20260510.json"
NO_GATE = ROOT / "reports/stwm_ostf_v33_10_no_change_gate_eval_summary_20260510.json"
REPORT = ROOT / "reports/stwm_ostf_v33_10_copy_residual_semantic_ablation_summary_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_10_COPY_RESIDUAL_SEMANTIC_ABLATION_20260510.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric(payload: dict[str, Any], key: str, split: str = "test") -> float | None:
    try:
        return float(payload["candidates"][0]["metrics"][key][split]["mean"])
    except Exception:
        return None


def delta(main: dict[str, Any], other: dict[str, Any], key: str, split: str = "test") -> float | None:
    a = metric(main, key, split)
    b = metric(other, key, split)
    return None if a is None or b is None else a - b


def main() -> int:
    m = load(MAIN)
    nc = load(NO_COPY)
    ng = load(NO_GATE)
    payload = {
        "generated_at_utc": utc_now(),
        "main_exists": bool(m),
        "no_copy_prior_exists": bool(nc),
        "no_change_gate_exists": bool(ng),
        "copy_residual_vs_no_copy_prior_delta": {
            "stable_top5_delta_test": delta(m, nc, "stable_model_top5"),
            "changed_top5_delta_test": delta(m, nc, "changed_model_top5"),
            "semantic_hard_top5_delta_test": delta(m, nc, "semantic_hard_model_top5"),
            "identity_auc_delta_test": delta(m, nc, "hard_identity_ROC_AUC"),
        },
        "copy_residual_vs_no_change_gate_delta": {
            "stable_top5_delta_test": delta(m, ng, "stable_model_top5"),
            "changed_top5_delta_test": delta(m, ng, "changed_model_top5"),
            "semantic_hard_top5_delta_test": delta(m, ng, "semantic_hard_model_top5"),
            "identity_auc_delta_test": delta(m, ng, "hard_identity_ROC_AUC"),
        },
        "stable_preservation_delta": delta(m, nc, "stable_model_top5"),
        "changed_hard_top5_delta": delta(m, nc, "semantic_hard_model_top5"),
        "identity_regression_delta": delta(m, nc, "hard_identity_ROC_AUC"),
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.10 Copy Residual Semantic Ablation", payload, ["main_exists", "no_copy_prior_exists", "no_change_gate_exists", "copy_residual_vs_no_copy_prior_delta", "copy_residual_vs_no_change_gate_delta"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
