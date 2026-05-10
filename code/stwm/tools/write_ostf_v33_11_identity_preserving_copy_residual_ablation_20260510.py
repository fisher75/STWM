#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_11_identity_preserving_copy_residual_ablation_summary_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_11_IDENTITY_PRESERVING_COPY_RESIDUAL_ABLATION_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def get(d: dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def delta(main: dict[str, Any], other: dict[str, Any], key: str) -> float | None:
    a = get(main, "decision", key, "test", "mean") or get(main, key, "test", "mean")
    b = get(other, "decision", key, "test", "mean") or get(other, key, "test", "mean")
    return None if a is None or b is None else float(a - b)


def main() -> int:
    main_eval = load("reports/stwm_ostf_v33_11_identity_preserving_copy_residual_eval_summary_20260510.json")
    no_id = load("reports/stwm_ostf_v33_11_no_identity_freeze_eval_summary_20260510.json")
    no_margin = load("reports/stwm_ostf_v33_11_no_stable_margin_eval_summary_20260510.json")
    no_gate = load("reports/stwm_ostf_v33_11_no_gate_focal_eval_summary_20260510.json")
    oracle = load("reports/stwm_ostf_v33_11_oracle_gate_upper_bound_20260510.json")
    main_dec = main_eval.get("decision", {})
    payload = {
        "generated_at_utc": utc_now(),
        "identity_freeze_load_bearing": (delta(main_eval, no_id, "hard_identity_ROC_AUC") or 0.0) >= 0.0,
        "stable_margin_load_bearing": bool(main_dec.get("stable_preservation_not_degraded_top5")) or ((delta(main_eval, no_margin, "stable_model_top5") or 0.0) > 0.0),
        "gate_focal_load_bearing": (delta(main_eval, no_gate, "semantic_change_AUROC") or 0.0) >= 0.0,
        "oracle_gap_remaining": bool(oracle.get("oracle_gate_passes") and not main_dec.get("pass_gate")),
        "which_component_fixes_stable_preservation": "stable_margin" if bool(main_dec.get("stable_preservation_not_degraded_top5")) else "unresolved",
        "which_component_prevents_identity_regression": "identity_freeze_or_distillation" if not bool(main_dec.get("identity_regressed_vs_v33_9")) else "unresolved",
        "main_eval_exists": bool(main_eval),
        "no_identity_freeze_eval_exists": bool(no_id),
        "no_stable_margin_eval_exists": bool(no_margin),
        "no_gate_focal_eval_exists": bool(no_gate),
        "oracle_gate_eval_exists": bool(oracle),
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.11 Identity-Preserving Copy Residual Ablation", payload, ["identity_freeze_load_bearing", "stable_margin_load_bearing", "gate_focal_load_bearing", "oracle_gap_remaining", "which_component_fixes_stable_preservation", "which_component_prevents_identity_regression", "main_eval_exists", "no_identity_freeze_eval_exists", "no_stable_margin_eval_exists", "no_gate_focal_eval_exists"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
