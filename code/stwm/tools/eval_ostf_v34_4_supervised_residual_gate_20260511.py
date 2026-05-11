#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRAIN = ROOT / "reports/stwm_ostf_v34_4_supervised_residual_gate_train_summary_20260511.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_4_supervised_residual_gate_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_4_supervised_residual_gate_eval_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_4_SUPERVISED_RESIDUAL_GATE_EVAL_DECISION_20260511.md"


def main() -> int:
    train = json.loads(TRAIN.read_text(encoding="utf-8")) if TRAIN.exists() else {}
    payload = {
        "generated_at_utc": utc_now(),
        "supervised_residual_gate_training_ran": bool(train.get("supervised_residual_gate_training_ran", False)),
        "evaluation_ran": False,
        "skipped_reason": train.get("skipped_reason", "supervised residual gate checkpoint missing"),
    }
    decision = {
        **payload,
        "semantic_gate_order_ok": False,
        "pointwise_baseline_dominates": True,
        "residual_improves_over_pointwise_on_hard": False,
        "residual_does_not_degrade_stable": False,
        "trajectory_degraded": False,
        "future_leakage_detected": False,
    }
    dump_json(SUMMARY, payload)
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V34.4 Supervised Residual Gate Eval Decision", decision, ["supervised_residual_gate_training_ran", "evaluation_ran", "skipped_reason", "semantic_gate_order_ok", "pointwise_baseline_dominates", "residual_improves_over_pointwise_on_hard", "residual_does_not_degrade_stable"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
