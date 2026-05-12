#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRAIN = ROOT / "reports/stwm_ostf_v34_7_assignment_residual_gate_train_summary_20260511.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_7_assignment_residual_gate_eval_summary_20260511.json"
DECISION = ROOT / "reports/stwm_ostf_v34_7_assignment_residual_gate_decision_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_7_ASSIGNMENT_RESIDUAL_GATE_DECISION_20260511.md"


def main() -> int:
    train = json.loads(TRAIN.read_text(encoding="utf-8")) if TRAIN.exists() else {}
    if not train.get("learned_gate_training_ran"):
        decision = {"generated_at_utc": utc_now(), "learned_gate_training_ran": False, "learned_gate_passed": "not_run", "semantic_gate_order_ok": "not_run", "skip_reason": train.get("skip_reason", "gate_training_not_run"), "recommended_next_step": "fix_assignment_bound_residual_model"}
        dump_json(SUMMARY, {"generated_at_utc": utc_now(), "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "STWM OSTF V34.7 Assignment Residual Gate Decision", decision, ["learned_gate_training_ran", "learned_gate_passed", "semantic_gate_order_ok", "skip_reason", "recommended_next_step"])
        print(SUMMARY.relative_to(ROOT))
        return 0
    decision = {"generated_at_utc": utc_now(), "learned_gate_training_ran": True, "learned_gate_passed": False, "semantic_gate_order_ok": False, "skip_reason": "full_gate_eval_not_enabled_in_this_run", "recommended_next_step": "fix_residual_gate"}
    dump_json(SUMMARY, {"generated_at_utc": utc_now(), "train_summary": train, "decision": decision})
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V34.7 Assignment Residual Gate Decision", decision, ["learned_gate_training_ran", "learned_gate_passed", "semantic_gate_order_ok", "recommended_next_step"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
