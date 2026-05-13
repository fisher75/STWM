#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRAIN = ROOT / "reports/stwm_ostf_v34_12_local_evidence_residual_gate_train_summary_20260513.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_12_local_evidence_residual_gate_eval_summary_20260513.json"
DECISION = ROOT / "reports/stwm_ostf_v34_12_local_evidence_residual_gate_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_12_LOCAL_EVIDENCE_RESIDUAL_GATE_DECISION_20260513.md"


def main() -> int:
    train = json.loads(TRAIN.read_text(encoding="utf-8")) if TRAIN.exists() else {}
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 learned gate 未运行，因此评估为 not_run；不允许包装 semantic field success。",
        "learned_gate_training_ran": bool(train.get("learned_gate_training_ran", False)),
        "learned_gate_passed": "not_run",
        "semantic_gate_order_ok": "not_run",
        "skip_reason": train.get("skip_reason", "gate_train_not_run"),
    }
    dump_json(SUMMARY, {"generated_at_utc": utc_now(), "decision": decision})
    dump_json(DECISION, decision)
    write_doc(DOC, "V34.12 local evidence residual gate 决策中文报告", decision, ["中文结论", "learned_gate_training_ran", "learned_gate_passed", "semantic_gate_order_ok", "skip_reason"])
    print(f"已写出 V34.12 learned gate 评估跳过报告: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
