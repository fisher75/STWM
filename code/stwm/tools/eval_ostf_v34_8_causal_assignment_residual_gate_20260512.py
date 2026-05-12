#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


TRAIN = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_gate_train_summary_20260512.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_gate_eval_summary_20260512.json"
DECISION = ROOT / "reports/stwm_ostf_v34_8_causal_assignment_residual_gate_decision_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_8_CAUSAL_ASSIGNMENT_RESIDUAL_GATE_DECISION_20260512.md"


def main() -> int:
    argparse.ArgumentParser().parse_args()
    train = json.loads(TRAIN.read_text(encoding="utf-8")) if TRAIN.exists() else {}
    if not train.get("learned_gate_training_ran"):
        decision = {
            "generated_at_utc": utc_now(),
            "中文结论": "V34.8 learned gate 未训练，因此 gate eval 为 not_run。",
            "learned_gate_training_ran": False,
            "learned_gate_passed": "not_run",
            "semantic_gate_order_ok": "not_run",
            "skip_reason": train.get("skip_reason", "gate_train_not_run"),
            "v30_backbone_frozen": train.get("v30_backbone_frozen", True),
            "future_leakage_detected": train.get("future_leakage_detected", False),
            "trajectory_degraded": train.get("trajectory_degraded", False),
        }
        dump_json(SUMMARY, {"generated_at_utc": utc_now(), "decision": decision})
        dump_json(DECISION, decision)
        write_doc(DOC, "V34.8 causal assignment residual gate 决策中文报告", decision, ["中文结论", "learned_gate_training_ran", "learned_gate_passed", "semantic_gate_order_ok", "skip_reason"])
        print(f"已写出 gate eval 跳过报告: {SUMMARY.relative_to(ROOT)}")
        return 0
    decision = {
        "generated_at_utc": utc_now(),
        "中文结论": "gate 训练报告存在但当前 eval 未实现实际 learned gate 评估。",
        "learned_gate_training_ran": True,
        "learned_gate_passed": False,
        "semantic_gate_order_ok": False,
        "recommended_next_step": "fix_residual_gate",
    }
    dump_json(SUMMARY, {"generated_at_utc": utc_now(), "decision": decision})
    dump_json(DECISION, decision)
    write_doc(DOC, "V34.8 causal assignment residual gate 决策中文报告", decision, ["中文结论", "learned_gate_training_ran", "learned_gate_passed", "semantic_gate_order_ok", "recommended_next_step"])
    print(f"已写出 gate eval 报告: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
