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


ORACLE = ROOT / "reports/stwm_ostf_v34_13_selector_conditioned_oracle_residual_probe_decision_20260513.json"
REPORT = ROOT / "reports/stwm_ostf_v34_13_selector_conditioned_residual_gate_train_summary_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_13_SELECTOR_CONDITIONED_RESIDUAL_GATE_TRAIN_SUMMARY_20260513.md"


def main() -> int:
    oracle = json.loads(ORACLE.read_text(encoding="utf-8")) if ORACLE.exists() else {}
    if not oracle.get("oracle_residual_probe_passed", False):
        payload = {
            "generated_at_utc": utc_now(),
            "中文结论": "V34.13 selector-conditioned oracle residual probe 未通过，learned gate 按协议跳过训练。",
            "learned_gate_training_ran": False,
            "learned_gate_passed": "not_run",
            "skip_reason": "oracle_residual_probe_not_passed",
        }
        dump_json(REPORT, payload)
        write_doc(DOC, "V34.13 selector-conditioned residual gate 训练跳过中文报告", payload, ["中文结论", "learned_gate_training_ran", "learned_gate_passed", "skip_reason"])
        print(f"已写出 V34.13 gate 训练跳过报告: {REPORT.relative_to(ROOT)}")
        return 0
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.13 oracle residual probe 已通过，但本脚本当前只负责协议守卫；如需训练 gate，应在下一轮显式启动。",
        "learned_gate_training_ran": False,
        "learned_gate_passed": "not_run",
        "skip_reason": "gate_training_not_started_by_protocol_guard",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "V34.13 selector-conditioned residual gate 训练守卫中文报告", payload, ["中文结论", "learned_gate_training_ran", "learned_gate_passed", "skip_reason"])
    print(f"已写出 V34.13 gate 训练守卫报告: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
