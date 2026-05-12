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


ORACLE = ROOT / "reports/stwm_ostf_v34_10_trace_contract_oracle_residual_probe_decision_20260512.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_10_trace_contract_residual_gate_train_summary_20260512.json"
DOC = ROOT / "docs/STWM_OSTF_V34_10_TRACE_CONTRACT_RESIDUAL_GATE_TRAIN_SUMMARY_20260512.md"


def main() -> int:
    argparse.ArgumentParser().parse_args()
    oracle = json.loads(ORACLE.read_text(encoding="utf-8")) if ORACLE.exists() else {}
    if not oracle.get("oracle_residual_probe_passed"):
        payload = {"generated_at_utc": utc_now(), "中文结论": "V34.10 oracle residual probe 未通过，按协议不训练 learned gate。", "learned_gate_training_ran": False, "learned_gate_passed": "not_run", "skip_reason": "oracle_residual_probe_not_passed", "v30_backbone_frozen": oracle.get("v30_backbone_frozen", True), "future_leakage_detected": oracle.get("future_leakage_detected", False), "trajectory_degraded": oracle.get("trajectory_degraded", False)}
        dump_json(SUMMARY, payload)
        write_doc(DOC, "V34.10 trace contract residual gate 训练中文摘要", payload, ["中文结论", "learned_gate_training_ran", "learned_gate_passed", "skip_reason"])
        print(f"已写出 V34.10 gate 训练跳过报告: {SUMMARY.relative_to(ROOT)}")
        return 0
    payload = {"generated_at_utc": utc_now(), "中文结论": "oracle 通过才应训练 gate；当前保护性脚本未自动启动 learned gate。", "learned_gate_training_ran": False, "learned_gate_passed": "not_run", "skip_reason": "gate_training_not_started"}
    dump_json(SUMMARY, payload)
    write_doc(DOC, "V34.10 trace contract residual gate 训练中文摘要", payload, ["中文结论", "learned_gate_training_ran", "learned_gate_passed", "skip_reason"])
    print(f"已写出 V34.10 gate 训练报告: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
