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


ORACLE = ROOT / "reports/stwm_ostf_v34_12_local_evidence_oracle_residual_probe_decision_20260513.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_12_local_evidence_residual_gate_train_summary_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_12_LOCAL_EVIDENCE_RESIDUAL_GATE_TRAIN_SUMMARY_20260513.md"


def main() -> int:
    oracle = json.loads(ORACLE.read_text(encoding="utf-8")) if ORACLE.exists() else {}
    ran = bool(oracle.get("oracle_residual_probe_passed", False))
    payload = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.12 learned gate 未训练；按规则只有 local evidence oracle residual probe 通过后才允许训练 gate。",
        "learned_gate_training_ran": False,
        "learned_gate_passed": "not_run",
        "skip_reason": "oracle_residual_probe_not_passed" if not ran else "not_requested_in_this_script",
        "oracle_residual_probe_passed": bool(oracle.get("oracle_residual_probe_passed", False)),
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "V34.12 local evidence residual gate 训练跳过中文报告", payload, ["中文结论", "learned_gate_training_ran", "learned_gate_passed", "skip_reason", "oracle_residual_probe_passed"])
    print(f"已写出 V34.12 learned gate 训练跳过报告: {SUMMARY.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
