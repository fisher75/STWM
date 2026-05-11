#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


ORACLE_DECISION = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_decision_20260511.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_4_supervised_residual_gate_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_4_SUPERVISED_RESIDUAL_GATE_TRAIN_SUMMARY_20260511.md"


def main() -> int:
    oracle = json.loads(ORACLE_DECISION.read_text(encoding="utf-8")) if ORACLE_DECISION.exists() else {}
    if not oracle.get("oracle_residual_probe_passed", False):
        payload = {
            "generated_at_utc": utc_now(),
            "supervised_residual_gate_training_ran": False,
            "skipped_reason": "oracle_residual_probe_failed_or_missing",
            "v30_backbone_frozen": bool(oracle.get("v30_backbone_frozen", True)),
            "future_leakage_detected": bool(oracle.get("future_leakage_detected", False)),
        }
        dump_json(SUMMARY, payload)
        write_doc(DOC, "STWM OSTF V34.4 Supervised Residual Gate Train Summary", payload, ["supervised_residual_gate_training_ran", "skipped_reason", "v30_backbone_frozen", "future_leakage_detected"])
        print(SUMMARY.relative_to(ROOT))
        return 0
    payload = {
        "generated_at_utc": utc_now(),
        "supervised_residual_gate_training_ran": False,
        "skipped_reason": "implementation_guard: run explicit full gate training after oracle pass review",
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.4 Supervised Residual Gate Train Summary", payload, ["supervised_residual_gate_training_ran", "skipped_reason", "v30_backbone_frozen", "future_leakage_detected"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
