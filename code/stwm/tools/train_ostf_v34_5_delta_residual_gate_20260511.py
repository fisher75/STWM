#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


DELTA = ROOT / "reports/stwm_ostf_v34_5_delta_residual_probe_decision_20260511.json"
SUMMARY = ROOT / "reports/stwm_ostf_v34_5_delta_residual_gate_train_summary_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_5_DELTA_RESIDUAL_GATE_TRAIN_SUMMARY_20260511.md"


def main() -> int:
    delta = json.loads(DELTA.read_text(encoding="utf-8")) if DELTA.exists() else {}
    ran = bool(delta.get("delta_residual_probe_passed", False))
    payload = {
        "generated_at_utc": utc_now(),
        "learned_gate_training_ran": False,
        "skipped_reason": "delta_residual_probe_failed_or_missing" if not ran else "implementation_guard_before_seed_replication",
        "v30_backbone_frozen": bool(delta.get("v30_backbone_frozen", True)),
        "future_leakage_detected": bool(delta.get("future_leakage_detected", False)),
    }
    dump_json(SUMMARY, payload)
    write_doc(DOC, "STWM OSTF V34.5 Delta Residual Gate Train Summary", payload, ["learned_gate_training_ran", "skipped_reason", "v30_backbone_frozen", "future_leakage_detected"])
    print(SUMMARY.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
