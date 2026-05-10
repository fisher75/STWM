#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SUMMARY = ROOT / "reports/stwm_ostf_v33_13_semantic_target_space_probe_summary_20260510.json"
DECISION = ROOT / "reports/stwm_ostf_v33_13_semantic_target_space_probe_decision_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_13_SEMANTIC_TARGET_SPACE_PROBE_DECISION_20260510.md"


def main() -> int:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    best = payload.get("best_probe_by_val")
    learnability = bool(payload.get("target_space_learnability_passed"))
    decision = {
        "generated_at_utc": utc_now(),
        "target_space_probe_done": bool(payload),
        "best_probe_by_val": best,
        "target_space_learnability_passed": learnability,
        "target_space_learnability_failed": bool(payload) and not learnability,
        "architecture_or_loss_bottleneck": learnability,
        "recommended_next_step": "fix_gate_repaired_model_loss" if learnability else "build_real_stronger_teacher_targets",
    }
    dump_json(DECISION, decision)
    write_doc(DOC, "STWM OSTF V33.13 Semantic Target-Space Probe Decision", decision, ["target_space_probe_done", "best_probe_by_val", "target_space_learnability_passed", "target_space_learnability_failed", "architecture_or_loss_bottleneck", "recommended_next_step"])
    print(DECISION.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
