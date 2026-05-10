#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_14_v33_13_target_failure_audit_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_14_V33_13_TARGET_FAILURE_AUDIT_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    dec = load("reports/stwm_ostf_v33_13_decision_20260510.json")
    probe = load("reports/stwm_ostf_v33_13_semantic_target_space_probe_decision_20260510.json")
    probe_summary = load("reports/stwm_ostf_v33_13_semantic_target_space_probe_summary_20260510.json")
    preflight = load("reports/stwm_ostf_v33_13_real_stronger_teacher_preflight_20260510.json")
    eval_dec = load("reports/stwm_ostf_v33_13_gate_repaired_eval_decision_20260510.json")
    probes = probe_summary.get("probes", {})
    best = probe.get("best_probe_by_val") or probe_summary.get("best_probe_by_val")
    best_row = probes.get(best, {})
    val = best_row.get("val", {})
    changed_positive = bool(val.get("changed_top5_beats_strongest_baseline"))
    hard_positive = bool(val.get("semantic_hard_top5_beats_strongest_baseline"))
    actual_available = bool(preflight.get("any_stronger_teacher_available"))
    target_failed = bool(probe.get("target_space_probe_done") and not probe.get("target_space_learnability_passed"))
    payload = {
        "generated_at_utc": utc_now(),
        "gate_protocol_repaired": bool(dec.get("gate_protocol_repaired")),
        "double_sigmoid_bug_fixed": bool(dec.get("double_sigmoid_bug_fixed")),
        "train_eval_gate_consistent": bool(dec.get("train_eval_gate_consistent")),
        "target_space_probe_done": bool(probe.get("target_space_probe_done")),
        "target_space_learnability_passed": bool(probe.get("target_space_learnability_passed")),
        "best_probe_by_val": best,
        "changed_probe_signal_positive": changed_positive,
        "semantic_hard_probe_failed": not hard_positive,
        "only_clip_b32_available": bool(dec.get("only_clip_b32_available", preflight.get("only_clip_b32_available"))),
        "stronger_teacher_actual_available": actual_available,
        "why_CLIP_B32_K256_is_insufficient": "observed-context probe failed to beat strongest baseline on changed/hard semantic subsets; one-hot target-space oracle is not a learnability proof",
        "whether_next_step_should_be_loss_tuning_or_teacher_targets": "real_stronger_teacher_targets" if target_failed else "loss_tuning_allowed_after_probe_pass",
        "stable_preservation_not_degraded_top5": bool(eval_dec.get("stable_preservation_not_degraded_top5")),
        "changed_top5_beats_strongest_baseline": bool(eval_dec.get("changed_top5_beats_strongest_baseline")),
        "semantic_hard_top5_beats_strongest_baseline": bool(eval_dec.get("semantic_hard_top5_beats_strongest_baseline")),
        "recommended_focus": "real_stronger_teacher_targets" if target_failed else "model_loss_or_replication",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.14 V33.13 Target Failure Audit",
        payload,
        [
            "gate_protocol_repaired",
            "double_sigmoid_bug_fixed",
            "train_eval_gate_consistent",
            "target_space_probe_done",
            "target_space_learnability_passed",
            "best_probe_by_val",
            "changed_probe_signal_positive",
            "semantic_hard_probe_failed",
            "only_clip_b32_available",
            "stronger_teacher_actual_available",
            "whether_next_step_should_be_loss_tuning_or_teacher_targets",
            "recommended_focus",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
