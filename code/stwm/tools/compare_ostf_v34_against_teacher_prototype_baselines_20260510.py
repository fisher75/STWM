#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v34_against_teacher_prototype_baselines_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V34_AGAINST_TEACHER_PROTOTYPE_BASELINES_20260510.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main() -> int:
    v34 = load("reports/stwm_ostf_v34_semantic_trace_units_eval_decision_20260510.json")
    v3314 = load("reports/stwm_ostf_v33_14_teacher_target_space_probe_sweep_20260510.json")
    v3313 = load("reports/stwm_ostf_v33_13_decision_20260510.json")
    stable = v34.get("stable_preservation", {})
    changed = v34.get("changed_semantic_signal", {})
    escaped = bool(v34.get("outputs_future_semantic_field") and not v34.get("teacher_as_method"))
    improved = bool(stable.get("val") and changed.get("val"))
    payload = {
        "generated_at_utc": utc_now(),
        "baselines": {
            "v33_14_best_teacher_prototype_target_probe": {
                "best_teacher_by_val": v3314.get("best_teacher_by_val"),
                "target_space_learnability_passed": v3314.get("target_space_learnability_passed"),
                "changed_signal_positive": v3314.get("changed_signal_positive"),
                "semantic_hard_signal_positive": v3314.get("semantic_hard_signal_positive"),
            },
            "teacher_only_nearest_observed_measurement": "represented_by_copy_observed_measurement_baseline_in_v34_eval",
            "sample_frequency_baseline": "represented_by_v33_14_probe_sweep",
            "copy_baseline": "represented_by_v34_stable/copy cosine comparisons",
            "v33_13_gate_repaired_model": {
                "stable_preservation_not_degraded_top5": v3313.get("stable_preservation_not_degraded_top5"),
                "changed_top5_beats_strongest_baseline": v3313.get("changed_top5_beats_strongest_baseline"),
                "semantic_hard_top5_beats_strongest_baseline": v3313.get("semantic_hard_top5_beats_strongest_baseline"),
            },
            "v34_semantic_trace_units": {
                "stable_preservation": stable,
                "changed_semantic_signal": changed,
                "semantic_belief_consistency": v34.get("semantic_belief_consistency"),
            },
        },
        "does_v34_escape_teacher_only_path": escaped,
        "does_v34_use_teacher_as_measurement_only": bool(not v34.get("teacher_as_method") and not v34.get("future_leakage_detected")),
        "does_v34_improve_trace_conditioned_semantic_belief": improved,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V34 Against Teacher Prototype Baselines", payload, ["does_v34_escape_teacher_only_path", "does_v34_use_teacher_as_measurement_only", "does_v34_improve_trace_conditioned_semantic_belief", "baselines"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
