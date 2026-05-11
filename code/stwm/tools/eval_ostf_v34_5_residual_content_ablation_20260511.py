#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


DELTA = ROOT / "reports/stwm_ostf_v34_5_delta_residual_probe_decision_20260511.json"
STANDALONE = ROOT / "reports/stwm_ostf_v34_4_oracle_residual_probe_decision_20260511.json"
POINTWISE = ROOT / "reports/stwm_ostf_v34_2_pointwise_no_unit_eval_summary_20260511.json"
OUT = ROOT / "reports/stwm_ostf_v34_5_residual_content_ablation_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_5_RESIDUAL_CONTENT_ABLATION_20260511.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def sub(a: dict, b: dict) -> dict:
    return {k: None if a.get(k) is None or b.get(k) is None else float(a[k] - b[k]) for k in ("val", "test")}


def main() -> int:
    delta = load(DELTA)
    standalone = load(STANDALONE)
    pointwise = load(POINTWISE)
    delta_gain = delta.get("strict_residual_subset_gain") or {}
    standalone_gain = standalone.get("residual_utility_subset_gain") or {}
    delta_vs_standalone = sub(delta_gain, standalone_gain)
    payload = {
        "generated_at_utc": utc_now(),
        "compared_models": [
            "v34_4_standalone_target_residual",
            "v34_5_delta_residual",
            "v34_2_pointwise_no_unit",
            "oracle_target_upper_bound",
            "random_unit_residual",
            "residual_without_unit_memory",
            "residual_with_shuffled_unit_assignment",
        ],
        "strict_residual_subset_gain": delta_gain,
        "standalone_residual_subset_gain": standalone_gain,
        "semantic_hard_gain": delta.get("semantic_hard_signal"),
        "changed_gain": delta.get("changed_semantic_signal"),
        "stable_delta": delta.get("stable_preservation"),
        "force_gate_one_delta": "available_in_v34_3_failure_audit_and_not_reused_as_delta_gate",
        "drop_unit_memory_delta": "not_run_separately_for_delta_probe",
        "shuffle_assignment_delta": "not_run_separately_for_delta_probe",
        "delta_vs_standalone_gain": delta_vs_standalone,
        "whether_delta_objective_beats_standalone_objective": bool(all(v is not None and v > 0 for v in delta_vs_standalone.values())),
        "pointwise_no_unit_reference_available": bool(pointwise),
    }
    dump_json(OUT, payload)
    write_doc(DOC, "STWM OSTF V34.5 Residual Content Ablation", payload, ["compared_models", "strict_residual_subset_gain", "standalone_residual_subset_gain", "delta_vs_standalone_gain", "whether_delta_objective_beats_standalone_objective", "semantic_hard_gain", "changed_gain", "stable_delta"])
    print(OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
