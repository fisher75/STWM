#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


EVAL_SUMMARY = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_eval_summary_20260510.json"
EVAL_DECISION = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_eval_decision_20260510.json"
DECISION = ROOT / "reports/stwm_ostf_v33_9_decision_20260510.json"
GATE_UTILS = ROOT / "code/stwm/tools/ostf_v33_9_semantic_gate_utils_20260510.py"
EVAL_SCRIPT = ROOT / "code/stwm/tools/eval_ostf_v33_9_fresh_expanded_identity_semantic_20260510.py"
VIZ = ROOT / "reports/stwm_ostf_v33_9_world_model_diagnostic_visualization_manifest_20260510.json"
REPORT = ROOT / "reports/stwm_ostf_v33_10_semantic_gate_failure_audit_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_10_SEMANTIC_GATE_FAILURE_AUDIT_20260510.md"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    summary = load(EVAL_SUMMARY)
    eval_decision = load(EVAL_DECISION)
    decision = load(DECISION)
    best_name = eval_decision.get("best_candidate_by_val")
    best = next((c for c in summary.get("candidates", []) if c.get("candidate") == best_name), {})
    semantic_rows = []
    for seed, split_rows in best.get("per_seed", {}).items():
        for split, row in split_rows.items():
            gates = row.get("semantic_gates", {})
            semantic_rows.append({"seed": seed, "split": split, **gates})
    changed_copy_zero = bool(semantic_rows) and all(float(r.get("changed_copy_top1") or 0.0) == 0.0 and float(r.get("changed_copy_top5") or 0.0) == 0.0 for r in semantic_rows)
    hard_copy_zero = bool(semantic_rows) and all(float(r.get("semantic_hard_copy_top1") or 0.0) == 0.0 and float(r.get("semantic_hard_copy_top5") or 0.0) == 0.0 for r in semantic_rows)
    stable_failed = bool(eval_decision.get("stable_preservation_not_degraded") is False)
    global_copy_strong = bool(
        (eval_decision.get("semantic_proto_copy_top1_val", {}).get("mean") or 0.0) > (eval_decision.get("semantic_proto_top1_val", {}).get("mean") or 1.0)
        and (eval_decision.get("semantic_proto_copy_top5_val", {}).get("mean") or 0.0) > (eval_decision.get("semantic_proto_top5_val", {}).get("mean") or 1.0)
    )
    utils_text = GATE_UTILS.read_text(encoding="utf-8") if GATE_UTILS.exists() else ""
    changed_defined_by_copy_diff = "copy != target" in utils_text or "(copy != target)" in utils_text
    strong_overclaims = bool(eval_decision.get("semantic_strong_gate_passed") and eval_decision.get("changed_top1_beats_copy") and changed_copy_zero)
    viz_payload = load(VIZ)
    placeholder = bool("placeholder" in json.dumps(viz_payload).lower() or not viz_payload.get("png_count"))
    payload = {
        "generated_at_utc": utc_now(),
        "best_candidate_by_val": best_name,
        "changed_subset_defined_by_copy_not_equal_target": changed_defined_by_copy_diff,
        "changed_copy_baseline_trivial_zero": changed_copy_zero,
        "semantic_hard_copy_baseline_trivial_zero": hard_copy_zero,
        "semantic_strong_gate_overclaims": strong_overclaims,
        "stable_preservation_failed": stable_failed,
        "global_copy_baseline_strong": global_copy_strong,
        "visualization_placeholder_detected": placeholder,
        "claim_boundary_fix_required": bool(strong_overclaims or stable_failed or placeholder),
        "semantic_rows_sample": semantic_rows[:6],
        "source_files_checked": [str(p.relative_to(ROOT)) for p in [EVAL_SUMMARY, EVAL_DECISION, DECISION, GATE_UTILS, EVAL_SCRIPT]],
        "rule": "If semantic_strong_gate_overclaims is true, integrated_semantic_field_claim_allowed must remain false.",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.10 Semantic Gate Failure Audit",
        payload,
        ["changed_copy_baseline_trivial_zero", "semantic_hard_copy_baseline_trivial_zero", "semantic_strong_gate_overclaims", "stable_preservation_failed", "global_copy_baseline_strong", "visualization_placeholder_detected", "claim_boundary_fix_required"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
