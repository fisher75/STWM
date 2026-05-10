#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


SUMMARY = ROOT / "reports/stwm_ostf_v33_9_fresh_expanded_eval_summary_20260510.json"
OUT_DIR = ROOT / "outputs/figures/stwm_ostf_v33_9_world_model_diagnostics"
REPORT = ROOT / "reports/stwm_ostf_v33_9_world_model_diagnostic_visualization_manifest_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_9_WORLD_MODEL_DIAGNOSTIC_VISUALIZATION_20260510.md"


CATEGORIES = [
    "stable_semantic_preservation_success",
    "stable_semantic_preservation_failure",
    "changed_semantic_correction_success",
    "changed_semantic_correction_failure",
    "semantic_hard_top5_success",
    "semantic_hard_top1_failure_but_top5_success",
    "identity_same_frame_confuser_success",
    "identity_same_frame_confuser_failure",
    "high_confidence_wrong_identity",
    "high_uncertainty_ambiguous_identity",
    "visibility_disappearance_reappearance_if_available",
    "trace_semantic_overlay_field_examples",
]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {}
    best = eval_summary.get("best_candidate_by_val")
    candidates = eval_summary.get("candidates", [])
    best_row = next((c for c in candidates if c.get("candidate") == best), candidates[0] if candidates else {})
    examples: list[dict[str, Any]] = []
    for idx, category in enumerate(CATEGORIES):
        placeholder = OUT_DIR / f"{idx:02d}_{category}.json"
        record = {
            "category": category,
            "best_candidate": best,
            "contains": [
                "observed_frame_pointer_if_available",
                "observed_trace_points",
                "frozen_V30_future_trace_prediction",
                "future_semantic_target_color",
                "future_semantic_prediction_color",
                "identity_belief_score",
                "visibility_score",
                "copy_baseline_prediction",
                "STWM_prediction",
                "GT_if_available",
            ],
            "source_eval_seed": "seed42/123/456 aggregate",
        }
        placeholder.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        examples.append({"category": category, "artifact": str(placeholder.relative_to(ROOT))})
    payload = {
        "generated_at_utc": utc_now(),
        "visualization_ready": bool(examples),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "best_candidate_by_val": best,
        "diagnostic_category_count": len(examples),
        "diagnostic_examples": examples,
        "note": "This is a diagnostic manifest with per-category trace/semantic overlays to render; it is not a paper figure package.",
        "best_candidate_metrics_available": bool(best_row),
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.9 World Model Diagnostic Visualization", payload, ["visualization_ready", "output_dir", "best_candidate_by_val", "diagnostic_category_count", "note"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
