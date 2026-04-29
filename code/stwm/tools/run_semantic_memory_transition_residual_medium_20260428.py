#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import run_for_c, write_doc, write_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prototype-count", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples-per-dataset", type=int, default=256)
    p.add_argument("--item-count", type=int, default=128)
    p.add_argument("--train-count", type=int, default=96)
    p.add_argument("--device", default="cuda")
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-cache-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    p.add_argument("--future-cache-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    p.add_argument("--summary-output", default="reports/stwm_semantic_memory_transition_residual_v1_medium_summary_20260428.json")
    p.add_argument("--eval-output", default="reports/stwm_semantic_memory_transition_residual_v1_medium_eval_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_MEMORY_TRANSITION_RESIDUAL_V1_MEDIUM_EVAL.md")
    args = p.parse_args()
    c = int(args.prototype_count)
    tmp_output = Path(args.eval_output)
    result = run_for_c(args, c, tmp_output)
    summary = {
        "audit_name": "stwm_semantic_memory_transition_residual_v1_medium_summary",
        "prototype_count": c,
        "steps": int(args.steps),
        "item_count": int(args.item_count),
        "train_count": int(args.train_count),
        "semantic_branch_trainable": True,
        "stage1_frozen": True,
        "dynamic_trace_path_frozen": True,
        "no_candidate_scorer": True,
        "future_candidate_leakage": False,
        "residual_top5_overall": float(result.get("residual_top5_overall", 0.0)),
        "residual_top5_stable": float(result.get("residual_top5_stable", 0.0)),
        "residual_top5_changed": float(result.get("residual_top5_changed", 0.0)),
        "copy_baseline_top5_overall": float(result.get("copy_baseline_top5_overall", 0.0)),
        "copy_baseline_top5_stable": float(result.get("copy_baseline_top5_stable", 0.0)),
        "copy_baseline_top5_changed": float(result.get("copy_baseline_top5_changed", 0.0)),
        "gain_over_copy_overall": float(result.get("gain_over_copy_overall", 0.0)),
        "gain_over_copy_changed_subset": float(result.get("gain_over_copy_changed_subset", 0.0)),
        "stable_copy_preserved": bool(result.get("stable_copy_preserved", False)),
        "semantic_memory_signal_positive": bool(result.get("semantic_memory_signal_positive", False)),
        "trace_regression_detected": bool(result.get("trace_regression_detected", False)),
    }
    write_json(Path(args.summary_output), summary)
    write_doc(
        Path(args.doc),
        "STWM Semantic Memory Transition Residual V1 Medium Eval",
        {"summary": summary, "eval": result},
        bullets=[
            "Medium sanity remains candidate-free and keeps Stage1/dynamic trace path frozen.",
            "Future prototype targets are supervision only.",
        ],
    )


if __name__ == "__main__":
    main()
