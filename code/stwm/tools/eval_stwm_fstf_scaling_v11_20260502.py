#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from stwm.tools.eval_fstf_strong_copyaware_baseline_v8_20260501 import main as v8_eval_main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scaling-axis", required=True)
    p.add_argument("--scaling-value", required=True)
    p.add_argument("--model-size", default="small")
    p.add_argument("--prototype-count", type=int, default=32)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json")
    p.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json")
    p.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    sys.argv = [
        "eval_fstf_strong_copyaware_baseline_v8_20260501.py",
        "--baseline",
        "copy_gated_residual_plain_trace_semantic",
        "--checkpoint",
        args.checkpoint,
        "--test-cache-report",
        args.test_cache_report,
        "--observed-report",
        args.observed_report,
        "--future-cache-report",
        args.future_cache_report,
        "--device",
        args.device,
        "--output",
        args.output,
    ]
    v8_eval_main()
    out = Path(args.output)
    if out.exists():
        payload: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        payload.update(
            {
                "audit_name": "stwm_fstf_scaling_v11_eval",
                "scaling_axis": args.scaling_axis,
                "scaling_value": args.scaling_value,
                "model_size": args.model_size,
                "prototype_count": int(args.prototype_count),
                "semantic_transition_backend": "materialized_cache_copy_gated_residual_plain_trace_semantic",
                "raw_video_end_to_end_training": False,
                "frozen_video_derived_trace_semantic_cache": True,
                "free_rollout_path": True,
                "candidate_scorer_used": False,
                "future_candidate_leakage": False,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
        )
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
