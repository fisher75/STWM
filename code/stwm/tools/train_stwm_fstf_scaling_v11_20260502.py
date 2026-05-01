#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from stwm.tools.train_fstf_strong_copyaware_baseline_v8_20260501 import main as v8_train_main


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scaling-axis", required=True)
    p.add_argument("--scaling-value", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--prototype-count", type=int, default=32)
    p.add_argument("--model-size", default="small")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--residual-scale", type=float, default=0.25)
    p.add_argument("--train-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json")
    p.add_argument("--val-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json")
    p.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json")
    p.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    p.add_argument("--checkpoint-output", required=True)
    p.add_argument("--summary-output", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Reuse the frozen STWM-FSTF same-output copy-gated residual training protocol,
    # while tagging outputs as V11 scaling artifacts for aggregation.
    argv = [
        "train_fstf_strong_copyaware_baseline_v8_20260501.py",
        "--baseline",
        "copy_gated_residual_plain_trace_semantic",
        "--prototype-count",
        str(args.prototype_count),
        "--seed",
        str(args.seed),
        "--train-cache-report",
        args.train_cache_report,
        "--val-cache-report",
        args.val_cache_report,
        "--observed-report",
        args.observed_report,
        "--future-cache-report",
        args.future_cache_report,
        "--steps",
        str(args.steps),
        "--lr",
        str(args.lr),
        "--d-model",
        str(args.d_model),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--residual-scale",
        str(args.residual_scale),
        "--device",
        args.device,
        "--checkpoint-output",
        args.checkpoint_output,
        "--summary-output",
        args.summary_output,
        "--progress-every",
        "250",
    ]
    sys.argv = argv
    v8_train_main()
    out = Path(args.summary_output)
    if out.exists():
        payload: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        payload.update(
            {
                "audit_name": "stwm_fstf_scaling_v11_train",
                "scaling_axis": args.scaling_axis,
                "scaling_value": args.scaling_value,
                "model_size": args.model_size,
                "semantic_transition_backend": "materialized_cache_copy_gated_residual_plain_trace_semantic",
                "raw_video_end_to_end_training": False,
                "frozen_video_derived_trace_semantic_cache": True,
                "candidate_scorer_used": False,
                "future_candidate_leakage": False,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
        )
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
