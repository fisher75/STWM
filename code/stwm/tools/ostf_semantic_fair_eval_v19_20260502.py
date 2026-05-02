#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import (
    analytic_affine_motion_predict,
    analytic_constant_velocity_predict,
    build_v18_rows,
    eval_metrics_extended,
)


def _eval_combo(combo: str, seed: int = 42) -> dict[str, Any]:
    rows, proto_centers = build_v18_rows(combo, seed=seed)
    samples = rows["test"]
    gt_points = np.stack([s.fut_points for s in samples])
    gt_vis = np.stack([s.fut_vis for s in samples])
    gt_anchor = np.stack([s.anchor_fut for s in samples])
    proto_target = np.asarray([s.proto_target for s in samples], dtype=np.int64)

    def metrics(kind: str, corrected: bool) -> dict[str, Any]:
        fn = analytic_constant_velocity_predict if kind == "constant_velocity_copy" else analytic_affine_motion_predict
        pred_points, pred_vis, pred_sem = fn(
            samples,
            proto_count=32,
            proto_centers=proto_centers,
            semantic_mode="observed_memory" if corrected else "oracle",
        )
        return eval_metrics_extended(
            pred_points=pred_points,
            pred_vis_logits=pred_vis,
            pred_proto_logits=pred_sem,
            gt_points=gt_points,
            gt_vis=gt_vis,
            gt_anchor=gt_anchor,
            proto_target=proto_target,
        )

    return {
        "constant_velocity_copy": {
            "old_semantic_top1": metrics("constant_velocity_copy", False).get("semantic_top1"),
            "old_semantic_top5": metrics("constant_velocity_copy", False).get("semantic_top5"),
            "corrected_semantic_top1": metrics("constant_velocity_copy", True).get("semantic_top1"),
            "corrected_semantic_top5": metrics("constant_velocity_copy", True).get("semantic_top5"),
        },
        "affine_motion_prior_only": {
            "old_semantic_top1": metrics("affine_motion_prior_only", False).get("semantic_top1"),
            "old_semantic_top5": metrics("affine_motion_prior_only", False).get("semantic_top5"),
            "corrected_semantic_top1": metrics("affine_motion_prior_only", True).get("semantic_top1"),
            "corrected_semantic_top5": metrics("affine_motion_prior_only", True).get("semantic_top5"),
        },
    }


def main() -> int:
    payload = {
        "audit_name": "stwm_ostf_semantic_fair_eval_v19",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_oracle_leakage_fixed": True,
        "correction_rule": "Analytic baselines now derive semantic logits from observed semantic memory / nearest train prototype centers instead of proto_target one-hot.",
        "combo_results": {
            "M128_H8": _eval_combo("M128_H8"),
            "M512_H8": _eval_combo("M512_H8"),
        },
    }
    out = ROOT / "reports/stwm_ostf_semantic_fair_eval_v19_20260502.json"
    dump_json(out, payload)
    write_doc(
        ROOT / "docs/STWM_OSTF_SEMANTIC_FAIR_EVAL_V19_20260502.md",
        "STWM OSTF Semantic Fair Eval V19",
        payload,
        [
            "semantic_oracle_leakage_fixed",
            "correction_rule",
            "combo_results",
        ],
    )
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
