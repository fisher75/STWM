#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, load_json, write_doc


def main() -> int:
    eval_summary = load_json(ROOT / "reports/stwm_ostf_v17_eval_summary_20260502.json")
    bootstrap = load_json(ROOT / "reports/stwm_ostf_v17_bootstrap_20260502.json")
    exps = eval_summary["experiments"]
    cv = exps["constant_velocity_copy_seed42_h8"]["test_metrics"]
    pt = exps["point_transformer_dense_seed42_h8"]["test_metrics"]
    m512 = exps["ostf_multitrace_m512_seed42_h8"]["test_metrics"]
    m128 = exps["ostf_multitrace_m128_seed42_h8"]["test_metrics"]
    no_dense = exps["ostf_m512_wo_dense_point_input_seed42_h8"]["test_metrics"]
    no_sem = exps["ostf_m512_wo_semantic_memory_seed42_h8"]["test_metrics"]
    no_res = exps["ostf_m512_wo_point_residual_decoder_seed42_h8"]["test_metrics"]

    point_gap = float(m512["point_L1_px"] - cv["point_L1_px"])
    extent_gap = float(cv["object_extent_iou"] - m512["object_extent_iou"])
    semantic_gap = float(pt["semantic_top5"] - m512["semantic_top5"])

    payload = {
        "audit_name": "stwm_ostf_v17_failure_diagnosis",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "constant_velocity_strength": {
            "why_strong": [
                "teacher trajectories over H8 are near-linear enough that last-step velocity extrapolation is a very strong physical prior",
                "constant_velocity_copy preserves per-point identity and object extent directly from observed dense tracks",
                "V17 predicts dense future points from a pooled semantic trace-unit token without anchoring to a strong motion prior",
            ],
            "test_point_L1_px": cv["point_L1_px"],
            "test_endpoint_error_px": cv["endpoint_error_px"],
            "test_object_extent_iou": cv["object_extent_iou"],
        },
        "v17_vs_constant_velocity": {
            "m128_point_L1_gap_px": float(m128["point_L1_px"] - cv["point_L1_px"]),
            "m512_point_L1_gap_px": point_gap,
            "m512_endpoint_gap_px": float(m512["endpoint_error_px"] - cv["endpoint_error_px"]),
            "m512_extent_iou_gap": -extent_gap,
            "m512_pck16_gap": float(m512["PCK_16px"] - cv["PCK_16px"]),
            "m512_pck32_missing_in_v17": True,
        },
        "point_transformer_semantic_advantage": {
            "semantic_top5_gap_vs_m512": semantic_gap,
            "likely_reason": [
                "future semantic target is effectively constant per object prototype in the current cache",
                "point_transformer_dense can map observed semantic feature directly to prototype logits without needing a compressed semantic trace unit bottleneck",
                "V17 semantic branch is too weak relative to its trajectory decoder and underuses the direct semantic memory signal",
            ],
        },
        "dense_failure_hypotheses": {
            "point_loss_scale_issue": True,
            "visibility_loss_issue": bool(m512["visibility_F1"] > 0.9 and no_res["visibility_F1"] == 0.0),
            "capacity_or_decoder_issue": True,
            "anchor_plus_residual_decoder_insufficient": True,
            "evidence": {
                "no_dense_points_hurts_only_slightly": float(no_dense["point_L1_px"] - m512["point_L1_px"]),
                "no_semantic_memory_hurts_only_slightly": float(no_sem["point_L1_px"] - m512["point_L1_px"]),
                "no_point_residual_decoder_hurts_strongly": float(no_res["point_L1_px"] - m512["point_L1_px"]),
            },
        },
        "bootstrap_evidence": bootstrap,
        "v18_must_fix": [
            "inject a strong physics prior instead of predicting dense future points from scratch",
            "make residuals low-frequency and regularized so the model cannot drift far from a good physical prior",
            "improve direct semantic-memory-to-prototype readout",
            "add extent/shape loss so object cloud geometry matters, not only average point error",
            "separate anchor, dense-point, visibility, and semantic objectives more explicitly",
        ],
    }
    dump_json(ROOT / "reports/stwm_ostf_v17_failure_diagnosis_20260502.json", payload)
    write_doc(
        ROOT / "docs/STWM_OSTF_V17_FAILURE_DIAGNOSIS_20260502.md",
        "STWM OSTF V17 Failure Diagnosis",
        payload,
        [
            "constant_velocity_strength",
            "v17_vs_constant_velocity",
            "point_transformer_semantic_advantage",
            "dense_failure_hypotheses",
            "v18_must_fix",
        ],
    )
    print("reports/stwm_ostf_v17_failure_diagnosis_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
