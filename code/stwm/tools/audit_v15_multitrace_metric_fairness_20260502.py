#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from stwm_v15b_forensic_utils_20260502 import (  # noqa: E402
    ROOT,
    dump_json,
    load_cache,
    load_json,
    now_utc,
    relative_point_unique_ratio,
    valid_object_pairs,
    write_doc,
)


def _coverage_stats(m: int, limit: int = 2048) -> dict[str, Any]:
    z = load_cache(m)
    rel = z["object_relative_xy"]
    pairs = valid_object_pairs(z, limit)
    unique = []
    area = []
    span_x = []
    span_y = []
    edge_frac = []
    for i, j in pairs:
        pts = np.asarray(rel[i, j], dtype=np.float32).reshape(-1, 2)
        unique.append(relative_point_unique_ratio(pts))
        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        span = np.maximum(mx - mn, 0)
        span_x.append(float(span[0]))
        span_y.append(float(span[1]))
        area.append(float(span[0] * span[1]))
        edge_frac.append(float(((pts < 0.05) | (pts > 0.95)).any(axis=1).mean()))
    return {
        "M": m,
        "sampled_object_count": len(pairs),
        "mean_unique_coordinate_ratio": float(np.mean(unique)) if unique else 0.0,
        "mean_relative_span_x": float(np.mean(span_x)) if span_x else 0.0,
        "mean_relative_span_y": float(np.mean(span_y)) if span_y else 0.0,
        "mean_relative_point_cloud_area": float(np.mean(area)) if area else 0.0,
        "mean_crop_edge_fraction_proxy": float(np.mean(edge_frac)) if edge_frac else 0.0,
    }


def _pilot_row(pilot: dict[str, Any], m: int) -> dict[str, Any]:
    row = pilot.get("runs", {}).get(f"M{m}_seed42", {})
    return {
        "point_l1": row.get("point_l1"),
        "endpoint_error": row.get("endpoint_error"),
        "pck_0_05": row.get("pck_0_05"),
        "constant_velocity_point_l1": row.get("constant_velocity_point_l1"),
        "constant_velocity_endpoint_error": row.get("constant_velocity_endpoint_error"),
        "constant_velocity_pck_0_05": row.get("constant_velocity_pck_0_05"),
        "semantic_changed_top5": row.get("semantic_changed_top5", "not_trained_in_phase1_pilot"),
        "stable_preservation": row.get("stable_preservation", "not_trained_in_phase1_pilot"),
        "visibility_F1": row.get("visibility_F1", "visibility_target_from_pseudo_valid_mask_not_claimed"),
    }


def main() -> int:
    pilot = load_json(ROOT / "reports/stwm_ostf_multitrace_pilot_v15_20260502.json")
    coverage = {f"M{m}": _coverage_stats(m) for m in [1, 128, 512]}
    metrics = {f"M{m}": _pilot_row(pilot, m) for m in [1, 128, 512]}
    m1_l1 = metrics["M1"].get("point_l1")
    m512_l1 = metrics["M512"].get("point_l1")
    m1_area = coverage["M1"]["mean_relative_point_cloud_area"]
    m512_area = coverage["M512"]["mean_relative_point_cloud_area"]
    payload: dict[str, Any] = {
        "audit_name": "stwm_v15_multitrace_metric_fairness",
        "generated_at_utc": now_utc(),
        "pilot_metrics": metrics,
        "point_cloud_coverage_stats": coverage,
        "anchor_centroid_L1": metrics["M1"],
        "internal_point_L1": {"M128": metrics["M128"], "M512": metrics["M512"]},
        "point_PCK": {
            "M1_pck_0_05": metrics["M1"].get("pck_0_05"),
            "M128_pck_0_05": metrics["M128"].get("pck_0_05"),
            "M512_pck_0_05": metrics["M512"].get("pck_0_05"),
        },
        "visibility_F1": "invalid_for_claim_pseudo_valid_mask_not_tracker_occlusion",
        "mask_or_object_coverage_by_point_cloud": coverage,
        "boundary_coverage": "not directly recoverable from V15 cache; only crop-edge proxy stored here",
        "semantic_changed_top5": "not_trained_in_phase1_pilot",
        "stable_preservation": "not_trained_in_phase1_pilot",
        "downstream_reacquisition": "not_evaluated_for_V15_OSTF_pilot",
        "M1_L1_vs_M512_point_L1_comparable": False,
        "M512_improves_coverage": bool(m512_area > m1_area and coverage["M512"]["mean_unique_coordinate_ratio"] > coverage["M1"]["mean_unique_coordinate_ratio"]),
        "M512_improves_anchor_or_extent": "improves_extent_coverage_not_anchor_metric",
        "current_failure_due_to_metric_mismatch": bool(m512_l1 is not None and m1_l1 is not None and m512_l1 > m1_l1),
        "interpretation": (
            "M1 predicts an object anchor/centroid-like target while M512 averages many internal pseudo-points. "
            "Those losses are not a fair final dense evidence comparison, especially without a physical teacher."
        ),
    }
    out = ROOT / "reports/stwm_v15_multitrace_metric_fairness_20260502.json"
    doc = ROOT / "docs/STWM_V15_MULTITRACE_METRIC_FAIRNESS_20260502.md"
    dump_json(out, payload)
    write_doc(
        doc,
        "STWM V15 MultiTrace Metric Fairness",
        payload,
        [
            "M1_L1_vs_M512_point_L1_comparable",
            "M512_improves_coverage",
            "M512_improves_anchor_or_extent",
            "current_failure_due_to_metric_mismatch",
            "visibility_F1",
            "interpretation",
        ],
    )
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
