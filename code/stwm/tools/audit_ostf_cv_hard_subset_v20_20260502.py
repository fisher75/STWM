#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, load_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import analytic_constant_velocity_predict, build_v18_rows
from stwm.tools.ostf_v20_common_20260502 import (
    bootstrap_delta,
    context_features_for_sample,
    evaluate_subset_metrics,
    hard_subset_flags,
    item_map,
    sample_key,
)


def _collect_records(combo: str) -> dict[str, Any]:
    rows, proto_centers = build_v18_rows(combo, seed=42)
    samples = rows["test"]
    pred_points, pred_vis, pred_sem = analytic_constant_velocity_predict(samples, proto_count=32, proto_centers=proto_centers, semantic_mode="observed_memory")
    ctx = []
    for s in samples:
        feats = context_features_for_sample(s)
        ctx.append(
            {
                "item_key": s.item_key,
                "object_id": int(s.object_id),
                "dataset": s.dataset,
                "split": s.split,
                **{k: float(v) if not isinstance(v, np.ndarray) else v for k, v in feats.items() if k in {
                    "cv_point_l1_proxy",
                    "cv_endpoint_proxy",
                    "curvature_proxy",
                    "occlusion_ratio",
                    "reappearance_flag",
                    "interaction_proxy",
                    "global_motion_proxy",
                }},
            }
        )
    flags = hard_subset_flags(ctx)
    for name, arr in flags.items():
        for i, flag in enumerate(arr.tolist()):
            ctx[i][name] = bool(flag)
    metrics_all = evaluate_subset_metrics(samples, pred_points, pred_vis, pred_sem)
    per_dataset = {}
    for ds in sorted({s.dataset for s in samples}):
        mask = np.asarray([s.dataset == ds for s in samples], dtype=bool)
        per_dataset[ds] = evaluate_subset_metrics(samples, pred_points, pred_vis, pred_sem, mask)
    return {
        "samples": samples,
        "proto_centers": proto_centers,
        "cv_pred_points": pred_points,
        "cv_pred_vis": pred_vis,
        "cv_pred_sem": pred_sem,
        "context_records": ctx,
        "all_metrics": metrics_all,
        "per_dataset": per_dataset,
        "flags": flags,
    }


def _subset_compare(report: dict[str, Any], keys: set[tuple[str, int]]) -> dict[str, Any]:
    mp = item_map(report)
    vals = [mp[k]["point_l1_px"] for k in sorted(keys) if k in mp and mp[k].get("point_l1_px") is not None]
    end = [mp[k]["endpoint_error_px"] for k in sorted(keys) if k in mp and mp[k].get("endpoint_error_px") is not None]
    ext = [mp[k]["extent_iou"] for k in sorted(keys) if k in mp and mp[k].get("extent_iou") is not None]
    return {
        "item_count": len(vals),
        "point_l1_mean": float(np.mean(vals)) if vals else None,
        "endpoint_mean": float(np.mean(end)) if end else None,
        "extent_iou_mean": float(np.mean(ext)) if ext else None,
    }


def main() -> int:
    m128 = _collect_records("M128_H8")
    m512 = _collect_records("M512_H8")

    v18_m128 = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m128_seed42_h8.json")
    v18_m512 = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m512_seed42_h8.json")
    v19_m128 = load_json(ROOT / "reports/stwm_ostf_v19_runs/v19_refinement_m128_seed42_h8.json")
    v19_m512 = load_json(ROOT / "reports/stwm_ostf_v19_runs/v19_refinement_m512_seed42_h8.json")
    m1_m128 = load_json(ROOT / "reports/stwm_ostf_v17_runs/m1_anchor_stwm_m128_seed42_h8.json")
    m1_m512 = load_json(ROOT / "reports/stwm_ostf_v17_runs/m1_anchor_stwm_seed42_h8.json")

    def subset_keys(bundle: dict[str, Any], flag_name: str) -> set[tuple[str, int]]:
        return {
            (r["item_key"], int(r["object_id"]))
            for r in bundle["context_records"]
            if r.get(flag_name)
        }

    top10_m128 = subset_keys(m128, "top10_cv_hard")
    top20_m128 = subset_keys(m128, "top20_cv_hard")
    top30_m128 = subset_keys(m128, "top30_cv_hard")
    top10_m512 = subset_keys(m512, "top10_cv_hard")
    top20_m512 = subset_keys(m512, "top20_cv_hard")
    top30_m512 = subset_keys(m512, "top30_cv_hard")
    nonlin_m512 = subset_keys(m512, "nonlinear_hard")
    occ_m512 = subset_keys(m512, "occlusion_hard")
    inter_m512 = subset_keys(m512, "interaction_hard")

    cv_easy_mean = float(np.mean([r["cv_point_l1_proxy"] for r in m512["context_records"] if not r["top20_cv_hard"]]))
    cv_hard_mean = float(np.mean([r["cv_point_l1_proxy"] for r in m512["context_records"] if r["top20_cv_hard"]]))
    all_mean = float(np.mean([r["cv_point_l1_proxy"] for r in m512["context_records"]]))
    share_top20 = sum(r["cv_point_l1_proxy"] for r in m512["context_records"] if r["top20_cv_hard"]) / max(
        1e-6, sum(r["cv_point_l1_proxy"] for r in m512["context_records"])
    )

    payload = {
        "audit_name": "stwm_ostf_cv_hard_subset_audit_v20",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "M128_H8": {
            "all_metrics": m128["all_metrics"],
            "per_dataset": m128["per_dataset"],
            "top10_cv_hard_count": len(top10_m128),
            "top20_cv_hard_count": len(top20_m128),
            "top30_cv_hard_count": len(top30_m128),
            "models_on_top20_cv_hard": {
                "m1_anchor": _subset_compare(m1_m128, top20_m128),
                "v18_m128": _subset_compare(v18_m128, top20_m128),
                "v19_m128": _subset_compare(v19_m128, top20_m128),
            },
        },
        "M512_H8": {
            "all_metrics": m512["all_metrics"],
            "per_dataset": m512["per_dataset"],
            "top10_cv_hard_count": len(top10_m512),
            "top20_cv_hard_count": len(top20_m512),
            "top30_cv_hard_count": len(top30_m512),
            "nonlinear_count": len(nonlin_m512),
            "occlusion_count": len(occ_m512),
            "interaction_count": len(inter_m512),
            "models_on_top20_cv_hard": {
                "m1_anchor": _subset_compare(m1_m512, top20_m512),
                "v18_m512": _subset_compare(v18_m512, top20_m512),
                "v19_m512": _subset_compare(v19_m512, top20_m512),
            },
            "models_on_nonlinear_subset": {
                "m1_anchor": _subset_compare(m1_m512, nonlin_m512),
                "v18_m512": _subset_compare(v18_m512, nonlin_m512),
                "v19_m512": _subset_compare(v19_m512, nonlin_m512),
            },
            "models_on_occlusion_subset": {
                "m1_anchor": _subset_compare(m1_m512, occ_m512),
                "v18_m512": _subset_compare(v18_m512, occ_m512),
                "v19_m512": _subset_compare(v19_m512, occ_m512),
            },
            "models_on_interaction_subset": {
                "m1_anchor": _subset_compare(m1_m512, inter_m512),
                "v18_m512": _subset_compare(v18_m512, inter_m512),
                "v19_m512": _subset_compare(v19_m512, inter_m512),
            },
        },
        "cv_distribution": {
            "all_mean_point_l1": all_mean,
            "easy80_mean_point_l1": cv_easy_mean,
            "hard20_mean_point_l1": cv_hard_mean,
            "hard20_error_share": float(share_top20),
            "cv_point_l1_percentiles": {
                "p50": float(np.percentile([r["cv_point_l1_proxy"] for r in m512["context_records"]], 50)),
                "p75": float(np.percentile([r["cv_point_l1_proxy"] for r in m512["context_records"]], 75)),
                "p90": float(np.percentile([r["cv_point_l1_proxy"] for r in m512["context_records"]], 90)),
                "p95": float(np.percentile([r["cv_point_l1_proxy"] for r in m512["context_records"]], 95)),
            },
        },
        "cv_saturation_detected": bool(cv_hard_mean > cv_easy_mean * 1.8 and share_top20 > 0.35),
        "cv_hard_subset_available": bool(len(top20_m512) >= 50),
        "object_dense_value_visible_on_hard_subset": bool(
            _subset_compare(v18_m128, top20_m128)["point_l1_mean"] is not None
            and _subset_compare(v18_m128, top20_m128)["point_l1_mean"] < _subset_compare(m1_m128, top20_m128)["point_l1_mean"]
        ),
        "next_model_needs_context_features": True,
        "notes": {
            "semantic_target_status": "OSTF semantic prototype target remains mostly static; hard-subset decision is driven by point/visibility/extent behavior, not changed/stable semantic evidence.",
            "all_average_metric_dominated_by_cv_easy_clips": bool(share_top20 < 0.50 and cv_hard_mean > cv_easy_mean * 1.8),
        },
    }
    out = ROOT / "reports/stwm_ostf_cv_hard_subset_audit_v20_20260502.json"
    dump_json(out, payload)
    write_doc(
        ROOT / "docs/STWM_OSTF_CV_HARD_SUBSET_AUDIT_V20_20260502.md",
        "STWM OSTF CV Hard Subset Audit V20",
        payload,
        [
            "cv_saturation_detected",
            "cv_hard_subset_available",
            "object_dense_value_visible_on_hard_subset",
            "next_model_needs_context_features",
            "cv_distribution",
        ],
    )
    # Save subset keys for downstream training/eval.
    subset_path = ROOT / "reports/stwm_ostf_cv_hard_subset_keys_v20_20260502.json"
    dump_json(
        subset_path,
        {
            "M128_H8_top20_cv_hard": sorted([list(x) for x in top20_m128]),
            "M512_H8_top20_cv_hard": sorted([list(x) for x in top20_m512]),
            "M512_H8_nonlinear_hard": sorted([list(x) for x in nonlin_m512]),
            "M512_H8_occlusion_hard": sorted([list(x) for x in occ_m512]),
            "M512_H8_interaction_hard": sorted([list(x) for x in inter_m512]),
        },
    )
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
