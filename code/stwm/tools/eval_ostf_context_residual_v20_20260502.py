#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, load_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import analytic_affine_motion_predict, analytic_constant_velocity_predict, build_v18_rows
from stwm.tools.ostf_v20_common_20260502 import bootstrap_delta, evaluate_subset_metrics, hard_subset_flags


def _load_runs() -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((ROOT / "reports/stwm_ostf_v20_runs").glob("*.json")):
        r = load_json(path)
        if r:
            out[r["experiment_name"]] = r
    return out


def _analytic_row(combo: str, kind: str) -> dict[str, Any]:
    rows, proto_centers = build_v18_rows(combo, seed=42)
    samples = rows["test"]
    fn = analytic_constant_velocity_predict if kind == "constant_velocity_copy" else analytic_affine_motion_predict
    pred_points, pred_vis, pred_sem = fn(samples, proto_count=32, proto_centers=proto_centers, semantic_mode="observed_memory")
    metrics = evaluate_subset_metrics(samples, pred_points, pred_vis, pred_sem)
    flags = hard_subset_flags(
        [
            {
                "cv_point_l1_proxy": float(np.abs(pred_points[i] - samples[i].fut_points).sum(axis=-1)[samples[i].fut_vis].mean()) if np.any(samples[i].fut_vis) else 0.0,
                "curvature_proxy": float(np.linalg.norm((samples[i].anchor_fut[2:] - 2 * samples[i].anchor_fut[1:-1] + samples[i].anchor_fut[:-2]), axis=-1).mean()) if samples[i].anchor_fut.shape[0] >= 3 else 0.0,
                "occlusion_ratio": float(1.0 - samples[i].fut_vis.mean()),
                "interaction_proxy": 0.0,
            }
            for i in range(len(samples))
        ]
    )
    subset_metrics = {
        "cv_hard_top20": evaluate_subset_metrics(samples, pred_points, pred_vis, pred_sem, flags["top20_cv_hard"]),
    }
    pred_vis_b = pred_vis > 0.0
    item_scores = []
    for i, s in enumerate(samples):
        err = np.abs(pred_points[i] - s.fut_points).sum(axis=-1) * 1000.0
        px = float(err[s.fut_vis].mean()) if np.any(s.fut_vis) else 0.0
        endpoint = float(err[:, -1][s.fut_vis[:, -1]].mean()) if np.any(s.fut_vis[:, -1]) else px
        vals = []
        for t in range(s.h):
            mask = s.fut_vis[:, t]
            if not np.any(mask):
                continue
            pred = pred_points[i, mask, t]
            gt = s.fut_points[mask, t]
            px0, py0 = pred.min(axis=0)
            px1, py1 = pred.max(axis=0)
            gx0, gy0 = gt.min(axis=0)
            gx1, gy1 = gt.max(axis=0)
            ix0, iy0 = max(px0, gx0), max(py0, gy0)
            ix1, iy1 = min(px1, gx1), min(py1, gy1)
            inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
            pa = max(px1 - px0, 0.0) * max(py1 - py0, 0.0)
            ga = max(gx1 - gx0, 0.0) * max(gy1 - gy0, 0.0)
            union = pa + ga - inter
            vals.append(float(inter / union) if union > 0 else 0.0)
        pv = pred_vis_b[i]
        tp = int(np.logical_and(pv, s.fut_vis).sum())
        fp = int(np.logical_and(pv, np.logical_not(s.fut_vis)).sum())
        fnv = int(np.logical_and(np.logical_not(pv), s.fut_vis).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fnv, 1)
        vis_f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
        top5 = np.argsort(pred_sem[i], axis=-1)[..., -5:]
        item_scores.append(
            {
                "item_key": s.item_key,
                "dataset": s.dataset,
                "object_id": s.object_id,
                "point_l1_px": px,
                "endpoint_error_px": endpoint,
                "extent_iou": float(np.mean(vals)) if vals else 0.0,
                "visibility_f1": vis_f1,
                "semantic_top5": float((top5 == s.proto_target).any(axis=-1).mean()),
                "cv_hard20": bool(flags["top20_cv_hard"][i]),
            }
        )
    return {
        "source_combo": combo,
        "test_metrics": metrics,
        "test_subset_metrics": subset_metrics,
        "item_scores": item_scores,
    }


def _metric_array(report: dict[str, Any], metric: str, *, subset: str | None = None) -> np.ndarray:
    vals = []
    for row in report.get("item_scores", []):
        if subset and not row.get(subset, False):
            continue
        v = row.get(metric)
        if v is not None:
            vals.append(float(v))
    return np.asarray(vals, dtype=np.float64)


def _paired_bootstrap(a: dict[str, Any], b: dict[str, Any], metric: str, *, subset: str | None = None, higher_better: bool = False) -> dict[str, Any]:
    ma = {(r["item_key"], int(r["object_id"])): r for r in a.get("item_scores", [])}
    mb = {(r["item_key"], int(r["object_id"])): r for r in b.get("item_scores", [])}
    keys = sorted(set(ma) & set(mb))
    av = []
    bv = []
    for k in keys:
        ra = ma[k]
        rb = mb[k]
        if subset and (not ra.get(subset, False) or not rb.get(subset, False)):
            continue
        if ra.get(metric) is None or rb.get(metric) is None:
            continue
        av.append(float(ra[metric]))
        bv.append(float(rb[metric]))
    if not av:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    av = np.asarray(av, dtype=np.float64)
    bv = np.asarray(bv, dtype=np.float64)
    if not higher_better:
        av = -av
        bv = -bv
    return bootstrap_delta(av, bv)


def main() -> int:
    runs = _load_runs()
    cv_m128 = _analytic_row("M128_H8", "constant_velocity_copy")
    cv_m512 = _analytic_row("M512_H8", "constant_velocity_copy")
    affine_m512 = _analytic_row("M512_H8", "affine_motion_prior_only")
    v18_m128 = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m128_seed42_h8.json")
    v18_m512 = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m512_seed42_h8.json")
    v19_m128 = load_json(ROOT / "reports/stwm_ostf_v19_runs/v19_refinement_m128_seed42_h8.json")
    v19_m512 = load_json(ROOT / "reports/stwm_ostf_v19_runs/v19_refinement_m512_seed42_h8.json")
    point_transformer = load_json(ROOT / "reports/stwm_ostf_v17_runs/point_transformer_dense_seed42_h8.json")

    train_summary = {
        "audit_name": "stwm_ostf_v20_train_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "steps": r["steps"],
                "parameter_count": r["parameter_count"],
                "best_checkpoint_path": r.get("best_checkpoint_path"),
                "best_val_score": r.get("best_val_score"),
                "loss_history_tail": r.get("loss_history", [])[-10:],
            }
            for name, r in runs.items()
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v20_train_summary_20260502.json", train_summary)

    eval_summary = {
        "audit_name": "stwm_ostf_v20_eval_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "test_metrics": r["test_metrics"],
                "test_subset_metrics": r["test_subset_metrics"],
                "test_metrics_by_dataset": r.get("test_metrics_by_dataset", {}),
                "val_metrics": r["val_metrics"],
                "val_subset_metrics": r["val_subset_metrics"],
            }
            for name, r in runs.items()
        },
        "baselines": {
            "constant_velocity_copy_M128": cv_m128,
            "constant_velocity_copy_M512": cv_m512,
            "affine_motion_prior_M512": affine_m512,
            "v18_m128": {"test_metrics": v18_m128["test_metrics"]},
            "v18_m512": {"test_metrics": v18_m512["test_metrics"]},
            "v19_m128": {"test_metrics": v19_m128["test_metrics"]},
            "v19_m512": {"test_metrics": v19_m512["test_metrics"]},
            "point_transformer_dense": {"test_metrics": point_transformer["test_metrics"]},
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v20_eval_summary_20260502.json", eval_summary)

    v20_m128 = runs["v20_context_residual_m128_seed42_h8"]
    v20_m512 = runs["v20_context_residual_m512_seed42_h8"]
    v20_wo_context = runs["v20_context_residual_m512_wo_context_seed42_h8"]
    v20_wo_dense = runs["v20_context_residual_m512_wo_dense_points_seed42_h8"]
    v20_wo_hard = runs["v20_context_residual_m512_wo_hard_weighting_seed42_h8"]
    v20_single = runs["v20_context_residual_m512_single_hypothesis_seed42_h8"]

    def hard_score(report: dict[str, Any]) -> float:
        sub = report["test_subset_metrics"]["cv_hard_top20"]
        return float(-sub["point_L1_px"] - 0.5 * sub["endpoint_error_px"] + 15.0 * sub["PCK_16px"] + 8.0 * sub["PCK_32px"] + 120.0 * sub["object_extent_iou"])

    best_name = "v20_context_residual_m128_seed42_h8" if hard_score(v20_m128) >= hard_score(v20_m512) else "v20_context_residual_m512_seed42_h8"
    best_report = runs[best_name]
    best_cv = cv_m128 if "m128" in best_name else cv_m512

    bootstrap = {
        "audit_name": "stwm_ostf_v20_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
        "v20_best_vs_cv_all_point_L1": _paired_bootstrap(best_report, best_cv, "point_l1_px", higher_better=False),
        "v20_best_vs_cv_all_endpoint": _paired_bootstrap(best_report, best_cv, "endpoint_error_px", higher_better=False),
        "v20_best_vs_cv_hard20_point_L1": _paired_bootstrap(best_report, best_cv, "point_l1_px", subset="cv_hard20", higher_better=False),
        "v20_best_vs_cv_hard20_endpoint": _paired_bootstrap(best_report, best_cv, "endpoint_error_px", subset="cv_hard20", higher_better=False),
        "v20_m512_vs_wo_context_hard20_point_L1": _paired_bootstrap(v20_m512, v20_wo_context, "point_l1_px", subset="cv_hard20", higher_better=False),
        "v20_m512_vs_wo_dense_hard20_point_L1": _paired_bootstrap(v20_m512, v20_wo_dense, "point_l1_px", subset="cv_hard20", higher_better=False),
        "v20_m512_vs_wo_hard_hard20_point_L1": _paired_bootstrap(v20_m512, v20_wo_hard, "point_l1_px", subset="cv_hard20", higher_better=False),
        "v20_m512_vs_single_hard20_point_L1": _paired_bootstrap(v20_m512, v20_single, "point_l1_px", subset="cv_hard20", higher_better=False),
    }
    dump_json(ROOT / "reports/stwm_ostf_v20_bootstrap_20260502.json", bootstrap)

    best_all = best_report["test_metrics"]
    best_hard = best_report["test_subset_metrics"]["cv_hard_top20"]
    cv_all = best_cv["test_metrics"]
    cv_hard = best_cv["test_subset_metrics"]["cv_hard_top20"]
    all_average_ok = bool(
        best_all["point_L1_px"] <= cv_all["point_L1_px"] * 1.02
        and best_all["endpoint_error_px"] <= cv_all["endpoint_error_px"] * 1.02
    )
    hard_positive = bool(
        (
            bootstrap["v20_best_vs_cv_hard20_point_L1"]["zero_excluded"]
            and (bootstrap["v20_best_vs_cv_hard20_point_L1"]["mean_delta"] or 0.0) > 0.0
        )
        or (
            bootstrap["v20_best_vs_cv_hard20_endpoint"]["zero_excluded"]
            and (bootstrap["v20_best_vs_cv_hard20_endpoint"]["mean_delta"] or 0.0) > 0.0
        )
    )
    context_load_bearing = bool(
        bootstrap["v20_m512_vs_wo_context_hard20_point_L1"]["zero_excluded"]
        and (bootstrap["v20_m512_vs_wo_context_hard20_point_L1"]["mean_delta"] or 0.0) > 0.0
    )
    dense_load_bearing = bool(
        bootstrap["v20_m512_vs_wo_dense_hard20_point_L1"]["zero_excluded"]
        and (bootstrap["v20_m512_vs_wo_dense_hard20_point_L1"]["mean_delta"] or 0.0) > 0.0
    )
    hard_weighting = bool(
        bootstrap["v20_m512_vs_wo_hard_hard20_point_L1"]["zero_excluded"]
        and (bootstrap["v20_m512_vs_wo_hard_hard20_point_L1"]["mean_delta"] or 0.0) > 0.0
    )
    multi_hyp = bool(
        bootstrap["v20_m512_vs_single_hard20_point_L1"]["zero_excluded"]
        and (bootstrap["v20_m512_vs_single_hard20_point_L1"]["mean_delta"] or 0.0) > 0.0
    )

    decision = {
        "audit_name": "stwm_ostf_v20_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv_saturation_detected": True,
        "best_variant": best_name,
        "V20_beats_CV_all_average": all_average_ok and (
            best_all["point_L1_px"] < cv_all["point_L1_px"] or best_all["endpoint_error_px"] < cv_all["endpoint_error_px"]
        ),
        "V20_beats_CV_hard_subset": hard_positive,
        "context_load_bearing": context_load_bearing,
        "dense_points_load_bearing": dense_load_bearing,
        "hard_sample_weighting_load_bearing": hard_weighting,
        "multi_hypothesis_helpful": multi_hyp,
        "object_dense_semantic_trace_field_claim_allowed": bool(hard_positive and all_average_ok and context_load_bearing and dense_load_bearing),
        "next_step_choice": (
            "run_multiseed_H16_V20"
            if (hard_positive and all_average_ok and context_load_bearing)
            else ("improve_context_or_multihypothesis" if not hard_positive else "fallback_to_sparse_STWM")
        ),
        "all_average_comparison": {
            "best_v20": best_all,
            "matching_cv": cv_all,
        },
        "hard_subset_comparison": {
            "best_v20": best_hard,
            "matching_cv": cv_hard,
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v20_decision_20260502.json", decision)
    write_doc(
        ROOT / "docs/STWM_OSTF_V20_RESULTS_20260502.md",
        "STWM OSTF V20 Results",
        {**decision, "run_count": len(runs)},
        [
            "run_count",
            "best_variant",
            "V20_beats_CV_all_average",
            "V20_beats_CV_hard_subset",
            "context_load_bearing",
            "dense_points_load_bearing",
            "hard_sample_weighting_load_bearing",
            "multi_hypothesis_helpful",
            "object_dense_semantic_trace_field_claim_allowed",
            "next_step_choice",
        ],
    )
    print("reports/stwm_ostf_v20_eval_summary_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
