#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, bootstrap_delta, dump_json, load_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import (
    analytic_affine_motion_predict,
    analytic_constant_velocity_predict,
    build_v18_rows,
    eval_metrics_extended,
    item_scores_from_predictions,
)


def _load_runs() -> dict[str, dict[str, Any]]:
    run_dir = ROOT / "reports/stwm_ostf_v19_runs"
    out = {}
    for path in sorted(run_dir.glob("*.json")):
        r = load_json(path)
        if r and not str(r.get("experiment_name", "")).startswith("smoke_"):
            out[r["experiment_name"]] = r
    return out


def _item_map(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for row in report.get("item_scores", []):
        out[(str(row["item_key"]), int(row["object_id"]))] = row
    return out


def _pair_bootstrap(a: dict[str, Any], b: dict[str, Any], metric: str, higher_better: bool) -> dict[str, Any]:
    ma = _item_map(a)
    mb = _item_map(b)
    keys = sorted(set(ma) & set(mb))
    keys = [k for k in keys if metric in ma[k] and metric in mb[k] and ma[k][metric] is not None and mb[k][metric] is not None]
    if not keys:
        return {
            "item_count": 0,
            "mean_delta": None,
            "ci95": [None, None],
            "zero_excluded": False,
            "metric_available_in_both_reports": False,
        }
    av = np.asarray([ma[k][metric] for k in keys], dtype=float)
    bv = np.asarray([mb[k][metric] for k in keys], dtype=float)
    if not higher_better:
        av = -av
        bv = -bv
    out = bootstrap_delta(av, bv)
    out["metric_available_in_both_reports"] = True
    return out


def _beats_cv(row: dict[str, Any], cv: dict[str, Any], point_boot: dict[str, Any], endpoint_boot: dict[str, Any], extent_tolerance: float = 0.03) -> bool:
    m = row["test_metrics"]
    c = cv["test_metrics"]
    better_point = bool(point_boot["zero_excluded"] and (point_boot["mean_delta"] or 0.0) > 0.0)
    better_endpoint = bool(endpoint_boot["zero_excluded"] and (endpoint_boot["mean_delta"] or 0.0) > 0.0)
    better_pck = float(m["PCK_16px"]) > float(c["PCK_16px"]) + 0.01 or float(m["PCK_32px"]) > float(c["PCK_32px"]) + 0.01
    extent_ok = float(m["object_extent_iou"]) >= float(c["object_extent_iou"]) - extent_tolerance
    return bool((better_point or better_endpoint or better_pck) and extent_ok)


def _analytic_report(combo: str, kind: str) -> dict[str, Any]:
    rows, proto_centers = build_v18_rows(combo, seed=42)
    samples = rows["test"]
    fn = analytic_constant_velocity_predict if kind == "constant_velocity_copy" else analytic_affine_motion_predict
    pred_points, pred_vis, pred_sem = fn(samples, proto_count=32, proto_centers=proto_centers, semantic_mode="observed_memory")
    metrics = eval_metrics_extended(
        pred_points=pred_points,
        pred_vis_logits=pred_vis,
        pred_proto_logits=pred_sem,
        gt_points=np.stack([s.fut_points for s in samples]),
        gt_vis=np.stack([s.fut_vis for s in samples]),
        gt_anchor=np.stack([s.anchor_fut for s in samples]),
        proto_target=np.asarray([s.proto_target for s in samples], dtype=np.int64),
    )
    items = item_scores_from_predictions(samples, pred_points, pred_vis, pred_sem)
    return {
        "experiment_name": f"{kind}_{combo.lower()}",
        "source_combo": combo,
        "test_metrics": metrics,
        "item_scores": items,
    }


def main() -> int:
    runs = _load_runs()
    fair = load_json(ROOT / "reports/stwm_ostf_semantic_fair_eval_v19_20260502.json")
    v18_m128 = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m128_seed42_h8.json")
    v18_m512 = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m512_seed42_h8.json")
    point_transformer = load_json(ROOT / "reports/stwm_ostf_v17_runs/point_transformer_dense_seed42_h8.json")
    cv_m128 = _analytic_report("M128_H8", "constant_velocity_copy")
    cv_m512 = _analytic_report("M512_H8", "constant_velocity_copy")
    affine_m512 = _analytic_report("M512_H8", "affine_motion_prior_only")

    fair_m128 = fair["combo_results"]["M128_H8"]
    fair_m512 = fair["combo_results"]["M512_H8"]
    cv_m128["test_metrics"]["corrected_semantic_top1"] = fair_m128["constant_velocity_copy"]["corrected_semantic_top1"]
    cv_m128["test_metrics"]["corrected_semantic_top5"] = fair_m128["constant_velocity_copy"]["corrected_semantic_top5"]
    cv_m512["test_metrics"]["corrected_semantic_top1"] = fair_m512["constant_velocity_copy"]["corrected_semantic_top1"]
    cv_m512["test_metrics"]["corrected_semantic_top5"] = fair_m512["constant_velocity_copy"]["corrected_semantic_top5"]
    affine_m512["test_metrics"]["corrected_semantic_top1"] = fair_m512["affine_motion_prior_only"]["corrected_semantic_top1"]
    affine_m512["test_metrics"]["corrected_semantic_top5"] = fair_m512["affine_motion_prior_only"]["corrected_semantic_top5"]

    train_summary = {
        "audit_name": "stwm_ostf_v19_train_summary",
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
    dump_json(ROOT / "reports/stwm_ostf_v19_train_summary_20260502.json", train_summary)

    eval_summary = {
        "audit_name": "stwm_ostf_v19_eval_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corrected_semantic_metrics_from": "reports/stwm_ostf_semantic_fair_eval_v19_20260502.json",
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "test_metrics": r["test_metrics"],
                "test_metrics_by_dataset": r.get("test_metrics_by_dataset", {}),
                "val_metrics": r["val_metrics"],
            }
            for name, r in runs.items()
        },
        "imported_rows": {
            "constant_velocity_copy_M128": cv_m128["test_metrics"],
            "constant_velocity_copy_M512": cv_m512["test_metrics"],
            "affine_motion_prior_only_M512": affine_m512["test_metrics"],
            "v18_m128": v18_m128["test_metrics"],
            "v18_m512": v18_m512["test_metrics"],
            "point_transformer_dense": point_transformer["test_metrics"],
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v19_eval_summary_20260502.json", eval_summary)

    m128 = runs["v19_refinement_m128_seed42_h8"]
    m512 = runs["v19_refinement_m512_seed42_h8"]
    wo_ref = runs["v19_refinement_m512_wo_refinement_transformer_seed42_h8"]
    wo_scale = runs["v19_refinement_m512_wo_learnable_residual_scale_seed42_h8"]
    wo_dense = runs["v19_refinement_m512_wo_dense_points_seed42_h8"]

    bootstrap = {
        "audit_name": "stwm_ostf_v19_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v19_m128_vs_cv_point_L1": _pair_bootstrap(m128, cv_m128, "point_l1_px", higher_better=False),
        "v19_m128_vs_cv_endpoint": _pair_bootstrap(m128, cv_m128, "endpoint_error_px", higher_better=False),
        "v19_m512_vs_cv_point_L1": _pair_bootstrap(m512, cv_m512, "point_l1_px", higher_better=False),
        "v19_m512_vs_cv_endpoint": _pair_bootstrap(m512, cv_m512, "endpoint_error_px", higher_better=False),
        "v19_m512_vs_v18_m512_point_L1": _pair_bootstrap(m512, v18_m512, "point_l1_px", higher_better=False),
        "v19_m512_vs_v18_m512_endpoint": _pair_bootstrap(m512, v18_m512, "endpoint_error_px", higher_better=False),
        "v19_m512_vs_point_transformer_point_L1": _pair_bootstrap(m512, point_transformer, "point_l1_px", higher_better=False),
        "v19_m512_vs_wo_refinement_point_L1": _pair_bootstrap(m512, wo_ref, "point_l1_px", higher_better=False),
        "v19_m512_vs_wo_dense_point_L1": _pair_bootstrap(m512, wo_dense, "point_l1_px", higher_better=False),
    }
    dump_json(ROOT / "reports/stwm_ostf_v19_bootstrap_20260502.json", bootstrap)

    v19_m128_beats_cv = _beats_cv(m128, cv_m128, bootstrap["v19_m128_vs_cv_point_L1"], bootstrap["v19_m128_vs_cv_endpoint"])
    v19_m512_beats_cv = _beats_cv(m512, cv_m512, bootstrap["v19_m512_vs_cv_point_L1"], bootstrap["v19_m512_vs_cv_endpoint"])
    v19_m512_beats_v18 = bool(
        bootstrap["v19_m512_vs_v18_m512_point_L1"]["zero_excluded"]
        and (bootstrap["v19_m512_vs_v18_m512_point_L1"]["mean_delta"] or 0.0) > 0.0
        and bootstrap["v19_m512_vs_v18_m512_endpoint"]["zero_excluded"]
        and (bootstrap["v19_m512_vs_v18_m512_endpoint"]["mean_delta"] or 0.0) > 0.0
    )
    refinement_load_bearing = bool(
        bootstrap["v19_m512_vs_wo_refinement_point_L1"]["zero_excluded"]
        and (bootstrap["v19_m512_vs_wo_refinement_point_L1"]["mean_delta"] or 0.0) > 0.0
    )
    dense_points_load_bearing = bool(
        bootstrap["v19_m512_vs_wo_dense_point_L1"]["zero_excluded"]
        and (bootstrap["v19_m512_vs_wo_dense_point_L1"]["mean_delta"] or 0.0) > 0.0
    )
    semantic_oracle_leakage_fixed = bool(fair.get("semantic_oracle_leakage_fixed"))
    object_dense_claim_allowed = bool(
        semantic_oracle_leakage_fixed
        and v19_m512_beats_cv
        and v19_m512_beats_v18
        and refinement_load_bearing
        and dense_points_load_bearing
    )
    point_transformer_tradeoff = bool(
        (
            float(m512["test_metrics"]["corrected_semantic_top5"] if "corrected_semantic_top5" in m512["test_metrics"] else m512["test_metrics"].get("semantic_top5", 0.0))
            >= float(point_transformer["test_metrics"].get("semantic_top5", 0.0)) - 0.05
        )
        or (
            bootstrap["v19_m512_vs_point_transformer_point_L1"]["zero_excluded"]
            and (bootstrap["v19_m512_vs_point_transformer_point_L1"]["mean_delta"] or 0.0) > 0.0
            and float(m512["test_metrics"]["object_extent_iou"]) > float(point_transformer["test_metrics"]["object_extent_iou"]) + 0.05
        )
    )

    decision = {
        "audit_name": "stwm_ostf_v19_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "semantic_oracle_leakage_fixed": semantic_oracle_leakage_fixed,
        "V19_M128_beats_CV": v19_m128_beats_cv,
        "V19_M512_beats_CV": v19_m512_beats_cv,
        "V19_M512_beats_V18_M512": v19_m512_beats_v18,
        "refinement_load_bearing": refinement_load_bearing,
        "dense_points_load_bearing": dense_points_load_bearing,
        "V19_beats_point_transformer_or_tradeoff": point_transformer_tradeoff,
        "object_dense_semantic_trace_field_claim_allowed": object_dense_claim_allowed,
        "next_step_choice": (
            "run_multiseed_H16_scaling"
            if object_dense_claim_allowed
            else ("fallback_to_M128_object_dense" if v19_m128_beats_cv and not v19_m512_beats_cv else "improve_refinement_again")
        ),
        "rows_used": {
            "constant_velocity_copy": "imported_v18::constant_velocity_copy_seed42_h8",
            "affine_motion_prior_only": "analytic_recomputed::M512_H8",
            "v18_m128": "imported_v18::v18_physics_residual_m128_seed42_h8",
            "v18_m512": "imported_v18::v18_physics_residual_m512_seed42_h8",
            "v19_m128": "v19_refinement_m128_seed42_h8",
            "v19_m512": "v19_refinement_m512_seed42_h8",
            "v19_wo_refinement": "v19_refinement_m512_wo_refinement_transformer_seed42_h8",
            "v19_wo_learnable_residual_scale": "v19_refinement_m512_wo_learnable_residual_scale_seed42_h8",
            "v19_wo_dense_points": "v19_refinement_m512_wo_dense_points_seed42_h8",
            "point_transformer_dense": "imported_v17::point_transformer_dense_seed42_h8",
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v19_decision_20260502.json", decision)
    write_doc(
        ROOT / "docs/STWM_OSTF_V19_RESULTS_20260502.md",
        "STWM OSTF V19 Results",
        {**decision, "run_count": len(runs)},
        [
            "run_count",
            "semantic_oracle_leakage_fixed",
            "V19_M128_beats_CV",
            "V19_M512_beats_CV",
            "V19_M512_beats_V18_M512",
            "refinement_load_bearing",
            "dense_points_load_bearing",
            "V19_beats_point_transformer_or_tradeoff",
            "object_dense_semantic_trace_field_claim_allowed",
            "next_step_choice",
        ],
    )
    print("reports/stwm_ostf_v19_eval_summary_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
