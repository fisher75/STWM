#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, load_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import bootstrap_delta


def _load_runs() -> dict[str, dict[str, Any]]:
    run_dir = ROOT / "reports/stwm_ostf_v18_runs"
    out = {}
    for path in sorted(run_dir.glob("*.json")):
        if path.stem.startswith("smoke_"):
            continue
        r = load_json(path)
        if str(r.get("experiment_name", "")).startswith("smoke_"):
            continue
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


def main() -> int:
    runs = _load_runs()
    v17_runs = {p["experiment_name"]: p for p in [load_json(path) for path in sorted((ROOT / "reports/stwm_ostf_v17_runs").glob("*seed42_h8.json"))]}
    train_summary = {
        "audit_name": "stwm_ostf_v18_train_summary",
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
                "loss_history": r.get("loss_history", []),
            }
            for name, r in runs.items()
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v18_train_summary_20260502.json", train_summary)

    eval_summary = {
        "audit_name": "stwm_ostf_v18_eval_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "imported_v17_baselines": {
            "point_transformer_dense_seed42_h8": {
                "test_metrics": v17_runs["point_transformer_dense_seed42_h8"]["test_metrics"],
                "source": "existing_v17_artifact_same_cache_protocol",
            },
            "ostf_multitrace_m512_seed42_h8": {
                "test_metrics": v17_runs["ostf_multitrace_m512_seed42_h8"]["test_metrics"],
                "source": "existing_v17_artifact_same_cache_protocol",
            },
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v18_eval_summary_20260502.json", eval_summary)

    full_m512 = runs["v18_physics_residual_m512_seed42_h8"]
    full_m128 = runs["v18_physics_residual_m128_seed42_h8"]
    cv = runs["constant_velocity_copy_seed42_h8"]
    affine = runs["affine_motion_prior_only_seed42_h8"]
    dct = runs["dct_residual_prior_only_seed42_h8"]
    wo_sem = runs["v18_wo_semantic_memory_seed42_h8"]
    wo_dense = runs["v18_wo_dense_points_seed42_h8"]
    wo_res = runs["v18_wo_residual_decoder_seed42_h8"]
    wo_aff = runs["v18_wo_affine_prior_seed42_h8"]
    wo_cv = runs["v18_wo_cv_prior_seed42_h8"]
    pt = v17_runs["point_transformer_dense_seed42_h8"]
    v17 = v17_runs["ostf_multitrace_m512_seed42_h8"]

    bootstrap = {
        "audit_name": "stwm_ostf_v18_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v18_m512_vs_constant_velocity_point_L1": _pair_bootstrap(full_m512, cv, "point_l1_px", higher_better=False),
        "v18_m512_vs_constant_velocity_endpoint": _pair_bootstrap(full_m512, cv, "endpoint_error_px", higher_better=False),
        "v18_m512_vs_constant_velocity_extent": _pair_bootstrap(full_m512, cv, "extent_iou", higher_better=True),
        "v18_m512_vs_v17_m512_point_L1": _pair_bootstrap(full_m512, v17, "point_l1_px", higher_better=False),
        "v18_m512_vs_v17_m512_extent": _pair_bootstrap(full_m512, v17, "extent_iou", higher_better=True),
        "v18_m512_vs_point_transformer_point_L1": _pair_bootstrap(full_m512, pt, "point_l1_px", higher_better=False),
        "v18_m512_vs_point_transformer_extent": _pair_bootstrap(full_m512, pt, "extent_iou", higher_better=True),
    }
    dump_json(ROOT / "reports/stwm_ostf_v18_bootstrap_20260502.json", bootstrap)

    m = full_m512["test_metrics"]
    cvm = cv["test_metrics"]
    ptm = pt["test_metrics"]
    v17m = v17["test_metrics"]
    cv_point_boot = bootstrap["v18_m512_vs_constant_velocity_point_L1"]
    cv_endpoint_boot = bootstrap["v18_m512_vs_constant_velocity_endpoint"]
    cv_extent_boot = bootstrap["v18_m512_vs_constant_velocity_extent"]
    v17_point_boot = bootstrap["v18_m512_vs_v17_m512_point_L1"]
    pt_point_boot = bootstrap["v18_m512_vs_point_transformer_point_L1"]
    beats_cv_point = bool(cv_point_boot["zero_excluded"] and (cv_point_boot["mean_delta"] or 0.0) > 0.0)
    beats_cv_endpoint = bool(cv_endpoint_boot["zero_excluded"] and (cv_endpoint_boot["mean_delta"] or 0.0) > 0.0)
    beats_cv_extent = bool(cv_extent_boot["zero_excluded"] and (cv_extent_boot["mean_delta"] or 0.0) > 0.0)
    decision = {
        "audit_name": "stwm_ostf_v18_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "V18_beats_constant_velocity": bool(
            beats_cv_point
            or beats_cv_endpoint
            or beats_cv_extent
            or (
                (m["PCK_32px"] - cvm["PCK_32px"] > 0.01 or m["PCK_16px"] - cvm["PCK_16px"] > 0.01)
                and m["point_L1_px"] <= cvm["point_L1_px"] + 1.0
            )
        ),
        "V18_beats_V17": bool(
            v17_point_boot["zero_excluded"]
            and (v17_point_boot["mean_delta"] or 0.0) > 0.0
            and m["object_extent_iou"] > v17m["object_extent_iou"] + 0.05
        ),
        "V18_beats_point_transformer_or_tradeoff": bool(
            (m["semantic_top5"] is not None and ptm["semantic_top5"] is not None and m["semantic_top5"] > ptm["semantic_top5"] + 0.02)
            or (
                pt_point_boot["zero_excluded"]
                and (pt_point_boot["mean_delta"] or 0.0) > 0.0
                and m["object_extent_iou"] > ptm["object_extent_iou"] + 0.05
            )
        ),
        "dense_points_load_bearing": bool(
            m["point_L1_px"] + 0.05 < wo_dense["test_metrics"]["point_L1_px"]
            or m["visibility_F1"] > wo_dense["test_metrics"]["visibility_F1"] + 0.05
            or m["object_extent_iou"] > wo_dense["test_metrics"]["object_extent_iou"] + 0.0005
        ),
        "physics_prior_load_bearing": bool(
            m["point_L1_px"] + 1.0 < wo_cv["test_metrics"]["point_L1_px"]
            or m["object_extent_iou"] > wo_cv["test_metrics"]["object_extent_iou"] + 0.05
        ),
        "point_residual_decoder_load_bearing": bool(
            m["point_L1_px"] + 0.05 < wo_res["test_metrics"]["point_L1_px"]
            or m["visibility_F1"] > wo_res["test_metrics"]["visibility_F1"] + 0.05
            or m["object_extent_iou"] > wo_res["test_metrics"]["object_extent_iou"] + 0.0005
        ),
        "semantic_unit_compression_helpful": bool(
            m["point_L1_px"] + 5.0 < ptm["point_L1_px"] and m["object_extent_iou"] > ptm["object_extent_iou"] + 0.05
        ),
        "object_dense_semantic_trace_field_claim_allowed": False,
    }
    decision["object_dense_semantic_trace_field_claim_allowed"] = bool(
        decision["V18_beats_V17"]
        and decision["V18_beats_constant_velocity"]
        and decision["V18_beats_point_transformer_or_tradeoff"]
        and decision["dense_points_load_bearing"]
        and decision["physics_prior_load_bearing"]
        and decision["point_residual_decoder_load_bearing"]
    )
    decision["next_step_choice"] = (
        "run_v18_multiseed_and_H16"
        if (
            decision["V18_beats_constant_velocity"]
            and decision["V18_beats_V17"]
            and decision["dense_points_load_bearing"]
            and decision["physics_prior_load_bearing"]
            and decision["point_residual_decoder_load_bearing"]
        )
        else ("improve_loss_or_capacity_again" if decision["V18_beats_V17"] else "fallback_to_sparse_STWM")
    )
    decision["rows_used"] = {
        "constant_velocity_copy": "constant_velocity_copy_seed42_h8",
        "affine_motion_prior_only": "affine_motion_prior_only_seed42_h8",
        "dct_residual_prior_only": "dct_residual_prior_only_seed42_h8",
        "point_transformer_dense": "imported_v17::point_transformer_dense_seed42_h8",
        "v17_ostf_m512": "imported_v17::ostf_multitrace_m512_seed42_h8",
        "v18_m128": "v18_physics_residual_m128_seed42_h8",
        "v18_m512": "v18_physics_residual_m512_seed42_h8",
    }
    dump_json(ROOT / "reports/stwm_ostf_v18_decision_20260502.json", decision)
    write_doc(
        ROOT / "docs/STWM_OSTF_V18_RESULTS_20260502.md",
        "STWM OSTF V18 Results",
        {
            "run_count": len(runs),
            **decision,
        },
        [
            "run_count",
            "V18_beats_constant_velocity",
            "V18_beats_V17",
            "V18_beats_point_transformer_or_tradeoff",
            "dense_points_load_bearing",
            "physics_prior_load_bearing",
            "semantic_unit_compression_helpful",
            "object_dense_semantic_trace_field_claim_allowed",
            "next_step_choice",
        ],
    )
    print("reports/stwm_ostf_v18_eval_summary_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
