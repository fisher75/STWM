#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_traceanything_common_v26_20260502 import (
    ROOT,
    V25_DECISION_PATH,
    analytic_affine_motion_predict,
    analytic_constant_velocity_predict,
    analytic_last_observed_copy_predict,
    build_v26_rows,
)
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import (
    aggregate_item_rows_v26,
    paired_bootstrap_from_rows_v26,
    multimodal_item_scores_v26,
)
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc


def _load_runs() -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((ROOT / "reports/stwm_ostf_v26_runs").glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report:
            out[report["experiment_name"]] = report
    return out


def _analytic_rows(combo: str, kind: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows, proto_centers = build_v26_rows(combo, seed=42)
    samples = rows["test"]
    if kind == "constant_velocity_copy":
        pred_points, pred_vis, pred_sem = analytic_constant_velocity_predict(samples, proto_count=32, proto_centers=proto_centers, semantic_mode="observed_memory")
    elif kind == "affine_motion_prior":
        pred_points, pred_vis, pred_sem = analytic_affine_motion_predict(samples, proto_count=32, proto_centers=proto_centers, semantic_mode="observed_memory")
    else:
        pred_points, pred_vis, pred_sem = analytic_last_observed_copy_predict(samples, proto_count=32, proto_centers=proto_centers)
    point_modes = pred_points[:, :, :, None, :]
    mode_logits = np.zeros((pred_points.shape[0], 1), dtype=np.float32)
    item_rows = multimodal_item_scores_v26(
        samples,
        point_modes=point_modes,
        mode_logits=mode_logits,
        top1_point_pred=pred_points,
        weighted_point_pred=pred_points,
        pred_vis_logits=pred_vis,
        pred_proto_logits=pred_sem,
        pred_logvar=None,
        cv_mode_index=0,
    )
    subsets = {
        "top20_cv_hard": aggregate_item_rows_v26(item_rows, subset_key="top20_cv_hard"),
        "top30_cv_hard": aggregate_item_rows_v26(item_rows, subset_key="top30_cv_hard"),
        "occlusion": aggregate_item_rows_v26(item_rows, subset_key="occlusion_hard"),
        "nonlinear": aggregate_item_rows_v26(item_rows, subset_key="nonlinear_hard"),
        "interaction": aggregate_item_rows_v26(item_rows, subset_key="interaction_hard"),
    }
    by_ds = {ds: aggregate_item_rows_v26(item_rows, dataset=ds) for ds in sorted({s.dataset for s in samples})}
    return item_rows, aggregate_item_rows_v26(item_rows), subsets, by_ds


def _bool_positive(boot: dict[str, Any]) -> bool:
    return bool(boot.get("zero_excluded") and (boot.get("mean_delta") or 0.0) > 0.0)


def _decision_from_bootstrap(runs: dict[str, dict[str, Any]], bootstrap: dict[str, Any]) -> dict[str, Any]:
    m128 = runs.get("v26_traceanything_m128_h32_seed42")
    m512 = runs.get("v26_traceanything_m512_h32_seed42")
    h64 = runs.get("v26_traceanything_m128_h64_seed42")
    dense = bootstrap.get("m128_vs_wo_dense_hard_minfde", {})
    sem = bootstrap.get("m128_vs_wo_semantic_hard_minfde", {})
    multi = bootstrap.get("m128_vs_single_mode_hard_minfde", {})
    phys = bootstrap.get("m128_vs_wo_physics_hard_minfde", {})
    v25_decision = json.loads(V25_DECISION_PATH.read_text(encoding="utf-8")) if V25_DECISION_PATH.exists() else {}
    beats_cv_minfde = _bool_positive(bootstrap.get("m128_vs_cv_all_minfde", {})) or _bool_positive(bootstrap.get("m128_vs_cv_hard_minfde", {}))
    beats_cv_miss = _bool_positive(bootstrap.get("m128_vs_cv_all_miss32", {})) or _bool_positive(bootstrap.get("m128_vs_cv_hard_miss32", {}))
    dense_lb = _bool_positive(dense) or _bool_positive(bootstrap.get("m128_vs_wo_dense_extent_hard", {}))
    sem_lb = _bool_positive(sem) or _bool_positive(bootstrap.get("m128_vs_wo_semantic_sem_top5", {}))
    multi_lb = _bool_positive(multi)
    phys_lb = _bool_positive(phys)
    claim = bool(beats_cv_minfde and beats_cv_miss and dense_lb and multi_lb and phys_lb and sem_lb)
    if claim:
        next_step = "run_v26_multiseed"
    elif not v25_decision.get("visibility_quality_acceptable", True):
        next_step = "improve_visibility_extraction"
    else:
        next_step = "improve_model_or_loss"
    return {
        "cache_verified": True,
        "H32_M128_trained": m128 is not None,
        "H32_M512_trained": m512 is not None,
        "H64_stress_trained": h64 is not None,
        "V26_beats_CV_minFDE": beats_cv_minfde,
        "V26_beats_CV_MissRate": beats_cv_miss,
        "dense_points_load_bearing": dense_lb,
        "semantic_memory_load_bearing": sem_lb,
        "multimodal_load_bearing": multi_lb,
        "physics_prior_load_bearing": phys_lb,
        "visibility_quality_sufficient": bool(v25_decision.get("visibility_quality_acceptable", True)),
        "object_dense_semantic_trace_field_claim_allowed": claim,
        "next_step_choice": next_step,
    }


def main() -> int:
    runs = _load_runs()
    if not runs:
        raise SystemExit("No V26 run reports found in reports/stwm_ostf_v26_runs")

    cv_m128_rows, cv_m128_all, cv_m128_sub, cv_m128_ds = _analytic_rows("M128_H32", "constant_velocity_copy")
    affine_m128_rows, affine_m128_all, affine_m128_sub, affine_m128_ds = _analytic_rows("M128_H32", "affine_motion_prior")
    last_m128_rows, last_m128_all, last_m128_sub, last_m128_ds = _analytic_rows("M128_H32", "last_observed_copy")
    cv_m512_rows, cv_m512_all, cv_m512_sub, cv_m512_ds = _analytic_rows("M512_H32", "constant_velocity_copy")
    affine_m512_rows, affine_m512_all, affine_m512_sub, affine_m512_ds = _analytic_rows("M512_H32", "affine_motion_prior")
    cv_h64_rows, cv_h64_all, cv_h64_sub, cv_h64_ds = _analytic_rows("M128_H64", "constant_velocity_copy")
    affine_h64_rows, affine_h64_all, affine_h64_sub, affine_h64_ds = _analytic_rows("M128_H64", "affine_motion_prior")

    bootstrap = {
        "audit_name": "stwm_ostf_v26_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if "v26_traceanything_m128_h32_seed42" in runs:
        r = runs["v26_traceanything_m128_h32_seed42"]
        bootstrap["m128_vs_cv_all_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_m128_rows, metric="minFDE_K_px", higher_better=False)
        bootstrap["m128_vs_cv_hard_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_m128_rows, metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard")
        bootstrap["m128_vs_cv_all_miss32"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_m128_rows, metric="MissRate_32px", higher_better=False)
        bootstrap["m128_vs_cv_hard_miss32"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_m128_rows, metric="MissRate_32px", higher_better=False, subset_key="top20_cv_hard")
        bootstrap["m128_vs_affine_all_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], affine_m128_rows, metric="minFDE_K_px", higher_better=False)
        if "v26_traceanything_m128_h32_wo_dense_points_seed42" in runs:
            bootstrap["m128_vs_wo_dense_hard_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], runs["v26_traceanything_m128_h32_wo_dense_points_seed42"]["item_scores"], metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard")
            bootstrap["m128_vs_wo_dense_extent_hard"] = paired_bootstrap_from_rows_v26(r["item_scores"], runs["v26_traceanything_m128_h32_wo_dense_points_seed42"]["item_scores"], metric="top1_object_extent_iou", higher_better=True, subset_key="top20_cv_hard")
        if "v26_traceanything_m128_h32_wo_semantic_memory_seed42" in runs:
            bootstrap["m128_vs_wo_semantic_hard_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], runs["v26_traceanything_m128_h32_wo_semantic_memory_seed42"]["item_scores"], metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard")
            bootstrap["m128_vs_wo_semantic_sem_top5"] = paired_bootstrap_from_rows_v26(r["item_scores"], runs["v26_traceanything_m128_h32_wo_semantic_memory_seed42"]["item_scores"], metric="semantic_top5", higher_better=True)
        if "v26_traceanything_m128_h32_single_mode_seed42" in runs:
            bootstrap["m128_vs_single_mode_hard_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], runs["v26_traceanything_m128_h32_single_mode_seed42"]["item_scores"], metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard")
        if "v26_traceanything_m128_h32_wo_physics_prior_seed42" in runs:
            bootstrap["m128_vs_wo_physics_hard_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], runs["v26_traceanything_m128_h32_wo_physics_prior_seed42"]["item_scores"], metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard")
    if "v26_traceanything_m512_h32_seed42" in runs:
        r = runs["v26_traceanything_m512_h32_seed42"]
        bootstrap["m512_vs_cv_all_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_m512_rows, metric="minFDE_K_px", higher_better=False)
        bootstrap["m512_vs_cv_hard_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_m512_rows, metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard")
        bootstrap["m512_vs_cv_all_miss32"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_m512_rows, metric="MissRate_32px", higher_better=False)
    if "v26_traceanything_m128_h64_seed42" in runs:
        r = runs["v26_traceanything_m128_h64_seed42"]
        bootstrap["h64_vs_cv_all_minfde"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_h64_rows, metric="minFDE_K_px", higher_better=False)
        bootstrap["h64_vs_cv_all_miss64"] = paired_bootstrap_from_rows_v26(r["item_scores"], cv_h64_rows, metric="MissRate_64px", higher_better=False)
    dump_json(ROOT / "reports/stwm_ostf_v26_bootstrap_20260502.json", bootstrap)

    train_summary = {
        "audit_name": "stwm_ostf_v26_train_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "steps": r["steps"],
                "parameter_count": r["parameter_count"],
                "best_checkpoint_path": r["best_checkpoint_path"],
                "best_val_score": r["best_val_score"],
                "best_step": r.get("best_step"),
                "duration_sec": r.get("duration_sec"),
            }
            for name, r in runs.items()
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v26_train_summary_20260502.json", train_summary)

    eval_summary = {
        "audit_name": "stwm_ostf_v26_eval_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "test_metrics": r["test_metrics"],
                "test_subset_metrics": r["test_subset_metrics"],
                "test_metrics_by_dataset": r["test_metrics_by_dataset"],
                "test_diversity_valid": r["test_diversity_valid"],
                "val_metrics": r["val_metrics"],
                "val_subset_metrics": r["val_subset_metrics"],
            }
            for name, r in runs.items()
        },
        "baselines": {
            "constant_velocity_copy_M128_H32": {"test_metrics": cv_m128_all, "test_subset_metrics": cv_m128_sub, "test_metrics_by_dataset": cv_m128_ds},
            "affine_motion_prior_M128_H32": {"test_metrics": affine_m128_all, "test_subset_metrics": affine_m128_sub, "test_metrics_by_dataset": affine_m128_ds},
            "last_observed_copy_M128_H32": {"test_metrics": last_m128_all, "test_subset_metrics": last_m128_sub, "test_metrics_by_dataset": last_m128_ds},
            "constant_velocity_copy_M512_H32": {"test_metrics": cv_m512_all, "test_subset_metrics": cv_m512_sub, "test_metrics_by_dataset": cv_m512_ds},
            "affine_motion_prior_M512_H32": {"test_metrics": affine_m512_all, "test_subset_metrics": affine_m512_sub, "test_metrics_by_dataset": affine_m512_ds},
            "constant_velocity_copy_M128_H64": {"test_metrics": cv_h64_all, "test_subset_metrics": cv_h64_sub, "test_metrics_by_dataset": cv_h64_ds},
            "affine_motion_prior_M128_H64": {"test_metrics": affine_h64_all, "test_subset_metrics": affine_h64_sub, "test_metrics_by_dataset": affine_h64_ds},
            "cotracker_v21_v22_if_comparable": {"comparable": False, "reason": "Existing V21/V22 runs are CoTracker-teacher H8/H16 models and are not same-cache/same-horizon comparable to TraceAnything H32/H64 V26."},
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v26_eval_summary_20260502.json", eval_summary)

    decision = _decision_from_bootstrap(runs, bootstrap)
    decision["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(ROOT / "reports/stwm_ostf_v26_decision_20260502.json", decision)
    write_doc(
        ROOT / "docs/STWM_OSTF_V26_RESULTS_20260502.md",
        "STWM OSTF V26 Results",
        {
            "run_count": len(runs),
            "H32_M128_trained": decision["H32_M128_trained"],
            "H32_M512_trained": decision["H32_M512_trained"],
            "H64_stress_trained": decision["H64_stress_trained"],
            "V26_beats_CV_minFDE": decision["V26_beats_CV_minFDE"],
            "V26_beats_CV_MissRate": decision["V26_beats_CV_MissRate"],
            "dense_points_load_bearing": decision["dense_points_load_bearing"],
            "semantic_memory_load_bearing": decision["semantic_memory_load_bearing"],
            "multimodal_load_bearing": decision["multimodal_load_bearing"],
            "physics_prior_load_bearing": decision["physics_prior_load_bearing"],
            "object_dense_semantic_trace_field_claim_allowed": decision["object_dense_semantic_trace_field_claim_allowed"],
            "next_step_choice": decision["next_step_choice"],
        },
        [
            "run_count",
            "H32_M128_trained",
            "H32_M512_trained",
            "H64_stress_trained",
            "V26_beats_CV_minFDE",
            "V26_beats_CV_MissRate",
            "dense_points_load_bearing",
            "semantic_memory_load_bearing",
            "multimodal_load_bearing",
            "physics_prior_load_bearing",
            "object_dense_semantic_trace_field_claim_allowed",
            "next_step_choice",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
