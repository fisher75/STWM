#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_multimodal_metrics_v21 import aggregate_item_rows, paired_bootstrap_from_rows
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, load_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import (
    analytic_affine_motion_predict,
    analytic_constant_velocity_predict,
    build_v18_rows,
)
from stwm.tools.ostf_v20_common_20260502 import hard_subset_flags, load_context_cache, sample_key
from stwm.tools.ostf_multimodal_metrics_v21 import multimodal_item_scores


def _load_runs() -> dict[str, dict[str, Any]]:
    out = {}
    for path in sorted((ROOT / "reports/stwm_ostf_v21_runs").glob("*.json")):
        report = load_json(path)
        if report:
            out[report["experiment_name"]] = report
    return out


def _subset_flags(samples: list[Any], context_cache_rel: str) -> dict[str, np.ndarray]:
    ctx_map = load_context_cache(ROOT / context_cache_rel)
    records = []
    for s in samples:
        c = ctx_map[sample_key(s)]
        records.append(
            {
                "cv_point_l1_proxy": c["cv_point_l1_proxy"],
                "curvature_proxy": c["curvature_proxy"],
                "occlusion_ratio": c["occlusion_ratio"],
                "interaction_proxy": c["interaction_proxy"],
            }
        )
    return hard_subset_flags(records)


def _analytic_rows(combo: str, context_cache_rel: str, kind: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows, proto_centers = build_v18_rows(combo, seed=42)
    samples = rows["test"]
    flags = _subset_flags(samples, context_cache_rel)
    fn = analytic_constant_velocity_predict if kind == "constant_velocity_copy" else analytic_affine_motion_predict
    pred_points, pred_vis, pred_sem = fn(samples, proto_count=32, proto_centers=proto_centers, semantic_mode="observed_memory")
    point_modes = pred_points[:, :, :, None, :]
    mode_logits = np.zeros((pred_points.shape[0], 1), dtype=np.float32)
    item_rows = multimodal_item_scores(
        samples,
        point_modes=point_modes,
        mode_logits=mode_logits,
        point_pred=pred_points,
        pred_vis_logits=pred_vis,
        pred_proto_logits=pred_sem,
        subset_flags=flags,
        cv_mode_index=0,
    )
    return (
        item_rows,
        aggregate_item_rows(item_rows),
        {
            "cv_hard_top20": aggregate_item_rows(item_rows, subset_key="top20_cv_hard"),
            "occlusion": aggregate_item_rows(item_rows, subset_key="occlusion_hard"),
            "nonlinear": aggregate_item_rows(item_rows, subset_key="nonlinear_hard"),
            "interaction": aggregate_item_rows(item_rows, subset_key="interaction_hard"),
        },
        {ds: aggregate_item_rows(item_rows, dataset=ds) for ds in sorted({s.dataset for s in samples})},
    )


def _same_rows_bootstrap(rows: list[dict[str, Any]], better_metric: str, worse_metric: str, *, subset_key: str | None = None, higher_better: bool = False) -> dict[str, Any]:
    a = []
    b = []
    for r in rows:
        if subset_key is not None and not r.get(subset_key, False):
            continue
        if r.get(better_metric) is None or r.get(worse_metric) is None:
            continue
        a.append(float(r[better_metric]))
        b.append(float(r[worse_metric]))
    if not a:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    a_np = np.asarray(a, dtype=np.float64)
    b_np = np.asarray(b, dtype=np.float64)
    if not higher_better:
        a_np = -a_np
        b_np = -b_np
    delta = a_np - b_np
    rng = np.random.default_rng(42)
    means = []
    for _ in range(1000):
        idx = rng.integers(0, len(delta), size=len(delta))
        means.append(float(delta[idx].mean()))
    lo, hi = np.percentile(np.asarray(means), [2.5, 97.5]).tolist()
    return {
        "item_count": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool((lo > 0.0) or (hi < 0.0)),
    }


def _hard_score(report: dict[str, Any]) -> float:
    hard = report["test_subset_metrics"]["cv_hard_top20"]
    allm = report["test_metrics"]
    return (
        -1.30 * float(hard["minFDE_K_px"])
        - 0.60 * float(hard["minADE_K_px"])
        + 18.0 * float(hard["BestOfK_PCK_16px"])
        + 12.0 * float(hard["BestOfK_PCK_32px"])
        - 6.0 * float(hard["MissRate_32px"])
        - 0.20 * float(allm["weighted_point_L1_px"])
        + 4.0 * float(allm["weighted_object_extent_iou"])
    )


def _bool_positive(boot: dict[str, Any]) -> bool:
    return bool(boot.get("zero_excluded") and (boot.get("mean_delta") or 0.0) > 0.0)


def main() -> int:
    runs = _load_runs()
    if not runs:
        raise SystemExit("No V21 run reports found in reports/stwm_ostf_v21_runs")

    audit = load_json(ROOT / "reports/stwm_ostf_multimodal_eval_gap_v21_20260502.json")
    v20_eval = load_json(ROOT / "reports/stwm_ostf_v20_eval_summary_20260502.json")
    point_transformer = load_json(ROOT / "reports/stwm_ostf_v17_runs/point_transformer_dense_seed42_h8.json")

    cv_m128_rows, cv_m128_all, cv_m128_sub, cv_m128_ds = _analytic_rows(
        "M128_H8",
        "outputs/cache/stwm_ostf_context_features_v20/M128_H8_context_features.npz",
        "constant_velocity_copy",
    )
    affine_m128_rows, affine_m128_all, affine_m128_sub, affine_m128_ds = _analytic_rows(
        "M128_H8",
        "outputs/cache/stwm_ostf_context_features_v20/M128_H8_context_features.npz",
        "affine_motion_prior_only",
    )
    cv_m512_rows, cv_m512_all, cv_m512_sub, cv_m512_ds = _analytic_rows(
        "M512_H8",
        "outputs/cache/stwm_ostf_context_features_v20/M512_H8_context_features.npz",
        "constant_velocity_copy",
    )
    affine_m512_rows, affine_m512_all, affine_m512_sub, affine_m512_ds = _analytic_rows(
        "M512_H8",
        "outputs/cache/stwm_ostf_context_features_v20/M512_H8_context_features.npz",
        "affine_motion_prior_only",
    )

    main_candidates = [r for r in runs.values() if r["model_kind"] in {"v21_multimodal_m128", "v21_multimodal_m512"}]
    best_report = max(main_candidates, key=_hard_score)
    best_name = str(best_report["experiment_name"])
    matching_cv_rows = cv_m128_rows if "m128" in best_name else cv_m512_rows
    matching_cv_all = cv_m128_all if "m128" in best_name else cv_m512_all
    matching_cv_sub = cv_m128_sub if "m128" in best_name else cv_m512_sub
    matching_affine_all = affine_m128_all if "m128" in best_name else affine_m512_all

    bootstrap = {
        "audit_name": "stwm_ostf_v21_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
        "best_of_k_vs_weighted_all_minFDE": _same_rows_bootstrap(best_report["item_scores"], "minFDE_K_px", "weighted_endpoint_error_px", higher_better=False),
        "best_of_k_vs_weighted_hard_minFDE": _same_rows_bootstrap(best_report["item_scores"], "minFDE_K_px", "weighted_endpoint_error_px", subset_key="top20_cv_hard", higher_better=False),
        "v21_best_vs_cv_all_minFDE": paired_bootstrap_from_rows(best_report["item_scores"], matching_cv_rows, metric="minFDE_K_px", higher_better=False),
        "v21_best_vs_cv_hard_minFDE": paired_bootstrap_from_rows(best_report["item_scores"], matching_cv_rows, metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard"),
        "v21_best_vs_cv_all_missrate32": paired_bootstrap_from_rows(best_report["item_scores"], matching_cv_rows, metric="MissRate_32px", higher_better=False),
        "v21_best_vs_cv_hard_missrate32": paired_bootstrap_from_rows(best_report["item_scores"], matching_cv_rows, metric="MissRate_32px", higher_better=False, subset_key="top20_cv_hard"),
    }
    if "v21_multimodal_m512_wo_context_seed42_h8" in runs:
        bootstrap["v21_m512_vs_wo_context_hard_minFDE"] = paired_bootstrap_from_rows(
            runs["v21_multimodal_m512_seed42_h8"]["item_scores"],
            runs["v21_multimodal_m512_wo_context_seed42_h8"]["item_scores"],
            metric="minFDE_K_px",
            higher_better=False,
            subset_key="top20_cv_hard",
        )
    if "v21_multimodal_m512_wo_dense_points_seed42_h8" in runs:
        bootstrap["v21_m512_vs_wo_dense_hard_minFDE"] = paired_bootstrap_from_rows(
            runs["v21_multimodal_m512_seed42_h8"]["item_scores"],
            runs["v21_multimodal_m512_wo_dense_points_seed42_h8"]["item_scores"],
            metric="minFDE_K_px",
            higher_better=False,
            subset_key="top20_cv_hard",
        )
    if "v21_multimodal_m512_wo_diversity_seed42_h8" in runs:
        bootstrap["v21_m512_vs_wo_diversity_hard_minFDE"] = paired_bootstrap_from_rows(
            runs["v21_multimodal_m512_seed42_h8"]["item_scores"],
            runs["v21_multimodal_m512_wo_diversity_seed42_h8"]["item_scores"],
            metric="minFDE_K_px",
            higher_better=False,
            subset_key="top20_cv_hard",
        )
    if "v21_multimodal_m512_single_mode_seed42_h8" in runs:
        bootstrap["v21_m512_vs_single_mode_hard_minFDE"] = paired_bootstrap_from_rows(
            runs["v21_multimodal_m512_seed42_h8"]["item_scores"],
            runs["v21_multimodal_m512_single_mode_seed42_h8"]["item_scores"],
            metric="minFDE_K_px",
            higher_better=False,
            subset_key="top20_cv_hard",
        )
    dump_json(ROOT / "reports/stwm_ostf_v21_bootstrap_20260502.json", bootstrap)

    train_summary = {
        "audit_name": "stwm_ostf_v21_train_summary",
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
    dump_json(ROOT / "reports/stwm_ostf_v21_train_summary_20260502.json", train_summary)

    eval_summary = {
        "audit_name": "stwm_ostf_v21_eval_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
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
            "constant_velocity_copy_M128": {"test_metrics": cv_m128_all, "test_subset_metrics": cv_m128_sub, "test_metrics_by_dataset": cv_m128_ds},
            "affine_motion_prior_M128": {"test_metrics": affine_m128_all, "test_subset_metrics": affine_m128_sub, "test_metrics_by_dataset": affine_m128_ds},
            "constant_velocity_copy_M512": {"test_metrics": cv_m512_all, "test_subset_metrics": cv_m512_sub, "test_metrics_by_dataset": cv_m512_ds},
            "affine_motion_prior_M512": {"test_metrics": affine_m512_all, "test_subset_metrics": affine_m512_sub, "test_metrics_by_dataset": affine_m512_ds},
            "v20_deterministic": v20_eval.get("experiments", {}).get("v20_context_residual_m512_seed42_h8"),
            "point_transformer_dense": {"test_metrics": point_transformer.get("test_metrics")},
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v21_eval_summary_20260502.json", eval_summary)

    best_metrics = best_report["test_metrics"]
    best_hard = best_report["test_subset_metrics"]["cv_hard_top20"]
    no_harm = bool(
        float(best_metrics["weighted_point_L1_px"]) <= float(matching_cv_all["weighted_point_L1_px"]) * 1.02
        and float(best_metrics["weighted_endpoint_error_px"]) <= float(matching_cv_all["weighted_endpoint_error_px"]) * 1.02
    )
    hard_positive = _bool_positive(bootstrap["v21_best_vs_cv_hard_minFDE"]) or _bool_positive(bootstrap["v21_best_vs_cv_hard_missrate32"])
    all_positive = _bool_positive(bootstrap["v21_best_vs_cv_all_minFDE"]) or _bool_positive(bootstrap["v21_best_vs_cv_all_missrate32"])
    best_of_k_beats_weighted = _bool_positive(bootstrap["best_of_k_vs_weighted_all_minFDE"]) or _bool_positive(bootstrap["best_of_k_vs_weighted_hard_minFDE"])
    diversity_valid = bool(best_report["test_diversity_valid"])
    context_load = _bool_positive(bootstrap.get("v21_m512_vs_wo_context_hard_minFDE", {})) if "v21_m512_vs_wo_context_hard_minFDE" in bootstrap else None
    dense_load = _bool_positive(bootstrap.get("v21_m512_vs_wo_dense_hard_minFDE", {})) if "v21_m512_vs_wo_dense_hard_minFDE" in bootstrap else None
    multi_vs_single = _bool_positive(bootstrap.get("v21_m512_vs_single_mode_hard_minFDE", {})) if "v21_m512_vs_single_mode_hard_minFDE" in bootstrap else None
    claim_allowed = bool(hard_positive and no_harm and diversity_valid and best_of_k_beats_weighted)
    object_dense_claim_allowed = bool(
        claim_allowed
        and best_metrics.get("semantic_top5") is not None
        and float(best_metrics["semantic_top5"]) >= float(matching_cv_all["semantic_top5"] or 0.0)
    )
    decision = {
        "audit_name": "stwm_ostf_v21_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
        "multimodal_eval_gap_confirmed": bool(audit.get("current_deterministic_eval_invalidates_multihypothesis_claim", False) or audit.get("best_of_K_beats_weighted_average", False)),
        "best_of_K_beats_weighted_average": best_of_k_beats_weighted,
        "V21_beats_CV_hard_subset_minFDE": _bool_positive(bootstrap["v21_best_vs_cv_hard_minFDE"]),
        "V21_beats_CV_all_average_minFDE": _bool_positive(bootstrap["v21_best_vs_cv_all_minFDE"]),
        "V21_no_harm_deterministic": no_harm,
        "hypothesis_diversity_valid": diversity_valid,
        "context_load_bearing": context_load,
        "dense_points_load_bearing": dense_load,
        "multi_hypothesis_beats_single_mode": multi_vs_single,
        "multimodal_world_model_claim_allowed": claim_allowed,
        "object_dense_semantic_trace_field_claim_allowed": object_dense_claim_allowed,
        "best_variant_metrics": best_metrics,
        "matching_cv_metrics": matching_cv_all,
        "matching_affine_metrics": matching_affine_all,
        "hard_subset_metrics": {"best_v21": best_hard, "matching_cv": matching_cv_sub["cv_hard_top20"]},
        "next_step_choice": (
            "run_v21_multiseed_H16"
            if claim_allowed
            else ("improve_multimodal_objective" if diversity_valid else "fallback_to_FSTF_paper")
        ),
    }
    dump_json(ROOT / "reports/stwm_ostf_v21_decision_20260502.json", decision)
    write_doc(
        ROOT / "docs/STWM_OSTF_V21_RESULTS_20260502.md",
        "STWM OSTF V21 Results",
        decision,
        [
            "best_variant_name",
            "multimodal_eval_gap_confirmed",
            "best_of_K_beats_weighted_average",
            "V21_beats_CV_hard_subset_minFDE",
            "V21_beats_CV_all_average_minFDE",
            "V21_no_harm_deterministic",
            "hypothesis_diversity_valid",
            "context_load_bearing",
            "dense_points_load_bearing",
            "multi_hypothesis_beats_single_mode",
            "multimodal_world_model_claim_allowed",
            "object_dense_semantic_trace_field_claim_allowed",
            "next_step_choice",
        ],
    )
    print((ROOT / "reports/stwm_ostf_v21_decision_20260502.json").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
