#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_lastobs_v28_common_20260502 import (
    ROOT,
    V28_SUBSET_KEYS,
    add_v28_flags_to_item_rows,
    build_v28_rows,
    predict_damped_velocity,
    predict_last,
    v28_subset_aggregate,
)
from stwm.tools.ostf_traceanything_common_v26_20260502 import (
    analytic_constant_velocity_predict,
)
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import (
    aggregate_item_rows_v26,
    multimodal_item_scores_v26,
    paired_bootstrap_from_rows_v26,
)
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v27_prior_utils_20260502 import predict_stable_affine


RUN_DIR = ROOT / "reports/stwm_ostf_v28_runs"
TRAIN_SUMMARY = ROOT / "reports/stwm_ostf_v28_train_summary_20260502.json"
EVAL_SUMMARY = ROOT / "reports/stwm_ostf_v28_eval_summary_20260502.json"
BOOTSTRAP_PATH = ROOT / "reports/stwm_ostf_v28_bootstrap_20260502.json"
DECISION_PATH = ROOT / "reports/stwm_ostf_v28_decision_20260502.json"
RESULTS_DOC = ROOT / "docs/STWM_OSTF_V28_RESULTS_20260502.md"


def _load_runs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(RUN_DIR.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        if report:
            out[str(report["experiment_name"])] = report
    return out


def _analytic_rows(combo: str, kind: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any], float]:
    rows, proto_centers, gamma, _ = build_v28_rows(combo, seed=42)
    samples = rows["test"]
    if kind == "last_observed_copy":
        pred_points = predict_last(samples)
        pred_vis = np.stack([np.where(np.repeat(s.obs_vis[:, -1:], s.h, axis=1), 8.0, -8.0).astype(np.float32) for s in samples], axis=0)
        pred_sem = None
    elif kind == "damped_velocity":
        pred_points = predict_damped_velocity(samples, gamma)
        pred_vis = np.stack([np.where(np.repeat(s.obs_vis[:, -1:], s.h, axis=1), 8.0, -8.0).astype(np.float32) for s in samples], axis=0)
        pred_sem = None
    elif kind == "constant_velocity_copy":
        pred_points, pred_vis, pred_sem = analytic_constant_velocity_predict(samples, proto_count=32, proto_centers=proto_centers, semantic_mode="observed_memory")
    elif kind == "fixed_affine_prior":
        pred_points = predict_stable_affine(samples, anchor_gamma=0.25)
        pred_vis = np.stack([np.where(np.repeat(s.obs_vis[:, -1:], s.h, axis=1), 8.0, -8.0).astype(np.float32) for s in samples], axis=0)
        pred_sem = None
    else:
        raise ValueError(kind)
    if pred_sem is None:
        from stwm.tools.ostf_v27_prior_utils_20260502 import observed_memory_logits

        pred_sem = observed_memory_logits(samples, proto_centers, proto_count=32)
    item_rows = multimodal_item_scores_v26(
        samples,
        point_modes=pred_points[:, :, :, None, :],
        mode_logits=np.zeros((len(samples), 1), dtype=np.float32),
        top1_point_pred=pred_points,
        weighted_point_pred=pred_points,
        pred_vis_logits=pred_vis,
        pred_proto_logits=pred_sem,
        pred_logvar=None,
        cv_mode_index=0,
    )
    item_rows = add_v28_flags_to_item_rows(item_rows, samples)
    by_ds = {ds: aggregate_item_rows_v26(item_rows, dataset=ds) for ds in sorted({s.dataset for s in samples})}
    return item_rows, aggregate_item_rows_v26(item_rows), v28_subset_aggregate(item_rows), by_ds, gamma


def _positive(boot: dict[str, Any]) -> bool:
    return bool(boot.get("zero_excluded") and (boot.get("mean_delta") or 0.0) > 0.0)


def _fmt_boot(boot: dict[str, Any]) -> dict[str, Any]:
    return {
        "mean_delta": boot.get("mean_delta"),
        "ci95": boot.get("ci95"),
        "zero_excluded": boot.get("zero_excluded"),
        "item_count": boot.get("item_count"),
    }


def _try_boot(bootstrap: dict[str, Any], name: str, rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str, higher_better: bool, subset_key: str | None = None) -> None:
    bootstrap[name] = paired_bootstrap_from_rows_v26(rows_a, rows_b, metric=metric, higher_better=higher_better, subset_key=subset_key)


def _decision(runs: dict[str, dict[str, Any]], bootstrap: dict[str, Any]) -> dict[str, Any]:
    main = runs.get("v28_lastobs_m128_h32_seed42")
    m512 = runs.get("v28_lastobs_m512_h32_seed42")
    h64 = runs.get("v28_lastobs_m128_h64_seed42")
    beats_last_all = _positive(bootstrap.get("m128_vs_last_all_minfde", {}))
    beats_last_hard = _positive(bootstrap.get("m128_vs_last_hard_minfde", {})) or _positive(bootstrap.get("m128_vs_last_hard_miss32", {}))
    beats_damped = _positive(bootstrap.get("m128_vs_damped_hard_minfde", {})) or _positive(bootstrap.get("m128_vs_damped_all_minfde", {}))
    residual_lb = _positive(bootstrap.get("m128_vs_wo_residual_hard_minfde", {})) or _positive(bootstrap.get("m128_vs_prior_only_hard_minfde", {}))
    dense_lb = _positive(bootstrap.get("m128_vs_wo_dense_hard_minfde", {})) or _positive(bootstrap.get("m128_vs_wo_dense_extent_hard", {}))
    semantic_lb = _positive(bootstrap.get("m128_vs_wo_semantic_sem_top5", {})) or _positive(bootstrap.get("m128_vs_wo_semantic_hard_minfde", {}))
    visibility_ok = bool(main and (main.get("test_metrics", {}).get("top1_visibility_F1") or 0.0) >= 0.35)
    claim = bool(beats_last_hard and not bootstrap.get("m128_vs_last_all_minfde", {}).get("ci95", [0, 0])[1] < -1e-9 and residual_lb and dense_lb and semantic_lb and visibility_ok)
    if claim:
        next_step = "run_v28_multiseed_H64"
    elif not residual_lb:
        next_step = "improve_residual_modes_or_semantics"
    elif not semantic_lb or not dense_lb:
        next_step = "improve_residual_modes_or_semantics"
    else:
        next_step = "fallback_to_FSTF"
    return {
        "cache_verified": True,
        "strongest_prior_used": "last_observed_copy",
        "H32_M128_trained": main is not None,
        "H32_M512_trained": m512 is not None,
        "H64_stress_trained": h64 is not None,
        "V28_beats_last_observed_all_average": beats_last_all,
        "V28_beats_last_observed_hard_subset": beats_last_hard,
        "V28_beats_damped_prior": beats_damped,
        "residual_modes_load_bearing": residual_lb,
        "dense_points_load_bearing": dense_lb,
        "semantic_memory_load_bearing": semantic_lb,
        "visibility_quality_sufficient": visibility_ok,
        "object_dense_semantic_trace_field_claim_allowed": claim,
        "next_step_choice": next_step,
    }


def main() -> int:
    runs = _load_runs()
    if not runs:
        raise SystemExit("No V28 run reports found in reports/stwm_ostf_v28_runs")

    baselines: dict[str, Any] = {}
    baseline_rows: dict[str, list[dict[str, Any]]] = {}
    for combo in ["M128_H32", "M512_H32", "M128_H64"]:
        for kind in ["last_observed_copy", "damped_velocity", "constant_velocity_copy", "fixed_affine_prior"]:
            rows, all_m, sub_m, by_ds, gamma = _analytic_rows(combo, kind)
            key = f"{kind}_{combo}"
            baseline_rows[key] = rows
            baselines[key] = {
                "test_metrics": all_m,
                "test_subset_metrics": sub_m,
                "test_metrics_by_dataset": by_ds,
                "val_selected_damped_gamma": gamma,
            }

    bootstrap: dict[str, Any] = {
        "audit_name": "stwm_ostf_v28_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "strongest_prior_used": "last_observed_copy",
    }
    main = runs.get("v28_lastobs_m128_h32_seed42")
    if main:
        main_rows = main["item_scores"]
        last = baseline_rows["last_observed_copy_M128_H32"]
        damped = baseline_rows["damped_velocity_M128_H32"]
        _try_boot(bootstrap, "m128_vs_last_all_minfde", main_rows, last, "minFDE_K_px", False)
        _try_boot(bootstrap, "m128_vs_last_hard_minfde", main_rows, last, "minFDE_K_px", False, subset_key="last_observed_hard_top20")
        _try_boot(bootstrap, "m128_vs_last_hard_miss32", main_rows, last, "MissRate_32px", False, subset_key="last_observed_hard_top20")
        _try_boot(bootstrap, "m128_vs_damped_all_minfde", main_rows, damped, "minFDE_K_px", False)
        _try_boot(bootstrap, "m128_vs_damped_hard_minfde", main_rows, damped, "minFDE_K_px", False, subset_key="damped_cv_hard_top20")
        if "v28_lastobs_m128_h32_wo_dense_points_seed42" in runs:
            r = runs["v28_lastobs_m128_h32_wo_dense_points_seed42"]["item_scores"]
            _try_boot(bootstrap, "m128_vs_wo_dense_hard_minfde", main_rows, r, "minFDE_K_px", False, subset_key="last_observed_hard_top20")
            _try_boot(bootstrap, "m128_vs_wo_dense_extent_hard", main_rows, r, "top1_object_extent_iou", True, subset_key="last_observed_hard_top20")
        if "v28_lastobs_m128_h32_wo_semantic_memory_seed42" in runs:
            r = runs["v28_lastobs_m128_h32_wo_semantic_memory_seed42"]["item_scores"]
            _try_boot(bootstrap, "m128_vs_wo_semantic_hard_minfde", main_rows, r, "minFDE_K_px", False, subset_key="last_observed_hard_top20")
            _try_boot(bootstrap, "m128_vs_wo_semantic_sem_top5", main_rows, r, "semantic_top5", True)
        if "v28_lastobs_m128_h32_wo_residual_modes_seed42" in runs:
            r = runs["v28_lastobs_m128_h32_wo_residual_modes_seed42"]["item_scores"]
            _try_boot(bootstrap, "m128_vs_wo_residual_hard_minfde", main_rows, r, "minFDE_K_px", False, subset_key="last_observed_hard_top20")
        if "v28_lastobs_m128_h32_prior_only_seed42" in runs:
            r = runs["v28_lastobs_m128_h32_prior_only_seed42"]["item_scores"]
            _try_boot(bootstrap, "m128_vs_prior_only_hard_minfde", main_rows, r, "minFDE_K_px", False, subset_key="last_observed_hard_top20")
    if "v28_lastobs_m512_h32_seed42" in runs:
        r = runs["v28_lastobs_m512_h32_seed42"]["item_scores"]
        _try_boot(bootstrap, "m512_vs_last_all_minfde", r, baseline_rows["last_observed_copy_M512_H32"], "minFDE_K_px", False)
        _try_boot(bootstrap, "m512_vs_last_hard_minfde", r, baseline_rows["last_observed_copy_M512_H32"], "minFDE_K_px", False, subset_key="last_observed_hard_top20")
    if "v28_lastobs_m128_h64_seed42" in runs:
        r = runs["v28_lastobs_m128_h64_seed42"]["item_scores"]
        _try_boot(bootstrap, "h64_vs_last_all_minfde", r, baseline_rows["last_observed_copy_M128_H64"], "minFDE_K_px", False)
        _try_boot(bootstrap, "h64_vs_last_hard_minfde", r, baseline_rows["last_observed_copy_M128_H64"], "minFDE_K_px", False, subset_key="last_observed_hard_top20")
    dump_json(BOOTSTRAP_PATH, bootstrap)

    train_summary = {
        "audit_name": "stwm_ostf_v28_train_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "damped_gamma": r.get("damped_gamma"),
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
    dump_json(TRAIN_SUMMARY, train_summary)

    eval_summary = {
        "audit_name": "stwm_ostf_v28_eval_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "damped_gamma": r.get("damped_gamma"),
                "test_metrics": r["test_metrics"],
                "test_subset_metrics": r["test_subset_metrics"],
                "test_metrics_by_dataset": r["test_metrics_by_dataset"],
                "test_diversity_valid": r["test_diversity_valid"],
                "val_metrics": r["val_metrics"],
                "val_subset_metrics": r["val_subset_metrics"],
            }
            for name, r in runs.items()
        },
        "baselines": baselines,
        "v26_best_comparison": {
            "path": "reports/stwm_ostf_v26_eval_summary_20260502.json",
            "same_teacher_cache": True,
            "note": "V26 is included as historical CV-prior model; V28 decisions use last_observed/damped priors as the fair nonlearned hierarchy.",
        },
    }
    dump_json(EVAL_SUMMARY, eval_summary)

    decision = _decision(runs, bootstrap)
    decision["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    decision["bootstrap_highlights"] = {
        k: _fmt_boot(v)
        for k, v in bootstrap.items()
        if isinstance(v, dict) and "mean_delta" in v
    }
    dump_json(DECISION_PATH, decision)
    write_doc(
        RESULTS_DOC,
        "STWM OSTF V28 Results",
        decision,
        [
            "cache_verified",
            "strongest_prior_used",
            "H32_M128_trained",
            "H32_M512_trained",
            "H64_stress_trained",
            "V28_beats_last_observed_all_average",
            "V28_beats_last_observed_hard_subset",
            "V28_beats_damped_prior",
            "residual_modes_load_bearing",
            "dense_points_load_bearing",
            "semantic_memory_load_bearing",
            "visibility_quality_sufficient",
            "object_dense_semantic_trace_field_claim_allowed",
            "next_step_choice",
        ],
    )
    print(f"[V28][eval] wrote {EVAL_SUMMARY.relative_to(ROOT)} {BOOTSTRAP_PATH.relative_to(ROOT)} {DECISION_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
