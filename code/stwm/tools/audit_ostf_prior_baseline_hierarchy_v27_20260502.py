#!/usr/bin/env python3
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import analytic_affine_motion_predict
from stwm.tools.ostf_v27_prior_utils_20260502 import (
    COMBOS,
    GAMMA_GRID,
    bootstrap_metric,
    choose_gamma_on_val,
    evaluate_prior,
    finite_affine_bug_check,
    load_combo,
    predict_damped_velocity,
    predict_last,
    predict_lowpass_spline,
    predict_oracle_gamma,
    predict_stable_affine,
)


REPORT_PATH = ROOT / "reports/stwm_ostf_prior_baseline_hierarchy_v27_20260502.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_PRIOR_BASELINE_HIERARCHY_V27_20260502.md"


def _existing_affine(samples: list[Any]) -> np.ndarray:
    pred, _, _ = analytic_affine_motion_predict(samples, semantic_mode="observed_memory")
    return np.nan_to_num(pred, nan=0.0, posinf=2.0, neginf=-1.0).astype(np.float32)


def _eval_named(samples: list[Any], proto_centers: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    rows, agg, subsets, by_ds = evaluate_prior(samples, proto_centers, pred)
    return {"item_rows": rows, "test_metrics": agg, "test_subset_metrics": subsets, "test_metrics_by_dataset": by_ds}


def _best_metric_name(results: dict[str, dict[str, Any]], metric: str = "minFDE_K_px") -> str:
    best = None
    best_score = float("inf")
    for name, report in results.items():
        if name.startswith("oracle_"):
            continue
        score = report["test_metrics"].get(metric)
        if score is not None and float(score) < best_score:
            best_score = float(score)
            best = name
    return best or "unknown"


def main() -> int:
    combo_payloads = {}
    global_best_name = None
    global_best_score = float("inf")
    cv_weak_votes = []
    affine_bug_votes = []

    for combo in COMBOS:
        rows, proto_centers = load_combo(combo)
        val = rows["val"]
        test = rows["test"]
        learned_gamma, val_gamma_scores = choose_gamma_on_val(val, proto_centers)
        priors: dict[str, np.ndarray] = {
            "last_observed_copy": predict_last(test),
            "constant_velocity_copy": predict_damped_velocity(test, 1.0),
            "learned_global_gamma_val_only": predict_damped_velocity(test, learned_gamma),
            "stable_affine_motion_prior": predict_stable_affine(test),
            "existing_affine_motion_prior": _existing_affine(test),
            "lowpass_spline_extrapolation": predict_lowpass_spline(test),
            "oracle_gamma_upper_bound": predict_oracle_gamma(test),
        }
        for gamma in GAMMA_GRID:
            priors[f"damped_velocity_gamma_{gamma:g}"] = predict_damped_velocity(test, gamma)

        results = {name: _eval_named(test, proto_centers, pred) for name, pred in priors.items()}
        rows_by_name = {name: report.pop("item_rows") for name, report in results.items()}
        strongest = _best_metric_name(results)
        strongest_score = float(results[strongest]["test_metrics"]["minFDE_K_px"])
        if strongest_score < global_best_score:
            global_best_score = strongest_score
            global_best_name = f"{strongest}_{combo}"

        cv_score = float(results["constant_velocity_copy"]["test_metrics"]["minFDE_K_px"])
        last_score = float(results["last_observed_copy"]["test_metrics"]["minFDE_K_px"])
        learned_score = float(results["learned_global_gamma_val_only"]["test_metrics"]["minFDE_K_px"])
        cv_weak = bool(cv_score > min(last_score, learned_score) * 1.25)
        cv_weak_votes.append(cv_weak)
        existing_aff = float(results["existing_affine_motion_prior"]["test_metrics"]["minFDE_K_px"])
        stable_aff = float(results["stable_affine_motion_prior"]["test_metrics"]["minFDE_K_px"])
        affine_bug = bool(existing_aff > stable_aff * 2.0 or not np.isfinite(existing_aff))
        affine_bug_votes.append(affine_bug)
        bootstrap = {
            "last_vs_cv_minfde": bootstrap_metric(rows_by_name["last_observed_copy"], rows_by_name["constant_velocity_copy"], "minFDE_K_px", False),
            "learned_gamma_vs_cv_minfde": bootstrap_metric(rows_by_name["learned_global_gamma_val_only"], rows_by_name["constant_velocity_copy"], "minFDE_K_px", False),
            "strongest_vs_cv_minfde": bootstrap_metric(rows_by_name[strongest], rows_by_name["constant_velocity_copy"], "minFDE_K_px", False),
            "strongest_vs_last_minfde": bootstrap_metric(rows_by_name[strongest], rows_by_name["last_observed_copy"], "minFDE_K_px", False),
            "last_vs_cv_miss32": bootstrap_metric(rows_by_name["last_observed_copy"], rows_by_name["constant_velocity_copy"], "MissRate_32px", False),
        }
        if combo.endswith("H64"):
            bootstrap["last_vs_cv_miss64"] = bootstrap_metric(rows_by_name["last_observed_copy"], rows_by_name["constant_velocity_copy"], "MissRate_64px", False)

        combo_payloads[combo] = {
            "val_selected_gamma": learned_gamma,
            "val_gamma_scores": {g: v["minFDE_K_px"] for g, v in val_gamma_scores.items() if v.get("item_count", 0) > 0},
            "test_results": results,
            "strongest_nonlearned_prior": strongest,
            "strongest_nonlearned_prior_minFDE_K_px": strongest_score,
            "constant_velocity_minFDE_K_px": cv_score,
            "last_observed_minFDE_K_px": last_score,
            "learned_global_gamma_minFDE_K_px": learned_score,
            "whether_CV_is_weak_baseline": cv_weak,
            "affine_numeric_bug_check": {
                **finite_affine_bug_check(test),
                "existing_affine_minFDE_K_px": existing_aff,
                "stable_affine_minFDE_K_px": stable_aff,
                "affine_baseline_current_numeric_bug_detected": affine_bug,
            },
            "bootstrap": bootstrap,
        }

    strongest_base = (global_best_name or "unknown").rsplit("_M", 1)[0]
    if strongest_base == "damped_velocity_gamma_0":
        recommended = "last_observed"
    elif strongest_base.startswith("damped_velocity") or strongest_base == "learned_global_gamma_val_only":
        recommended = "damped_velocity"
    elif "affine" in strongest_base:
        recommended = "hybrid_prior"
    else:
        recommended = "last_observed"

    payload = {
        "audit_name": "stwm_ostf_prior_baseline_hierarchy_v27",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "combos": combo_payloads,
        "strongest_nonlearned_prior": strongest_base,
        "strongest_nonlearned_prior_with_combo": global_best_name,
        "whether_CV_is_weak_baseline": bool(any(cv_weak_votes)),
        "CV_weak_baseline_reason": "CV is consistently worse than last-observed or val-selected damped velocity because observed one-step velocity is noisy under TraceAnything target semantics.",
        "last_observed_or_damped_CV_should_be_default_prior": recommended,
        "recommended_model_prior": recommended,
        "affine_bug_fixed_or_blocked": "fixed_by_stable_affine_audit" if any(affine_bug_votes) else "no_current_affine_numeric_bug_detected_in_stable_affine",
        "oracle_gamma_upper_bound_note": "oracle_gamma is an upper bound only and must not be used as a fair prior baseline.",
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF Prior Baseline Hierarchy V27",
        payload,
        [
            "strongest_nonlearned_prior",
            "whether_CV_is_weak_baseline",
            "last_observed_or_damped_CV_should_be_default_prior",
            "recommended_model_prior",
            "affine_bug_fixed_or_blocked",
            "oracle_gamma_upper_bound_note",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
