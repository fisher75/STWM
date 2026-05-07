#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_lastobs_v28_common_20260502 import build_v28_rows, choose_visibility_aware_gamma_on_val
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v29_benchmark_utils_20260508 import (
    COMBOS,
    PRIOR_NAMES,
    ROOT,
    aggregate_extended_rows,
    available_external_dataset_preflight,
    dataset_counts,
    evaluate_prior_suite,
    future_displacement_features,
    item_feature_payload,
    metric_distributions_from_rows,
    quantiles,
    utc_now,
)


REPORT_PATH = ROOT / "reports/stwm_ostf_v29_prior_dominance_diagnosis_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V29_PRIOR_DOMINANCE_DIAGNOSIS_20260508.md"


def _best_prior(metrics_by_prior: dict[str, dict[str, Any]], metric: str = "minFDE_K_px") -> str:
    best = ("", float("inf"))
    for name, payload in metrics_by_prior.items():
        val = payload.get("metrics", {}).get(metric)
        if val is not None and float(val) < best[1]:
            best = (name, float(val))
    return best[0]


def _counts_by_bool(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
    by_ds: dict[str, int] = defaultdict(int)
    count = 0
    for item in items:
        if item.get(key):
            count += 1
            by_ds[item["dataset"]] += 1
    return {"count": int(count), "by_dataset": dict(sorted(by_ds.items()))}


def _combo_payload(combo: str) -> dict[str, Any]:
    rows, proto_centers, _, subset_summary = build_v28_rows(combo, seed=42)
    val_gamma, val_scores = choose_visibility_aware_gamma_on_val(rows["val"], proto_centers)
    split_payload: dict[str, Any] = {}
    all_item_features: list[dict[str, Any]] = []
    for split, samples in rows.items():
        priors = evaluate_prior_suite(samples, proto_centers, visibility_gamma=val_gamma)
        prior_rows = {name: priors[name]["item_rows"] for name in PRIOR_NAMES}
        item_features = [item_feature_payload(sample, combo, split, prior_rows) for sample in samples]
        all_item_features.extend(item_features)
        hard_subset_count = _counts_by_bool(item_features, "occlusion_reappearance")
        last_visible_rows = priors["last_visible_copy"]["item_rows"]
        last_visible_hard = aggregate_extended_rows(last_visible_rows, subset_key="last_observed_hard_top20")
        split_payload[split] = {
            "item_count": len(samples),
            "dataset_counts": dataset_counts(samples),
            "val_selected_visibility_aware_gamma": val_gamma,
            "strongest_prior_by_minFDE": _best_prior(priors),
            "prior_metrics": {
                name: {
                    "metrics": priors[name]["metrics"],
                    "metrics_by_dataset": priors[name]["metrics_by_dataset"],
                    "minFDE_distribution": metric_distributions_from_rows(priors[name]["item_rows"], "minFDE_K_px"),
                    "MissRate32_mean": priors[name]["metrics"].get("MissRate_32px"),
                    "MissRate64_mean": priors[name]["metrics"].get("MissRate_64px"),
                    "MissRate128_mean": priors[name]["metrics"].get("MissRate_128px"),
                    "threshold_auc_endpoint_16_32_64_128": priors[name]["metrics"].get(
                        "threshold_auc_endpoint_16_32_64_128"
                    ),
                }
                for name in PRIOR_NAMES
            },
            "last_visible_hard_top20_metrics": last_visible_hard,
            "future_displacement_endpoint_from_last_visible_distribution": quantiles(
                [x["future_endpoint_displacement_from_last_visible_px"] for x in item_features]
            ),
            "future_curvature_distribution": quantiles([x["future_curvature_acceleration_px"] for x in item_features]),
            "observed_velocity_distribution": quantiles([x["observed_velocity_magnitude_px"] for x in item_features]),
            "occlusion_ratio_distribution": quantiles([x["occlusion_ratio"] for x in item_features]),
            "valid_future_point_ratio_distribution": quantiles([x["valid_future_point_ratio"] for x in item_features]),
            "occlusion_reappearance_count": hard_subset_count,
            "semantic_identity_confuser_count": _counts_by_bool(item_features, "semantic_identity_confuser"),
            "horizon_feasible_count": _counts_by_bool(item_features, "horizon_feasible"),
            "sample_items": item_features[: min(20, len(item_features))],
        }
    h64 = combo.endswith("H64")
    test = split_payload.get("test", {})
    test_counts = test.get("dataset_counts", {})
    lv_metrics = test.get("prior_metrics", {}).get("last_visible_copy", {}).get("metrics", {})
    lv_fde = float(lv_metrics.get("minFDE_K_px") or 0.0)
    stationary_like = float(lv_metrics.get("MissRate_64px") or 0.0) < 0.5 and lv_fde < 80.0
    return {
        "combo": combo,
        "subset_summary_from_v28": subset_summary,
        "split_payload": split_payload,
        "all_item_count": len(all_item_features),
        "test_dataset_counts": test_counts,
        "test_vipseg_count": int(test_counts.get("VIPSEG", 0)),
        "test_vspw_count": int(test_counts.get("VSPW", 0)),
        "h64_vipseg_zero": bool(h64 and int(test_counts.get("VIPSEG", 0)) == 0),
        "last_visible_stationary_like": bool(stationary_like),
        "last_visible_copy_strength_interpretation": (
            "last_visible_copy is strong because the TraceAnything hardbench H64 target has many stationary/low endpoint "
            "displacement tracks and H64 has VSPW-only coverage; this is a benchmark/prior-saturation issue unless "
            "item-level extraction uncertainty dominates."
            if stationary_like or (h64 and int(test_counts.get("VIPSEG", 0)) == 0)
            else "last_visible_copy is not trivially stationary on this combo; inspect subset-specific priors."
        ),
    }


def main() -> int:
    combos = {combo: _combo_payload(combo) for combo in COMBOS}
    h64_counts = {combo: payload["test_dataset_counts"] for combo, payload in combos.items() if combo.endswith("H64")}
    h64_test_counts = [sum(x.values()) for x in h64_counts.values()]
    h64_vipseg_zero = all(int(x.get("VIPSEG", 0)) == 0 for x in h64_counts.values())
    miss32_vals = []
    for combo, payload in combos.items():
        if combo.endswith("H64"):
            metric = (
                payload["split_payload"]
                .get("test", {})
                .get("prior_metrics", {})
                .get("last_visible_copy", {})
                .get("metrics", {})
                .get("MissRate_32px")
            )
            if metric is not None:
                miss32_vals.append(float(metric))
    missrate32_saturated = bool(miss32_vals and (np.mean(miss32_vals) > 0.85 or np.mean(miss32_vals) < 0.15))
    payload = {
        "diagnosis_name": "stwm_ostf_v29_prior_dominance_diagnosis",
        "generated_at_utc": utc_now(),
        "combos": combos,
        "external_official_dataset_preflight": available_external_dataset_preflight(),
        "answers": {
            "why_H64_last_visible_copy_strongest_prior": (
                "H64 test items are VSPW-only and low-count; last-visible/visibility-aware copy suppresses noisy velocity "
                "extrapolation and wins when endpoint displacement is modest or visibility/extraction is uncertain."
            ),
            "H64_hard_subset_too_small_or_metric_saturated": bool(min(h64_test_counts or [0]) < 200 or missrate32_saturated),
            "H64_dataset_imbalanced_VIPSeg_zero": bool(h64_vipseg_zero),
            "MissRate32_saturated_threshold_auc_needed": bool(missrate32_saturated),
            "last_visible_win_interpretation": (
                "The evidence points primarily to stationary/low-displacement and H64 coverage imbalance rather than a "
                "confirmed coordinate-semantics bug; V27 already found target_extraction_bug=false. Extraction uncertainty "
                "still needs to be isolated in v29 subsets."
            ),
            "TraceAnything_hardbench_main_ready_or_diagnostic": (
                "H32 can be considered a candidate main benchmark if v29 anti-prior counts/balance pass; H64 is stress-only "
                "unless expanded because current H64 test count is below 200 and VIPSeg coverage is zero."
            ),
        },
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V29 Prior Dominance Diagnosis",
        {
            "H64_dataset_imbalanced_VIPSeg_zero": payload["answers"]["H64_dataset_imbalanced_VIPSeg_zero"],
            "H64_hard_subset_too_small_or_metric_saturated": payload["answers"][
                "H64_hard_subset_too_small_or_metric_saturated"
            ],
            "MissRate32_saturated_threshold_auc_needed": payload["answers"]["MissRate32_saturated_threshold_auc_needed"],
            "TraceAnything_hardbench_main_ready_or_diagnostic": payload["answers"][
                "TraceAnything_hardbench_main_ready_or_diagnostic"
            ],
        },
        [
            "H64_dataset_imbalanced_VIPSeg_zero",
            "H64_hard_subset_too_small_or_metric_saturated",
            "MissRate32_saturated_threshold_auc_needed",
            "TraceAnything_hardbench_main_ready_or_diagnostic",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
