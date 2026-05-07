#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v29_benchmark_utils_20260508 import ROOT, load_json, utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v29_benchmark_decision_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V29_BENCHMARK_DECISION_20260508.md"
MANIFEST_PATH = ROOT / "reports/stwm_ostf_v29_antiprior_hardbench_manifest_20260508.json"
EVAL_PATH = ROOT / "reports/stwm_ostf_v29_antiprior_existing_eval_20260508.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v29_antiprior_existing_bootstrap_20260508.json"
DIAG_PATH = ROOT / "reports/stwm_ostf_v29_prior_dominance_diagnosis_20260508.json"
PREFLIGHT_PATH = ROOT / "reports/stwm_ostf_v29_prefight_from_v28_20260508.json"


def _positive(boot: dict[str, Any]) -> bool:
    return bool(boot.get("zero_excluded") and boot.get("mean_delta") is not None and float(boot["mean_delta"]) > 0.0)


def _metric(eval_payload: dict[str, Any], combo: str, prior: str, subset: str, metric: str) -> float | None:
    try:
        val = eval_payload["priors"][combo][prior]["subset_metrics"][subset][metric]
    except Exception:
        return None
    return float(val) if val is not None else None


def _best_prior(eval_payload: dict[str, Any], combo: str, subset: str) -> str | None:
    best: tuple[str | None, float] = (None, float("inf"))
    for prior, payload in eval_payload.get("priors", {}).get(combo, {}).items():
        val = payload.get("subset_metrics", {}).get(subset, {}).get("minFDE_K_px")
        if val is not None and float(val) < best[1]:
            best = (prior, float(val))
    return best[0]


def main() -> int:
    manifest = load_json(MANIFEST_PATH)
    eval_payload = load_json(EVAL_PATH)
    boot = load_json(BOOT_PATH)
    diag = load_json(DIAG_PATH)
    preflight = load_json(PREFLIGHT_PATH)
    h32_main_ready = bool(manifest.get("h32_main_ready", False))
    h64_main_ready = bool(manifest.get("h64_main_ready", False))
    h64_stress_only = bool(manifest.get("h64_stress_only", not h64_main_ready))
    h32_counts = manifest.get("per_subset_counts", {}).get("test_h32_mixed", {})
    h64_counts = manifest.get("per_subset_counts", {}).get("test_h64_motion", {})
    h64_vipseg_zero = int(h64_counts.get("by_dataset", {}).get("VIPSEG", 0)) == 0
    dataset_balance_ok = bool(
        h32_counts.get("by_dataset", {}).get("VSPW", 0) > 0
        and h32_counts.get("by_dataset", {}).get("VIPSEG", 0) > 0
        and (not h64_main_ready or h64_counts.get("by_dataset", {}).get("VIPSEG", 0) > 0)
    )
    best_h32 = _best_prior(eval_payload, "M128_H32", "last_visible_hard")
    best_h64 = _best_prior(eval_payload, "M128_H64", "last_visible_hard")
    comparisons = boot.get("comparisons", {})
    h32_v28_beats_last_visible = _positive(comparisons.get("V28_H32_best_available_vs_last_visible_copy_last_visible_hard_minFDE", {}))
    h64_v28_beats_last_visible = _positive(comparisons.get("V28_H64_best_available_vs_last_visible_copy_last_visible_hard_minFDE", {}))
    last_visible_prior_dominates_after_fix = bool(
        (best_h32 == "last_visible_copy" and not h32_v28_beats_last_visible)
        or (best_h64 == "last_visible_copy" and not h64_v28_beats_last_visible)
    )
    miss32_saturated = bool(
        diag.get("answers", {}).get("MissRate32_saturated_threshold_auc_needed", False)
        or (
            (_metric(eval_payload, "M128_H64", "last_visible_copy", "last_visible_hard", "MissRate_32px") in {0.0, 1.0})
            if _metric(eval_payload, "M128_H64", "last_visible_copy", "last_visible_hard", "MissRate_32px") is not None
            else False
        )
    )
    threshold_auc_needed = bool(miss32_saturated or manifest.get("threshold_auc_metrics_defined"))
    extraction_artifact_flag = bool(
        diag.get("answers", {}).get("last_visible_win_interpretation", "").lower().find("extraction") >= 0
        and manifest.get("per_subset_counts", {}).get("test_occlusion_reappearance", {}).get("item_count", 0) == 0
    )
    external_gt_dataset_needed = bool(h64_vipseg_zero or h64_stress_only or last_visible_prior_dominates_after_fix)

    if h32_main_ready and dataset_balance_ok and not last_visible_prior_dominates_after_fix:
        recommended = "train_v29_on_antiprior_h32_h64"
    elif h64_vipseg_zero or h64_stress_only or (
        last_visible_prior_dominates_after_fix and "stationary" in diag.get("answers", {}).get("last_visible_win_interpretation", "")
    ):
        recommended = "integrate_PointOdyssey_TAPVid3D_GT"
    elif extraction_artifact_flag:
        recommended = "fix_visibility_or_target_extraction"
    elif manifest.get("total_item_counts", {}).get("train", 0) > 0:
        recommended = "expand_traceanything_cache_to_1k_motion_clips"
    else:
        recommended = "pause_OSTF_return_to_FSTF_backup"

    payload = {
        "decision_name": "stwm_ostf_v29_benchmark_decision",
        "generated_at_utc": utc_now(),
        "v28_prefight_complete": bool(preflight.get("required_artifacts_complete")),
        "v29_benchmark_main_ready": bool(manifest.get("v29_benchmark_main_ready", False) and not h64_main_ready is False),
        "h32_main_ready": h32_main_ready,
        "h64_main_ready": h64_main_ready,
        "h64_stress_only": h64_stress_only,
        "h64_vipseg_zero": h64_vipseg_zero,
        "last_visible_prior_dominates_after_fix": last_visible_prior_dominates_after_fix,
        "best_prior_h32_last_visible_hard": best_h32,
        "best_prior_h64_last_visible_hard": best_h64,
        "V28_H32_beats_last_visible_hard_minFDE": h32_v28_beats_last_visible,
        "V28_H64_beats_last_visible_hard_minFDE": h64_v28_beats_last_visible,
        "missrate32_saturated": miss32_saturated,
        "threshold_auc_needed": threshold_auc_needed,
        "dataset_balance_ok": dataset_balance_ok,
        "external_gt_dataset_needed": external_gt_dataset_needed,
        "recommended_next_step": recommended,
        "decision_rationale": (
            "H64 remains stress-only and/or last_visible_copy still dominates key anti-prior subsets; do not train a new "
            "OSTF-v2 model until the benchmark route is fixed with external GT or expanded motion clips."
            if recommended != "train_v29_on_antiprior_h32_h64"
            else "Anti-prior benchmark passes count/balance and prior-dominance checks under the frozen construction rule."
        ),
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V29 Benchmark Decision",
        payload,
        [
            "v29_benchmark_main_ready",
            "h32_main_ready",
            "h64_main_ready",
            "h64_stress_only",
            "last_visible_prior_dominates_after_fix",
            "missrate32_saturated",
            "threshold_auc_needed",
            "dataset_balance_ok",
            "external_gt_dataset_needed",
            "recommended_next_step",
            "decision_rationale",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
