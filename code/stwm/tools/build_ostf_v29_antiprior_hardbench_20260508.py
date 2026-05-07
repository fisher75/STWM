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
    PRIMARY_MANIFEST_COMBOS,
    ROOT,
    V29_MANIFEST_DIR,
    available_external_dataset_preflight,
    dataset_counts,
    dump_manifest,
    evaluate_prior_suite,
    future_displacement_features,
    item_feature_payload,
    quantiles,
    summarize_manifest_entries,
    utc_now,
)


REPORT_PATH = ROOT / "reports/stwm_ostf_v29_antiprior_hardbench_manifest_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V29_ANTIPRIOR_HARDBENCH_PROTOCOL_20260508.md"


def _thresholds_from_train_val(items: list[dict[str, Any]], combo: str) -> dict[str, Any]:
    valid = [x for x in items if x["valid_future_point_ratio"] >= 0.4]
    endpoint = np.asarray([x["future_endpoint_displacement_from_last_visible_px"] for x in valid], dtype=np.float64)
    last_visible_fde = np.asarray([x["prior_metrics"]["last_visible_copy"]["minFDE_K_px"] for x in valid], dtype=np.float64)
    nonlinear = np.asarray([x["future_curvature_acceleration_px"] for x in valid], dtype=np.float64)
    uncertainty = np.asarray([x["target_extraction_uncertainty"] for x in valid], dtype=np.float64)
    if endpoint.size == 0:
        return {"valid": False, "exact_blocker": "no train/val valid_future_point_ratio>=0.4 items"}
    # Thresholds are selected from train/val only. The quantile is lowered only if train/val itself would be too small.
    base_q = 70.0
    candidate_count = int((endpoint >= np.percentile(endpoint, base_q)).sum())
    if candidate_count < 200 and combo.endswith("H32"):
        base_q = 60.0
    if candidate_count < 80 and combo.endswith("H64"):
        base_q = 55.0
    return {
        "valid": True,
        "selection_source": "train_val_only",
        "motion_quantile": base_q,
        "endpoint_displacement_from_last_visible_px": float(np.percentile(endpoint, base_q)),
        "last_visible_minFDE_px": float(np.percentile(last_visible_fde, base_q)),
        "nonlinear_curvature_px": float(np.percentile(nonlinear, 80.0)),
        "target_extraction_uncertainty": float(np.percentile(uncertainty, 80.0)),
        "outlier_last_visible_minFDE_px": float(min(np.percentile(last_visible_fde, 99.5), 768.0)),
        "train_val_valid_count": len(valid),
    }


def _tag_item(item: dict[str, Any], thr: dict[str, Any]) -> tuple[bool, list[str], str | None]:
    if not thr.get("valid"):
        return False, [], str(thr.get("exact_blocker"))
    if item["valid_future_point_ratio"] < 0.4:
        return False, [], "valid_future_point_ratio_lt_0.4"
    last_visible_fde = float(item["prior_metrics"]["last_visible_copy"]["minFDE_K_px"])
    if last_visible_fde > float(thr["outlier_last_visible_minFDE_px"]):
        return False, [], "excluded_impossible_or_extraction_corrupt_outlier"
    tags: list[str] = []
    if item["future_endpoint_displacement_from_last_visible_px"] >= float(thr["endpoint_displacement_from_last_visible_px"]):
        tags.append("last_visible_hard")
    if last_visible_fde >= float(thr["last_visible_minFDE_px"]):
        tags.append("anti_prior_motion")
    if item["future_curvature_acceleration_px"] >= float(thr["nonlinear_curvature_px"]):
        tags.append("nonlinear_large_disp")
    if bool(item.get("occlusion_reappearance")) and item["future_endpoint_displacement_from_last_visible_px"] >= 0.5 * float(
        thr["endpoint_displacement_from_last_visible_px"]
    ):
        tags.append("occlusion_reappearance")
    if bool(item.get("semantic_identity_confuser")):
        tags.append("semantic_confuser")
    if item["target_extraction_uncertainty"] >= float(thr["target_extraction_uncertainty"]):
        tags.append("extraction_uncertainty")
    return bool(tags), sorted(set(tags)), None if tags else "not_antiprior_hard_by_train_val_thresholds"


def _manifest_entry(item: dict[str, Any], tags: list[str]) -> dict[str, Any]:
    return {
        "uid": item["uid"],
        "logical_uid": item["logical_uid"],
        "combo": item["combo"],
        "available_density_combos": [item["combo"].replace("M128", "M512")] if item["combo"].startswith("M128") else [],
        "split": item["split"],
        "dataset": item["dataset"],
        "item_key": item["item_key"],
        "object_id": item["object_id"],
        "semantic_id": item["semantic_id"],
        "source_cache_path": item["source_cache_path"],
        "H": item["H"],
        "M_manifest_reference": item["M"],
        "valid_future_point_ratio": item["valid_future_point_ratio"],
        "future_endpoint_displacement_from_last_visible_px": item["future_endpoint_displacement_from_last_visible_px"],
        "last_visible_minFDE_px": item["prior_metrics"]["last_visible_copy"]["minFDE_K_px"],
        "visibility_aware_damped_minFDE_px": item["prior_metrics"]["visibility_aware_damped"]["minFDE_K_px"],
        "target_extraction_uncertainty": item["target_extraction_uncertainty"],
        "horizon_feasible": item["horizon_feasible"],
        "v29_subset_tags": tags,
    }


def main() -> int:
    all_entries: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    subset_entries: dict[str, list[dict[str, Any]]] = {
        "test_h32_mixed": [],
        "test_h64_motion": [],
        "test_occlusion_reappearance": [],
        "test_semantic_confuser": [],
        "test_nonlinear_large_disp": [],
    }
    combo_reports: dict[str, Any] = {}
    rejected: dict[str, dict[str, int]] = {}
    for combo in PRIMARY_MANIFEST_COMBOS:
        rows, proto_centers, _, _ = build_v28_rows(combo, seed=42)
        val_gamma, _ = choose_visibility_aware_gamma_on_val(rows["val"], proto_centers)
        split_items: dict[str, list[dict[str, Any]]] = {}
        for split, samples in rows.items():
            priors = evaluate_prior_suite(samples, proto_centers, visibility_gamma=val_gamma)
            prior_rows = {name: priors[name]["item_rows"] for name in priors}
            split_items[split] = [item_feature_payload(sample, combo, split, prior_rows) for sample in samples]
        train_val = split_items["train"] + split_items["val"]
        thresholds = _thresholds_from_train_val(train_val, combo)
        combo_rejected: dict[str, int] = defaultdict(int)
        selected_by_split = {}
        for split, items in split_items.items():
            selected = []
            for item in items:
                keep, tags, reason = _tag_item(item, thresholds)
                if not keep:
                    combo_rejected[str(reason)] += 1
                    continue
                entry = _manifest_entry(item, tags)
                selected.append(entry)
                all_entries[split].append(entry)
                if split == "test":
                    if combo.endswith("H32"):
                        subset_entries["test_h32_mixed"].append(entry)
                    if combo.endswith("H64") and ("anti_prior_motion" in tags or "last_visible_hard" in tags):
                        subset_entries["test_h64_motion"].append(entry)
                    if "occlusion_reappearance" in tags:
                        subset_entries["test_occlusion_reappearance"].append(entry)
                    if "semantic_confuser" in tags:
                        subset_entries["test_semantic_confuser"].append(entry)
                    if "nonlinear_large_disp" in tags:
                        subset_entries["test_nonlinear_large_disp"].append(entry)
            selected_by_split[split] = selected
        combo_reports[combo] = {
            "thresholds": thresholds,
            "input_counts": {split: len(items) for split, items in split_items.items()},
            "input_dataset_counts": {split: dataset_counts(rows[split]) for split in rows},
            "selected_counts": {split: len(selected_by_split[split]) for split in selected_by_split},
            "selected_dataset_counts": {
                split: summarize_manifest_entries(selected_by_split[split])["by_dataset"] for split in selected_by_split
            },
            "prior_metric_distributions_on_selected_test": {
                "last_visible_minFDE_px": quantiles(
                    [e["last_visible_minFDE_px"] for e in selected_by_split.get("test", [])]
                ),
                "visibility_aware_damped_minFDE_px": quantiles(
                    [e["visibility_aware_damped_minFDE_px"] for e in selected_by_split.get("test", [])]
                ),
            },
        }
        rejected[combo] = dict(sorted(combo_rejected.items()))

    V29_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        dump_manifest(V29_MANIFEST_DIR / f"{split}.json", all_entries[split])
    for name, entries in subset_entries.items():
        dump_manifest(V29_MANIFEST_DIR / f"{name}.json", entries)

    h32_summary = summarize_manifest_entries(subset_entries["test_h32_mixed"])
    h64_summary = summarize_manifest_entries(subset_entries["test_h64_motion"])
    h32_main_ready = bool(h32_summary["item_count"] >= 100 and h32_summary["by_dataset"].get("VSPW", 0) > 0 and h32_summary["by_dataset"].get("VIPSEG", 0) > 0)
    h64_main_ready = bool(h64_summary["item_count"] >= 200 and h64_summary["by_dataset"].get("VSPW", 0) > 0 and h64_summary["by_dataset"].get("VIPSEG", 0) > 0)
    h64_stress_only = not h64_main_ready
    payload = {
        "manifest_name": "stwm_ostf_v29_antiprior_hardbench_manifest",
        "generated_at_utc": utc_now(),
        "manifest_dir": str(V29_MANIFEST_DIR.relative_to(ROOT)),
        "manifest_paths": {
            "train": str((V29_MANIFEST_DIR / "train.json").relative_to(ROOT)),
            "val": str((V29_MANIFEST_DIR / "val.json").relative_to(ROOT)),
            "test": str((V29_MANIFEST_DIR / "test.json").relative_to(ROOT)),
            **{name: str((V29_MANIFEST_DIR / f"{name}.json").relative_to(ROOT)) for name in subset_entries},
        },
        "total_item_counts": {split: len(entries) for split, entries in all_entries.items()},
        "per_dataset_counts": {split: summarize_manifest_entries(entries)["by_dataset"] for split, entries in all_entries.items()},
        "per_subset_counts": {name: summarize_manifest_entries(entries) for name, entries in subset_entries.items()},
        "combo_reports": combo_reports,
        "rejected_counts": rejected,
        "external_official_dataset_preflight": available_external_dataset_preflight(),
        "h32_main_ready": h32_main_ready,
        "h64_main_ready": h64_main_ready,
        "h64_stress_only": h64_stress_only,
        "v29_benchmark_main_ready": bool(h32_main_ready),
        "main_ready_note": (
            "H32 anti-prior split has both datasets and sufficient count; H64 remains stress-only when count<200 or VIPSeg=0."
            if h32_main_ready
            else "TraceAnything anti-prior split is diagnostic until H32 count/balance improves or external GT benchmark is integrated."
        ),
        "threshold_auc_metrics_defined": ["threshold_auc_endpoint_16_32_64_128"],
        "no_test_selection_rule": "All thresholds are selected on train/val only; test is filtered once by the frozen rule.",
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V29 Anti-Prior Hardbench Protocol",
        payload,
        [
            "manifest_dir",
            "total_item_counts",
            "per_dataset_counts",
            "per_subset_counts",
            "h32_main_ready",
            "h64_main_ready",
            "h64_stress_only",
            "v29_benchmark_main_ready",
            "main_ready_note",
            "no_test_selection_rule",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
