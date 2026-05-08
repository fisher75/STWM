#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import (
    EXTERNAL_MANIFEST_DIR,
    ROOT,
    aggregate_metric_rows,
    discover_external_cache_files,
    load_external_sample,
    point_metrics,
    prior_predictions,
    utc_now,
)


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_antiprior_manifest_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_ANTIPRIOR_PROTOCOL_20260508.md"
MAIN_READY_DATASETS = {"pointodyssey"}


def _entry(path: Path) -> dict[str, Any]:
    sample = load_external_sample(path)
    priors = prior_predictions(sample, gamma=0.0)
    metrics = {name: point_metrics(sample, pred) for name, pred in priors.items()}
    lv = metrics["last_visible_copy"]
    last = metrics["last_observed_copy"]
    fut_disp = lv["minFDE"]
    return {
        "uid": sample.uid,
        "dataset": sample.dataset,
        "split": sample.split,
        "cache_path": str(path.relative_to(ROOT)),
        "M": int(sample.obs_points.shape[0]),
        "H": int(sample.fut_points.shape[1]),
        "coordinate_system": sample.coordinate_system,
        "valid_future_point_ratio": float(sample.fut_vis.mean()) if sample.fut_vis.size else 0.0,
        "last_visible_minFDE": lv["minFDE"],
        "last_observed_minFDE": last["minFDE"],
        "constant_velocity_minFDE": metrics["constant_velocity"]["minFDE"],
        "fixed_affine_minFDE": metrics["fixed_affine"]["minFDE"],
        "last_visible_MissRate32": lv["MissRate@32"],
        "last_visible_MissRate64": lv["MissRate@64"],
        "last_visible_MissRate128": lv["MissRate@128"],
        "threshold_auc_endpoint_16_32_64_128": lv["threshold_auc_endpoint_16_32_64_128"],
        "occlusion_ratio": float(1.0 - sample.fut_vis.mean()),
        "future_endpoint_displacement_score": float(fut_disp),
        "prior_metrics": metrics,
    }


def _thresholds(entries: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [e for e in entries if e["valid_future_point_ratio"] >= 0.4]
    if not usable:
        return {"valid": False, "exact_blocker": "no train/val usable entries"}
    fde = np.asarray([e["last_visible_minFDE"] for e in usable], dtype=np.float64)
    cv_gap = np.asarray([e["constant_velocity_minFDE"] - e["last_visible_minFDE"] for e in usable], dtype=np.float64)
    occ = np.asarray([e["occlusion_ratio"] for e in usable], dtype=np.float64)
    return {
        "valid": True,
        "selection_source": "train_val_only",
        "last_visible_minFDE_motion_threshold": float(np.percentile(fde, 60)),
        "last_visible_minFDE_outlier_threshold": float(min(np.percentile(fde, 99.5), np.percentile(fde, 95) * 3.0 + 1e-6)),
        "cv_gap_threshold": float(np.percentile(cv_gap, 60)),
        "occlusion_threshold": float(np.percentile(occ, 75)),
    }


def _tag(e: dict[str, Any], thr: dict[str, Any]) -> tuple[bool, list[str], str | None]:
    if e["dataset"] not in MAIN_READY_DATASETS:
        return False, [], "excluded_partial_or_diagnostic_external_gt_source"
    if e["valid_future_point_ratio"] < 0.4:
        return False, [], "valid_future_point_ratio_lt_0.4"
    if not thr.get("valid"):
        return False, [], str(thr.get("exact_blocker"))
    if e["last_visible_minFDE"] > thr["last_visible_minFDE_outlier_threshold"]:
        return False, [], "excluded_extraction_or_extreme_outlier"
    tags = []
    if e["last_visible_minFDE"] >= thr["last_visible_minFDE_motion_threshold"]:
        tags.append("motion")
    if e["constant_velocity_minFDE"] - e["last_visible_minFDE"] >= thr["cv_gap_threshold"]:
        tags.append("anti_cv")
    if e["occlusion_ratio"] >= thr["occlusion_threshold"] and e["last_visible_minFDE"] >= 0.5 * thr["last_visible_minFDE_motion_threshold"]:
        tags.append("occlusion_reappearance")
    if e["last_visible_minFDE"] >= 1.5 * thr["last_visible_minFDE_motion_threshold"]:
        tags.append("nonlinear_large_disp")
    if e["H"] >= 96:
        tags.append("long_gap")
    return bool(tags), sorted(set(tags)), None if tags else "below_antiprior_thresholds"


def _write_manifest(name: str, entries: list[dict[str, Any]]) -> None:
    EXTERNAL_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = EXTERNAL_MANIFEST_DIR / f"{name}.json"
    path.write_text(
        json.dumps({"manifest_name": name, "generated_at_utc": utc_now(), "item_count": len(entries), "entries": entries}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset = defaultdict(int)
    by_h = defaultdict(int)
    by_m = defaultdict(int)
    for e in entries:
        by_dataset[e["dataset"]] += 1
        by_h[f"H{e['H']}"] += 1
        by_m[f"M{e['M']}"] += 1
    return {"item_count": len(entries), "by_dataset": dict(sorted(by_dataset.items())), "by_horizon": dict(sorted(by_h.items())), "by_M": dict(sorted(by_m.items()))}


def main() -> int:
    all_entries = [_entry(path) for path in discover_external_cache_files()]
    split_raw = {split: [e for e in all_entries if e["split"] == split] for split in ("train", "val", "test")}
    thr = _thresholds(split_raw["train"] + split_raw["val"])
    manifests = {"train": [], "val": [], "test": [], "test_h32_motion": [], "test_h64_motion": [], "test_h96_motion": [], "test_occlusion_reappearance": [], "test_nonlinear_large_disp": [], "test_long_gap": []}
    rejected = defaultdict(int)
    for split, entries in split_raw.items():
        for e in entries:
            keep, tags, reason = _tag(e, thr)
            if not keep:
                rejected[str(reason)] += 1
                continue
            e = dict(e)
            e["v30_subset_tags"] = tags
            manifests[split].append(e)
            if split == "test":
                if e["H"] == 32 and "motion" in tags:
                    manifests["test_h32_motion"].append(e)
                if e["H"] == 64 and "motion" in tags:
                    manifests["test_h64_motion"].append(e)
                if e["H"] == 96 and "motion" in tags:
                    manifests["test_h96_motion"].append(e)
                if "occlusion_reappearance" in tags:
                    manifests["test_occlusion_reappearance"].append(e)
                if "nonlinear_large_disp" in tags:
                    manifests["test_nonlinear_large_disp"].append(e)
                if "long_gap" in tags:
                    manifests["test_long_gap"].append(e)
    for name, entries in manifests.items():
        _write_manifest(name, entries)
    h32 = _summary(manifests["test_h32_motion"])
    h64 = _summary(manifests["test_h64_motion"])
    h96 = _summary(manifests["test_h96_motion"])
    h32_ready = h32["item_count"] >= 200
    h64_ready = h64["item_count"] >= 200
    h96_ready = h96["item_count"] >= 100
    single_source = len(set(e["dataset"] for e in manifests["test"])) < 2 if manifests["test"] else True
    payload = {
        "manifest_name": "stwm_ostf_v30_external_gt_antiprior_manifest",
        "generated_at_utc": utc_now(),
        "manifest_dir": str(EXTERNAL_MANIFEST_DIR.relative_to(ROOT)),
        "thresholds": thr,
        "total_raw_cache_items": len(all_entries),
        "selected_counts": {name: _summary(entries) for name, entries in manifests.items()},
        "rejected_counts": dict(sorted(rejected.items())),
        "h32_external_gt_main_ready": bool(h32_ready),
        "h64_external_gt_main_ready": bool(h64_ready),
        "h96_external_gt_main_ready": bool(h96_ready),
        "single_source_only": bool(single_source),
        "main_ready_datasets": sorted(MAIN_READY_DATASETS),
        "excluded_partial_sources_are_diagnostic_only": True,
        "cross_domain_main_ready": bool((h32_ready or h64_ready) and not single_source),
        "pointodyssey_only_diagnostic": False,
        "missrate32_saturation_handling": "Use threshold_auc_endpoint_16_32_64_128 plus MissRate@64/128 when MissRate@32 is non-discriminative.",
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 External GT Anti-Prior Protocol",
        payload,
        [
            "manifest_dir",
            "total_raw_cache_items",
            "selected_counts",
            "h32_external_gt_main_ready",
            "h64_external_gt_main_ready",
            "h96_external_gt_main_ready",
            "single_source_only",
            "cross_domain_main_ready",
            "pointodyssey_only_diagnostic",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
