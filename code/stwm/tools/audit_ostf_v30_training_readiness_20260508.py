#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.datasets.ostf_v30_external_gt_dataset_20260508 import dataset_summary
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT_PATH = ROOT / "reports/stwm_ostf_v30_training_readiness_audit_20260508.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V30_TRAINING_READINESS_AUDIT_20260508.md"

REQUIRED_FILES = [
    "code/stwm/tools/ostf_v30_external_gt_schema_20260508.py",
    "code/stwm/tools/audit_ostf_v30_external_gt_data_roots_20260508.py",
    "code/stwm/tools/build_ostf_v30_pointodyssey_gt_cache_20260508.py",
    "code/stwm/tools/build_ostf_v30_tapvid_gt_cache_20260508.py",
    "code/stwm/tools/build_ostf_v30_tapvid3d_gt_cache_20260508.py",
    "code/stwm/tools/build_ostf_v30_external_gt_antiprior_benchmark_20260508.py",
    "code/stwm/tools/eval_ostf_v30_external_gt_priors_and_existing_20260508.py",
    "code/stwm/tools/write_ostf_v30_external_gt_decision_20260508.py",
    "scripts/run_ostf_v30_external_gt_preflight_20260508.sh",
    "scripts/start_ostf_v30_external_gt_preflight_tmux_20260508.sh",
    "reports/stwm_ostf_v30_v29_bugfix_audit_20260508.json",
    "reports/stwm_ostf_v30_external_gt_data_root_audit_20260508.json",
    "reports/stwm_ostf_v30_external_gt_cache_build_20260508.json",
    "reports/stwm_ostf_v30_external_gt_antiprior_manifest_20260508.json",
    "reports/stwm_ostf_v30_external_gt_existing_eval_20260508.json",
    "reports/stwm_ostf_v30_external_gt_existing_bootstrap_20260508.json",
    "reports/stwm_ostf_v30_external_gt_decision_20260508.json",
]


def _read_json(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _sample_npz(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    return {
        "path": str(path.relative_to(ROOT)),
        "keys": sorted(z.files),
        "obs_points_shape": list(np.asarray(z["obs_points"]).shape),
        "fut_points_shape": list(np.asarray(z["fut_points"]).shape),
        "obs_vis_shape": list(np.asarray(z["obs_vis"]).shape),
        "fut_vis_shape": list(np.asarray(z["fut_vis"]).shape),
        "has_2d": "obs_points" in z and "fut_points" in z,
        "has_3d": "obs_points_3d" in z and "fut_points_3d" in z,
        "has_visibility": "obs_vis" in z and "fut_vis" in z,
        "no_future_leakage": bool(np.asarray(z["no_future_leakage"]).item()) if "no_future_leakage" in z else False,
        "source_gt_not_teacher": bool(np.asarray(z["source_gt_not_teacher"]).item()) if "source_gt_not_teacher" in z else False,
    }


def main() -> int:
    files = {f: {"exists": (ROOT / f).exists(), "size_bytes": (ROOT / f).stat().st_size if (ROOT / f).exists() else 0} for f in REQUIRED_FILES}
    missing = [f for f, rec in files.items() if not rec["exists"] or rec["size_bytes"] <= 0]
    manifest = _read_json("reports/stwm_ostf_v30_external_gt_antiprior_manifest_20260508.json")
    cache = _read_json("reports/stwm_ostf_v30_external_gt_cache_build_20260508.json")
    decision = _read_json("reports/stwm_ostf_v30_external_gt_decision_20260508.json")
    sample_path = next((ROOT / "outputs/cache/stwm_ostf_v30_external_gt/pointodyssey/M128_H32/train").glob("*.npz"))
    sample = _sample_npz(sample_path)
    pointodyssey = cache.get("datasets", {}).get("pointodyssey", {})
    combo_counts = pointodyssey.get("combo_counts", {})
    video_to_split: dict[str, str] = {}
    leakage = []
    for split in ("train", "val", "test"):
        for entry in json.loads((ROOT / f"manifests/ostf_v30_external_gt/{split}.json").read_text(encoding="utf-8")).get("entries", []):
            uid = str(entry["uid"]).split("_M")[0]
            if uid in video_to_split and video_to_split[uid] != split:
                leakage.append({"uid": uid, "first_split": video_to_split[uid], "second_split": split})
            video_to_split[uid] = split
    recommended = ["M128_H32_seed42", "M128_H64_seed42", "M128_H32_wo_semantic_seed42", "M128_H64_wo_semantic_seed42"]
    if combo_counts.get("M512_H32", {}).get("train", 0) and combo_counts.get("M512_H64", {}).get("train", 0):
        recommended += ["M512_H32_seed42_optional", "M512_H64_seed42_optional"]
    payload = {
        "audit_name": "stwm_ostf_v30_training_readiness_audit",
        "generated_at_utc": utc_now(),
        "files_checked": files,
        "missing_or_empty_files": missing,
        "PointOdyssey_cache_path": str((ROOT / "outputs/cache/stwm_ostf_v30_external_gt/pointodyssey").relative_to(ROOT)),
        "total_raw_cache_items": manifest.get("total_raw_cache_items"),
        "H32_H64_H96_split_counts": {
            "H32_motion_test": manifest.get("selected_counts", {}).get("test_h32_motion", {}).get("item_count"),
            "H64_motion_test": manifest.get("selected_counts", {}).get("test_h64_motion", {}).get("item_count"),
            "H96_motion_test": manifest.get("selected_counts", {}).get("test_h96_motion", {}).get("item_count"),
        },
        "M128_M512_M1024_availability": {combo: counts for combo, counts in combo_counts.items()},
        "fields_sample": sample,
        "two_d_fields_available": bool(sample["has_2d"]),
        "three_d_fields_available": bool(sample["has_3d"]),
        "obs_fut_visibility_fields_available": bool(sample["has_visibility"]),
        "train_val_test_video_level_leakage_check": {
            "checked_video_uid_count": len(video_to_split),
            "leakage_detected": bool(leakage),
            "leakage_examples": leakage[:10],
        },
        "external_GT_benchmark_main_ready": bool(
            decision.get("h32_external_gt_main_ready") and decision.get("h64_external_gt_main_ready")
        ),
        "existing_V28_incompatible": bool(not decision.get("existing_v28_compatible")),
        "existing_V28_incompatibility_reason": _read_json("reports/stwm_ostf_v30_external_gt_existing_eval_20260508.json").get(
            "existing_v28_incompatibility_reason"
        ),
        "V30_new_model_training_required": bool(decision.get("benchmark_requires_training_new_model")),
        "exact_training_combos_recommended": recommended,
        "readiness_passed": bool(not missing and not leakage and decision.get("external_gt_cache_ready") and decision.get("h32_external_gt_main_ready") and decision.get("h64_external_gt_main_ready")),
        "dataset_summaries": {
            "M128_H32": dataset_summary(32, 128),
            "M128_H64": dataset_summary(64, 128),
        },
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF V30 Training Readiness Audit",
        payload,
        [
            "readiness_passed",
            "PointOdyssey_cache_path",
            "total_raw_cache_items",
            "H32_H64_H96_split_counts",
            "M128_M512_M1024_availability",
            "two_d_fields_available",
            "three_d_fields_available",
            "obs_fut_visibility_fields_available",
            "train_val_test_video_level_leakage_check",
            "external_GT_benchmark_main_ready",
            "existing_V28_incompatible",
            "V30_new_model_training_required",
            "exact_training_combos_recommended",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if payload["readiness_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
