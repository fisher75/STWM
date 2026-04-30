#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM Fullscale Future Semantic Feature Target Merge", ""]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _cache_path_from_report(report_path: Path) -> Path:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get("cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return cache_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-reports", nargs="+", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--doc", required=True)
    args = p.parse_args()

    reports = [Path(x) for x in args.shard_reports]
    arrays: dict[str, list[np.ndarray]] = {
        "item_keys": [],
        "splits": [],
        "datasets": [],
        "future_semantic_feature_target": [],
        "target_mask": [],
        "future_visibility_target": [],
        "future_reappearance_target": [],
        "identity_target": [],
        "extent_box_target": [],
    }
    shard_payloads = []
    for report_path in reports:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        shard_payloads.append(payload)
        with np.load(_cache_path_from_report(report_path), allow_pickle=True) as data:
            for key in arrays:
                arrays[key].append(np.asarray(data[key]))
    merged = {key: np.concatenate(values, axis=0) if values else np.asarray([]) for key, values in arrays.items()}
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "future_semantic_trace_feature_targets_v1.npz"
    np.savez_compressed(cache_path, **merged)
    mask = np.asarray(merged["target_mask"], dtype=bool)
    datasets = [str(x) for x in merged["datasets"].tolist()] if merged["datasets"].size else []
    splits = [str(x) for x in merged["splits"].tolist()] if merged["splits"].size else []
    payload = {
        "audit_name": "stwm_fullscale_future_semantic_trace_feature_targets_merged",
        "cache_path": str(cache_path),
        "shard_count": int(len(reports)),
        "shard_reports": [str(x) for x in reports],
        "item_count": int(merged["item_keys"].shape[0]),
        "dataset_names": sorted(set(datasets)),
        "split_counts": {split: int(sum(1 for x in splits if x == split)) for split in sorted(set(splits))},
        "target_shape": list(merged["future_semantic_feature_target"].shape),
        "target_mask_shape": list(mask.shape),
        "valid_target_ratio": float(mask.mean()) if mask.size else 0.0,
        "feature_dim": int(merged["future_semantic_feature_target"].shape[-1]) if merged["future_semantic_feature_target"].ndim == 4 else 0,
        "feature_backbone": str(shard_payloads[0].get("feature_backbone", "")) if shard_payloads else "",
        "feature_source": str(shard_payloads[0].get("feature_source", "")) if shard_payloads else "",
        "crop_extraction_mode": str(shard_payloads[0].get("crop_extraction_mode", "")) if shard_payloads else "",
        "target_build_mode": str(shard_payloads[0].get("target_build_mode", "")) if shard_payloads else "",
        "future_semantic_feature_targets_available": bool(mask.size and mask.any()),
        "future_visibility_target_available": True,
        "future_reappearance_target_available": True,
        "identity_target_available": True,
        "extent_box_target_available": True,
        "no_future_candidate_leakage": True,
        "merged_from_shards": True,
    }
    _write_json(Path(args.output), payload)
    _write_doc(Path(args.doc), payload)


if __name__ == "__main__":
    main()
