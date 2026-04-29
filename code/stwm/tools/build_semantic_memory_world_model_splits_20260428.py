#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_report_npz(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get("target_cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = path.parent.parent / cache_path
    return payload, dict(np.load(cache_path, allow_pickle=True))


def _load_observed_npz(observed_report: Path, c: int) -> dict[str, np.ndarray]:
    payload = json.loads(observed_report.read_text(encoding="utf-8"))
    cache_path = Path(str(payload["target_cache_paths_by_prototype_count"][str(c)]))
    if not cache_path.is_absolute():
        cache_path = observed_report.parent.parent / cache_path
    return dict(np.load(cache_path, allow_pickle=True))


def _stable_hash(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:12], 16)


def _stats_for_keys(keys: list[str], future: dict[str, np.ndarray], observed: dict[str, np.ndarray]) -> dict[str, Any]:
    index = {str(k): i for i, k in enumerate(future["item_keys"].tolist())}
    obs_index = {str(k): i for i, k in enumerate(observed["item_keys"].tolist())}
    changed = stable = future_valid = observed_valid = overlap = 0
    datasets: dict[str, int] = {}
    for key in keys:
        idx = index.get(key)
        oidx = obs_index.get(key)
        if idx is None or oidx is None:
            continue
        ds = str(key.split("::", 1)[0]) if "::" in key else "unknown"
        datasets[ds] = datasets.get(ds, 0) + 1
        future_target = np.asarray(future["future_semantic_proto_target"][idx], dtype=np.int64)
        future_mask = np.asarray(future["target_mask"][idx], dtype=bool) & (future_target >= 0)
        obs_target = np.asarray(observed["observed_semantic_proto_target"][oidx], dtype=np.int64)
        obs_mask = np.asarray(observed["observed_semantic_proto_mask"][oidx], dtype=bool) & (obs_target >= 0)
        mask = future_mask & obs_mask[None, :]
        ch = mask & (future_target != obs_target[None, :])
        changed += int(ch.sum())
        stable += int((mask & (~ch)).sum())
        future_valid += int(future_mask.sum())
        observed_valid += int(obs_mask.sum())
        overlap += int(mask.sum())
    return {
        "item_count": int(len(keys)),
        "dataset_counts": datasets,
        "changed_count": changed,
        "stable_count": stable,
        "changed_ratio": float(changed / max(changed + stable, 1)),
        "observed_coverage": float(observed_valid / max(len(keys) * int(observed["observed_semantic_proto_mask"].shape[1]), 1)),
        "target_coverage": float(overlap / max(future_valid, 1)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--observed-report", default="reports/stwm_observed_semantic_prototype_targets_v2_20260428.json")
    p.add_argument("--future-report-c32", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c32_20260428.json")
    p.add_argument("--future-report-c64", default="reports/stwm_future_semantic_trace_prototype_targets_v2_c64_20260428.json")
    p.add_argument("--output", default="reports/stwm_semantic_memory_world_model_v2_splits_20260428.json")
    p.add_argument("--doc", default="docs/STWM_SEMANTIC_MEMORY_WORLD_MODEL_V2_SPLITS_20260428.md")
    args = p.parse_args()
    observed_report = Path(args.observed_report)
    future_payload32, future32 = _load_report_npz(Path(args.future_report_c32))
    _, future64 = _load_report_npz(Path(args.future_report_c64))
    observed32 = _load_observed_npz(observed_report, 32)
    observed64 = _load_observed_npz(observed_report, 64)
    obs_index = {str(k): i for i, k in enumerate(observed32["item_keys"].tolist())}
    eligible: list[str] = []
    for key in [str(x) for x in future32["item_keys"].tolist()]:
        idx = obs_index.get(key)
        if idx is None:
            continue
        if bool(np.asarray(observed32["observed_semantic_proto_mask"][idx], dtype=bool).any()):
            eligible.append(key)
    by_dataset: dict[str, list[str]] = {}
    for key in eligible:
        ds = key.split("::", 1)[0] if "::" in key else "unknown"
        by_dataset.setdefault(ds, []).append(key)
    splits = {"train": [], "val": [], "test": []}
    for ds, keys in by_dataset.items():
        keys = sorted(keys, key=_stable_hash)
        n = len(keys)
        n_train = int(round(0.70 * n))
        n_val = int(round(0.15 * n))
        splits["train"].extend(keys[:n_train])
        splits["val"].extend(keys[n_train : n_train + n_val])
        splits["test"].extend(keys[n_train + n_val :])
    for name in splits:
        splits[name] = sorted(splits[name])
    payload = {
        "audit_name": "stwm_semantic_memory_world_model_v2_splits",
        "prototype_count_choices": [32, 64],
        "eligible_item_count": int(len(eligible)),
        "train_item_count": int(len(splits["train"])),
        "val_item_count": int(len(splits["val"])),
        "test_item_count": int(len(splits["test"])),
        "splits": splits,
        "stats_c32": {name: _stats_for_keys(keys, future32, observed32) for name, keys in splits.items()},
        "stats_c64": {name: _stats_for_keys(keys, future64, observed64) for name, keys in splits.items()},
        "item_level_split": True,
        "no_item_leakage": True,
        "dataset_balance_preserved": True,
    }
    _write_json(Path(args.output), payload)
    _write_doc(Path(args.doc), "STWM Semantic Memory World Model V2 Splits", payload)


if __name__ == "__main__":
    main()
