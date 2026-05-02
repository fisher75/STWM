#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(path: str | None) -> Path:
    p = Path(path or "")
    return p if p.is_absolute() else Path(".").resolve() / p


def file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    h = hashlib.sha1(path.read_bytes()[:1_000_000]).hexdigest()
    return {"path": str(path), "exists": True, "size_bytes": int(stat.st_size), "mtime": float(stat.st_mtime), "sha1_8": h[:8]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--axis", required=True, choices=["horizon", "density"])
    p.add_argument("--value", required=True)
    p.add_argument("--future-report", required=True)
    p.add_argument("--observed-report", required=True)
    p.add_argument("--train-report", required=True)
    p.add_argument("--val-report", required=True)
    p.add_argument("--test-report", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--doc", required=True)
    args = p.parse_args()
    future = load(args.future_report)
    observed = load(args.observed_report)
    train = load(args.train_report)
    val = load(args.val_report)
    test = load(args.test_report)
    future_cache = resolve(future.get("target_cache_path"))
    observed_cache = resolve(observed.get("target_cache_paths_by_prototype_count", {}).get("32", observed.get("target_cache_path", "")))
    changed = stable = 0
    target_coverage = float(future.get("target_valid_ratio", future.get("valid_target_ratio", 0.0)) or 0.0)
    try:
        with np.load(future_cache, allow_pickle=True) as fut, np.load(observed_cache, allow_pickle=True) as obs:
            fut_target = np.asarray(fut["future_semantic_proto_target"], dtype=np.int64)
            fut_mask = np.asarray(fut["target_mask"], dtype=bool)
            obs_target = np.asarray(obs["observed_semantic_proto_target"], dtype=np.int64)
            obs_mask = np.asarray(obs["observed_semantic_proto_mask"], dtype=bool)
            valid = fut_mask & (fut_target >= 0) & obs_mask[:, None, : fut_target.shape[2]]
            obs_rep = obs_target[:, None, : fut_target.shape[2]]
            changed = int((valid & (fut_target != obs_rep)).sum())
            stable = int((valid & (fut_target == obs_rep)).sum())
    except Exception:
        changed = stable = 0
    horizon = int(future.get("target_shape", [0, 0, 0])[1]) if future.get("target_shape") else int(train.get("horizon", 0) or train.get("fut_len", 0) or 0)
    slot_count = int(future.get("target_shape", [0, 0, 0])[2]) if future.get("target_shape") else int(train.get("slot_count_verified", 0) or 0)
    payload = {
        "audit_name": f"stwm_fstf_{args.axis}_cache_{args.value}_v12",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "axis": args.axis,
        "value": args.value,
        "horizon": horizon,
        "K": slot_count,
        "slot_count_verified": slot_count,
        "train_item_count": int(train.get("final_eval_item_count", 0)),
        "val_item_count": int(val.get("final_eval_item_count", 0)),
        "test_item_count": int(test.get("final_eval_item_count", 0)),
        "future_target_cache_path": str(future_cache),
        "observed_memory_cache_path": str(observed_cache),
        "batch_cache_paths": {
            "train": train.get("batch_cache_path", ""),
            "val": val.get("batch_cache_path", ""),
            "test": test.get("batch_cache_path", ""),
        },
        "changed_count": int(changed),
        "stable_count": int(stable),
        "target_coverage": target_coverage,
        "observed_semantic_memory_coverage": float(observed.get("observed_proto_valid_ratio", 0.0) or 0.0),
        "future_leakage_audit": bool(future.get("no_future_candidate_leakage", True)),
        "future_cache_file": file_meta(future_cache),
        "observed_cache_file": file_meta(observed_cache),
        "train_batch_cache_file": file_meta(resolve(train.get("batch_cache_path"))),
        "val_batch_cache_file": file_meta(resolve(val.get("batch_cache_path"))),
        "test_batch_cache_file": file_meta(resolve(test.get("batch_cache_path"))),
        "materialization_success": bool(
            train.get("materialization_success", False)
            and val.get("materialization_success", False)
            and test.get("materialization_success", False)
            and future_cache.exists()
            and observed_cache.exists()
        ),
        "exact_blocking_reason": ""
        if (
            train.get("materialization_success", False)
            and val.get("materialization_success", False)
            and test.get("materialization_success", False)
            and future_cache.exists()
            and observed_cache.exists()
        )
        else (
            "future or observed cache missing"
            if not (future_cache.exists() and observed_cache.exists())
            else "one or more train/val/test batch cache materializations failed"
        ),
    }
    write_json(Path(args.output), payload)
    write_doc(Path(args.doc), f"STWM-FSTF {args.axis} cache {args.value} V12", payload)


if __name__ == "__main__":
    main()
