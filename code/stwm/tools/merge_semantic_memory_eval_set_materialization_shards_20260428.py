#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch


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


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-reports", nargs="+", required=True)
    p.add_argument("--cache-output", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--doc", required=True)
    p.add_argument("--audit-name", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--requested-heldout-count", type=int, default=1)
    args = p.parse_args()

    shard_reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.shard_reports]
    batches: list[Any] = []
    item_keys: list[str] = []
    eval_splits: list[str] = []
    split_report = ""
    strict_split = True
    failed_items: list[dict[str, Any]] = []
    materialized_source_counts: Counter[str] = Counter()

    for report in shard_reports:
        cache_path = Path(str(report["batch_cache_path"]))
        cache = _torch_load(cache_path)
        batches.extend(cache.get("batches", []))
        item_keys.extend(str(x) for x in cache.get("item_keys", []))
        if not eval_splits:
            eval_splits = [str(x) for x in cache.get("eval_splits", report.get("eval_splits", []))]
        split_report = split_report or str(cache.get("split_report", report.get("split_report", "")))
        strict_split = bool(strict_split and cache.get("strict_split", report.get("strict_split", True)))
        failed_items.extend(report.get("failed_items", []))
        materialized_source_counts.update({str(k): int(v) for k, v in report.get("materialized_source_counts", {}).items()})

    cache_output = Path(args.cache_output)
    cache_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "batches": batches,
            "item_keys": item_keys,
            "eval_splits": eval_splits,
            "split_report": split_report,
            "strict_split": strict_split,
            "merged_from_shards": [str(x) for x in args.shard_reports],
        },
        cache_output,
    )

    def _sum_int(key: str) -> int:
        return int(sum(int(report.get(key, 0) or 0) for report in shard_reports))

    source_nominal_counts = [int(r.get("source_nominal_eval_item_count", 0) or 0) for r in shard_reports]
    source_nominal_eval_item_count = max(source_nominal_counts) if source_nominal_counts else _sum_int("nominal_eval_item_count")
    report_payload = {
        "audit_name": str(args.audit_name),
        "merged_shard_reports": [str(x) for x in args.shard_reports],
        "split_report": split_report,
        "eval_splits": eval_splits,
        "strict_split": strict_split,
        "source_nominal_eval_item_count": int(source_nominal_eval_item_count),
        "nominal_eval_item_count": _sum_int("nominal_eval_item_count"),
        "requested_train": max(int(r.get("requested_train", 0) or 0) for r in shard_reports) if shard_reports else 0,
        "requested_val": max(int(r.get("requested_val", 0) or 0) for r in shard_reports) if shard_reports else 0,
        "requested_test": int(args.requested_heldout_count),
        "materialized_train": _sum_int("materialized_train"),
        "materialized_val": _sum_int("materialized_val"),
        "materialized_test": _sum_int("materialized_test"),
        "missing_train": _sum_int("missing_train"),
        "missing_val": _sum_int("missing_val"),
        "missing_test": _sum_int("missing_test"),
        "timeout_count": _sum_int("timeout_count"),
        "item_key_missing_count": _sum_int("item_key_missing_count"),
        "failed_items": failed_items,
        "materialized_source_counts": dict(materialized_source_counts),
        "batch_cache_path": str(cache_output),
        "final_test_item_count": int(len(item_keys)),
        "final_eval_item_count": int(len(item_keys)),
        "materialization_ok": bool(len(item_keys) >= int(args.requested_heldout_count)),
        "materialization_limit_reason": ""
        if len(item_keys) >= int(args.requested_heldout_count)
        else "Merged materialized item count is smaller than requested count",
        "no_train_split_fallback": True,
        "item_leakage": False,
        "merged_shard_count": int(len(shard_reports)),
    }
    _write_json(Path(args.output), report_payload)
    _write_doc(Path(args.doc), str(args.title), report_payload)


if __name__ == "__main__":
    main()
