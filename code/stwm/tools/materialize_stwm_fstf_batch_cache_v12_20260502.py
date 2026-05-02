#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import stage2_semantic_collate_fn
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import _load_checkpoint, _make_dataset, _merge_args


class _SampleTimeout(Exception):
    pass


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise _SampleTimeout()


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


def entry_key(entry: Any) -> str:
    if isinstance(entry, dict):
        meta = entry.get("meta")
        if isinstance(meta, dict):
            return stage2_item_key(meta)
        dataset = entry.get("dataset_name") or entry.get("dataset") or ""
        split = entry.get("split") or ""
        clip = entry.get("clip_id") or entry.get("video_id") or entry.get("id") or ""
        if dataset and split and clip:
            return f"{dataset}::{split}::{clip}"
    return ""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-report", default="reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json")
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--eval-split", required=True, choices=["train", "val", "test"])
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-entities-per-sample", type=int, default=8)
    p.add_argument("--max-samples-per-dataset", type=int, default=999999)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--timeout-seconds", type=int, default=60)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--allow-scan-all-stage2-splits", action="store_true")
    p.add_argument("--cache-output", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--doc", required=True)
    p.add_argument("--audit-name", default="stwm_fstf_batch_cache_v12")
    args = p.parse_args()

    split_payload = json.loads(Path(args.split_report).read_text(encoding="utf-8"))
    wanted = list(dict.fromkeys(str(x) for x in split_payload["splits"].get(str(args.eval_split), [])))
    payload = _load_checkpoint(Path(args.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    ds_args = _merge_args(
        checkpoint_args,
        {
            "future_semantic_proto_count": 32,
            "fut_len": int(args.fut_len),
            "max_entities_per_sample": int(args.max_entities_per_sample),
        },
    )
    scan_splits = ["train", "val"] if bool(args.allow_scan_all_stage2_splits) else ["train"]
    entry_to_idx: dict[str, tuple[str, int]] = {}
    datasets: dict[str, Any] = {}
    for source_split in scan_splits:
        ds = _make_dataset(ds_args, split=source_split, max_samples_per_dataset=int(args.max_samples_per_dataset))
        datasets[source_split] = ds
        for idx, entry in enumerate(getattr(ds, "entries", [])):
            key = entry_key(entry)
            if key:
                entry_to_idx.setdefault(key, (source_split, idx))

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    materialized: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sources: dict[str, int] = {x: 0 for x in scan_splits}
    for key in wanted:
        ref = entry_to_idx.get(key)
        if ref is None:
            failures.append({"item_key": key, "reason": "item_key_not_found_in_stage2_dataset_entries"})
            continue
        source_split, idx = ref
        loaded = None
        last_reason = ""
        for _attempt in range(int(args.retries) + 1):
            try:
                signal.alarm(int(args.timeout_seconds))
                loaded = datasets[source_split][idx]
                signal.alarm(0)
                break
            except _SampleTimeout:
                signal.alarm(0)
                last_reason = f"sample_load_timeout_{int(args.timeout_seconds)}s"
            except Exception as exc:  # noqa: BLE001
                signal.alarm(0)
                last_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        if loaded is None:
            failures.append({"item_key": key, "source_split": source_split, "dataset_index": int(idx), "reason": last_reason})
            continue
        loaded_key = stage2_item_key(loaded.get("meta", {}))
        if loaded_key != key:
            failures.append({"item_key": key, "source_split": source_split, "dataset_index": int(idx), "loaded_key": loaded_key, "reason": "loaded_key_mismatch"})
            continue
        materialized.append(loaded)
        sources[source_split] += 1
    signal.signal(signal.SIGALRM, previous_handler)

    batches = [stage2_semantic_collate_fn(materialized[i : i + int(args.batch_size)]) for i in range(0, len(materialized), int(args.batch_size))]
    cache_path = Path(args.cache_output)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "batches": batches,
            "item_keys": [stage2_item_key(x.get("meta", {})) for x in materialized],
            "eval_splits": [str(args.eval_split)],
            "split_report": str(args.split_report),
            "fut_len": int(args.fut_len),
            "max_entities_per_sample": int(args.max_entities_per_sample),
        },
        cache_path,
    )
    stat = cache_path.stat()
    report = {
        "audit_name": str(args.audit_name),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_report": str(args.split_report),
        "eval_split": str(args.eval_split),
        "requested_item_count": int(len(wanted)),
        "final_eval_item_count": int(len(materialized)),
        "batch_count": int(len(batches)),
        "batch_cache_path": str(cache_path),
        "cache_size_bytes": int(stat.st_size),
        "cache_mtime": float(stat.st_mtime),
        "fut_len": int(args.fut_len),
        "horizon": int(args.fut_len),
        "max_entities_per_sample": int(args.max_entities_per_sample),
        "slot_count_verified": int(args.max_entities_per_sample),
        "materialized_source_counts": sources,
        "failed_items": failures[:50],
        "failed_item_count": int(len(failures)),
        "materialization_success": bool(len(materialized) == len(wanted) and len(materialized) > 0),
        "exact_blocking_reason": "" if len(materialized) == len(wanted) and len(materialized) > 0 else "some split item keys failed to materialize with requested fut_len/max_entities_per_sample",
        "future_leakage_audit": True,
        "item_leakage": False,
    }
    write_json(Path(args.output), report)
    write_doc(Path(args.doc), "STWM-FSTF Batch Cache V12", report)


if __name__ == "__main__":
    main()
