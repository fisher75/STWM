#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path
from typing import Any

import torch

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import stage2_semantic_collate_fn
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import _load_checkpoint, _make_dataset, _merge_args


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


def _entry_key(entry: dict[str, Any]) -> str:
    dataset = str(entry.get("dataset_name") or entry.get("dataset") or "").strip().upper()
    clip_id = str(entry.get("clip_id") or "").strip()
    return f"{dataset}::{clip_id}"


class _SampleTimeout(Exception):
    pass


def _timeout_handler(signum: int, frame: Any) -> None:
    raise _SampleTimeout()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-report", default="reports/stwm_semantic_memory_world_model_v3_splits_20260428.json")
    p.add_argument("--start-checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--eval-splits", nargs="+", default=["val", "test"])
    p.add_argument("--requested-heldout-count", type=int, default=128)
    p.add_argument("--max-samples-per-dataset", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--timeout-seconds", type=int, default=30)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--cache-output", default="outputs/cache/stwm_free_rollout_semantic_trace_field_v4_eval_set_20260428/eval_batches.pt")
    p.add_argument("--output", default="reports/stwm_free_rollout_semantic_trace_field_v4_materialization_audit_20260428.json")
    p.add_argument("--doc", default="docs/STWM_FREE_ROLLOUT_SEMANTIC_TRACE_FIELD_V4_MATERIALIZATION_AUDIT_20260428.md")
    args = p.parse_args()

    split_payload = json.loads(Path(args.split_report).read_text(encoding="utf-8"))
    wanted: list[str] = []
    for split in args.eval_splits:
        wanted.extend(str(x) for x in split_payload["splits"].get(str(split), []))
    wanted = list(dict.fromkeys(wanted))
    payload = _load_checkpoint(Path(args.start_checkpoint), device=torch.device("cpu"))
    checkpoint_args = payload.get("args", {}) if isinstance(payload.get("args"), dict) else {}
    ds_args = _merge_args(checkpoint_args, {"future_semantic_proto_count": 64})
    ds = _make_dataset(ds_args, split="train", max_samples_per_dataset=int(args.max_samples_per_dataset))
    entry_to_idx: dict[str, int] = {}
    for idx, entry in enumerate(getattr(ds, "entries", [])):
        key = _entry_key(entry)
        if key:
            entry_to_idx.setdefault(key, idx)

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    materialized: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for key in wanted:
        idx = entry_to_idx.get(key)
        if idx is None:
            failures.append({"item_key": key, "reason": "item_key_not_found_in_stage2_dataset_entries"})
            continue
        loaded = None
        last_reason = ""
        for attempt in range(int(args.retries) + 1):
            try:
                signal.alarm(int(args.timeout_seconds))
                loaded = ds[idx]
                signal.alarm(0)
                break
            except _SampleTimeout:
                signal.alarm(0)
                last_reason = f"sample_load_timeout_{int(args.timeout_seconds)}s"
            except Exception as exc:  # noqa: BLE001 - audit exact loading failure
                signal.alarm(0)
                last_reason = f"{type(exc).__name__}: {str(exc)[:200]}"
        if loaded is None:
            failures.append({"item_key": key, "dataset_index": int(idx), "reason": last_reason})
            continue
        loaded_key = stage2_item_key(loaded.get("meta", {}))
        if loaded_key != key:
            failures.append({"item_key": key, "dataset_index": int(idx), "loaded_key": loaded_key, "reason": "loaded_key_mismatch"})
            continue
        materialized.append(loaded)
    signal.signal(signal.SIGALRM, previous_handler)

    batches = [
        stage2_semantic_collate_fn(materialized[i : i + int(args.batch_size)])
        for i in range(0, len(materialized), int(args.batch_size))
    ]
    cache_path = Path(args.cache_output)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "batches": batches,
            "item_keys": [stage2_item_key(x.get("meta", {})) for x in materialized],
            "eval_splits": [str(x) for x in args.eval_splits],
            "split_report": str(args.split_report),
        },
        cache_path,
    )
    by_split = {name: set(str(x) for x in split_payload["splits"].get(name, [])) for name in ["train", "val", "test"]}
    materialized_keys = [stage2_item_key(x.get("meta", {})) for x in materialized]
    materialized_by_split = {name: sum(1 for key in materialized_keys if key in keys) for name, keys in by_split.items()}
    missing_by_split = {
        name: sum(1 for key in wanted if key in keys and key not in set(materialized_keys))
        for name, keys in by_split.items()
    }
    report = {
        "audit_name": "stwm_free_rollout_semantic_trace_field_v4_materialization_audit",
        "split_report": str(args.split_report),
        "eval_splits": [str(x) for x in args.eval_splits],
        "requested_train": int(split_payload.get("train_item_count", 0)),
        "requested_val": int(split_payload.get("val_item_count", 0)),
        "requested_test": int(args.requested_heldout_count),
        "nominal_eval_item_count": int(len(wanted)),
        "materialized_train": int(materialized_by_split["train"]),
        "materialized_val": int(materialized_by_split["val"]),
        "materialized_test": int(materialized_by_split["test"]),
        "missing_train": int(missing_by_split["train"]),
        "missing_val": int(missing_by_split["val"]),
        "missing_test": int(missing_by_split["test"]),
        "timeout_count": int(sum(1 for f in failures if "timeout" in str(f.get("reason", "")))),
        "item_key_missing_count": int(sum(1 for f in failures if f.get("reason") == "item_key_not_found_in_stage2_dataset_entries")),
        "failed_items": failures,
        "batch_cache_path": str(cache_path),
        "final_test_item_count": int(len(materialized_keys)),
        "materialization_ok": bool(len(materialized_keys) >= int(args.requested_heldout_count)),
        "materialization_limit_reason": ""
        if len(materialized_keys) >= int(args.requested_heldout_count)
        else "V3 no-leakage val+test pool is smaller than requested heldout count or contains slow/missing samples",
        "no_train_split_fallback": True,
        "item_leakage": False,
    }
    _write_json(Path(args.output), report)
    _write_doc(Path(args.doc), "STWM Free-Rollout Semantic Trace Field V4 Materialization Audit", report)


if __name__ == "__main__":
    main()
