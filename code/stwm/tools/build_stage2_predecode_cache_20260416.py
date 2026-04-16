#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import re
import time

import numpy as np
from torch.utils.data import DataLoader, Subset

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
    _cache_key,
)


def _repo_root() -> Path:
    for candidate in [Path("/raid/chen034/workspace/stwm"), Path("/home/chen034/workspace/stwm")]:
        if candidate.exists():
            return candidate
    raise RuntimeError("unable to resolve STWM repo root")


ROOT = _repo_root()


def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _flush_index(
    *,
    cache_root: Path,
    entries: Dict[str, str],
    semantic_temporal_window: int,
    semantic_crop_size: int,
) -> None:
    write_json(
        cache_root / "index.json",
        {
            "generated_at_utc": now_iso(),
            "cache_root": str(cache_root),
            "semantic_temporal_window": int(semantic_temporal_window),
            "semantic_crop_size": int(semantic_crop_size),
            "entries": dict(entries),
        },
    )


def _identity_collate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return rows[0]


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(value))


def _sample_to_numpy(sample: Dict[str, Any]) -> Dict[str, Any]:
    meta = dict(sample.get("meta", {}))
    source_summary = dict(sample.get("semantic_source_summary", {}))
    return {
        "obs_state": sample["obs_state"].numpy(),
        "fut_state": sample["fut_state"].numpy(),
        "obs_valid": sample["obs_valid"].numpy(),
        "fut_valid": sample["fut_valid"].numpy(),
        "point_ids": sample["point_ids"].numpy(),
        "semantic_features": sample["semantic_features"].numpy(),
        "semantic_boxes": sample["semantic_boxes"].numpy(),
        "semantic_mask": sample["semantic_mask"].numpy(),
        "semantic_rgb_crop": sample["semantic_rgb_crop"].numpy(),
        "semantic_mask_crop": sample["semantic_mask_crop"].numpy(),
        "semantic_crop_valid": sample["semantic_crop_valid"].numpy(),
        "semantic_mask_crop_valid": sample["semantic_mask_crop_valid"].numpy(),
        "semantic_rgb_crop_temporal": sample["semantic_rgb_crop_temporal"].numpy(),
        "semantic_mask_crop_temporal": sample["semantic_mask_crop_temporal"].numpy(),
        "semantic_temporal_valid": sample["semantic_temporal_valid"].numpy(),
        "semantic_frame_path": np.asarray(str(sample.get("semantic_frame_path", "")), dtype=object),
        "semantic_mask_path": np.asarray(str(sample.get("semantic_mask_path", "")), dtype=object),
        "meta_json": np.asarray(meta, dtype=object),
        "semantic_source_summary_json": np.asarray(source_summary, dtype=object),
    }


def _build_split_cache(
    *,
    dataset_names: List[str],
    split: str,
    contract_path: str,
    cache_root: Path,
    semantic_crop_size: int,
    semantic_temporal_window: int,
    max_samples_per_dataset: int,
    num_workers: int,
    prefetch_factor: int,
    aggregate_entries: Dict[str, str],
) -> Dict[str, Any]:
    cfg = Stage2SemanticDatasetConfig(
        dataset_names=list(dataset_names),
        split=str(split),
        contract_path=str(contract_path),
        obs_len=8,
        fut_len=8,
        max_tokens=64,
        max_samples_per_dataset=int(max_samples_per_dataset),
        semantic_patch_radius=12,
        semantic_crop_size=int(semantic_crop_size),
        semantic_source_mainline="crop_visual_encoder",
        semantic_temporal_window=int(semantic_temporal_window),
        predecode_cache_path="",
    )
    dataset = Stage2SemanticDataset(cfg)
    split_dir = cache_root / str(split)
    split_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    reused_existing = 0
    entries: Dict[str, str] = {}
    pending_rows: List[Tuple[int, str, Path, str]] = []
    start = time.perf_counter()
    per_dataset_counts: Dict[str, int] = {}
    for item in dataset.entries:
        ds_name = str(item.get("dataset_name", ""))
        per_dataset_counts[ds_name] = per_dataset_counts.get(ds_name, 0) + 1
    for idx, entry in enumerate(dataset.entries):
        key = _cache_key(str(entry.get("dataset_name", "")), str(split), str(entry.get("clip_id", "")))
        out_path = split_dir / f"{_safe_name(key)}.npz"
        entries[key] = str(out_path)
        aggregate_entries[key] = str(out_path)
        if out_path.exists():
            reused_existing += 1
            continue
        pending_rows.append((idx, key, out_path, str(entry.get("dataset_name", ""))))

    _flush_index(
        cache_root=cache_root,
        entries=aggregate_entries,
        semantic_temporal_window=semantic_temporal_window,
        semantic_crop_size=semantic_crop_size,
    )

    if pending_rows:
        subset = Subset(dataset, [row[0] for row in pending_rows])
        loader_kwargs: Dict[str, Any] = {
            "dataset": subset,
            "batch_size": 1,
            "shuffle": False,
            "num_workers": max(int(num_workers), 0),
            "collate_fn": _identity_collate,
        }
        if int(num_workers) > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = max(int(prefetch_factor), 2)
        loader = DataLoader(**loader_kwargs)
    else:
        loader = []

    flush_counter = 0
    for (row, sample) in zip(pending_rows, loader):
        _, key, out_path, _ = row
        np.savez_compressed(out_path, **_sample_to_numpy(sample))
        written += 1
        flush_counter += 1
        if flush_counter >= 50:
            _flush_index(
                cache_root=cache_root,
                entries=aggregate_entries,
                semantic_temporal_window=semantic_temporal_window,
                semantic_crop_size=semantic_crop_size,
            )
            flush_counter = 0

    if pending_rows:
        _flush_index(
            cache_root=cache_root,
            entries=aggregate_entries,
            semantic_temporal_window=semantic_temporal_window,
            semantic_crop_size=semantic_crop_size,
        )
    duration = max(time.perf_counter() - start, 1e-6)
    return {
        "split": str(split),
        "entry_count": int(len(dataset.entries)),
        "newly_written_count": int(written),
        "reused_existing_count": int(reused_existing),
        "pending_before_run_count": int(len(pending_rows)),
        "entries": entries,
        "per_dataset_counts": per_dataset_counts,
        "duration_sec": float(duration),
        "samples_per_sec": float(max(written, 1) / duration),
    }


def parse_args() -> Any:
    p = ArgumentParser(description="Build Stage2 predecode/sample cache for runtime bottleneck relief")
    p.add_argument("--contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--cache-root", default=str(ROOT / "data/processed/stage2_predecode_cache_20260416"))
    p.add_argument("--output-json", default=str(ROOT / "reports/stage2_predecode_cache_build_20260416.json"))
    p.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_RUNTIME_PIPELINE_OPTIMIZATION_20260416.md"))
    p.add_argument("--dataset-names", nargs="*", default=["vspw", "vipseg", "burst"])
    p.add_argument("--splits", nargs="*", default=["train", "val"])
    p.add_argument("--semantic-crop-size", type=int, default=64)
    p.add_argument("--semantic-temporal-window", type=int, default=5)
    p.add_argument("--max-samples-per-dataset", type=int, default=-1)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--prefetch-factor", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    aggregate_entries: Dict[str, str] = {}
    split_payloads: List[Dict[str, Any]] = []
    total_written = 0
    total_reused = 0
    total_duration = 0.0
    for split in [str(x) for x in args.splits]:
        payload = _build_split_cache(
            dataset_names=[str(x) for x in args.dataset_names],
            split=split,
            contract_path=str(args.contract_json),
            cache_root=cache_root,
            semantic_crop_size=int(args.semantic_crop_size),
            semantic_temporal_window=int(args.semantic_temporal_window),
            max_samples_per_dataset=int(args.max_samples_per_dataset),
            num_workers=int(args.num_workers),
            prefetch_factor=int(args.prefetch_factor),
            aggregate_entries=aggregate_entries,
        )
        split_payloads.append(payload)
        aggregate_entries.update(payload["entries"])
        total_written += int(payload["newly_written_count"])
        total_reused += int(payload["reused_existing_count"])
        total_duration += float(payload["duration_sec"])
    index_payload = {
        "generated_at_utc": now_iso(),
        "cache_root": str(cache_root),
        "semantic_temporal_window": int(args.semantic_temporal_window),
        "semantic_crop_size": int(args.semantic_crop_size),
        "entries": aggregate_entries,
    }
    write_json(cache_root / "index.json", index_payload)
    report = {
        "generated_at_utc": now_iso(),
        "cache_root": str(cache_root),
        "index_json": str(cache_root / "index.json"),
        "dataset_names": [str(x) for x in args.dataset_names],
        "splits": [str(x) for x in args.splits],
        "semantic_temporal_window": int(args.semantic_temporal_window),
        "semantic_crop_size": int(args.semantic_crop_size),
        "total_cached_entries": int(len(aggregate_entries)),
        "newly_written_entries": int(total_written),
        "reused_existing_entries": int(total_reused),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "total_duration_sec": float(total_duration),
        "aggregate_samples_per_sec": float(max(total_written, 1) / max(total_duration, 1e-6)),
        "split_summaries": split_payloads,
    }
    write_json(args.output_json, report)
    write_md(
        args.output_md,
        [
            "# Stage2 Runtime Pipeline Optimization 20260416",
            "",
            f"- cache_root: {report['cache_root']}",
            f"- total_cached_entries: {report['total_cached_entries']}",
            f"- aggregate_samples_per_sec: {report['aggregate_samples_per_sec']:.3f}",
            "",
            "## Split Summaries",
            "",
            *[
                f"- {row['split']}: entry_count={row['entry_count']} samples_per_sec={row['samples_per_sec']:.3f} per_dataset={json.dumps(row['per_dataset_counts'], ensure_ascii=True)}"
                for row in split_payloads
            ],
        ],
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
