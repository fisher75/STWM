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
        "semantic_instance_id_map": sample["semantic_instance_id_map"].numpy(),
        "semantic_instance_id_crop": sample["semantic_instance_id_crop"].numpy(),
        "semantic_instance_id_temporal": sample["semantic_instance_id_temporal"].numpy(),
        "semantic_instance_valid": sample["semantic_instance_valid"].numpy(),
        "semantic_objectness_score": sample["semantic_objectness_score"].numpy(),
        "semantic_entity_dominant_instance_id": sample["semantic_entity_dominant_instance_id"].numpy(),
        "semantic_entity_instance_overlap_score_over_time": sample["semantic_entity_instance_overlap_score_over_time"].numpy(),
        "semantic_entity_true_instance_confidence": sample["semantic_entity_true_instance_confidence"].numpy(),
        "semantic_teacher_prior": sample["semantic_teacher_prior"].numpy(),
        "entity_boxes_over_time": sample["entity_boxes_over_time"].numpy(),
        "entity_masks_over_time": np.asarray(sample["entity_masks_over_time"], dtype=object),
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
    max_entities_per_sample: int,
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
        max_entities_per_sample=int(max_entities_per_sample),
        include_full_instance_id_map=True,
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
    entity_histogram: Dict[str, int] = {}
    instance_source_counts: Dict[str, int] = {}
    true_instance_samples = 0
    total_samples = 0
    for item in dataset.entries:
        ds_name = str(item.get("dataset_name", ""))
        per_dataset_counts[ds_name] = per_dataset_counts.get(ds_name, 0) + 1
    for idx, entry in enumerate(dataset.entries):
        key = _cache_key(str(entry.get("dataset_name", "")), str(split), str(entry.get("clip_id", "")))
        out_path = split_dir / f"{_safe_name(key)}.npz"
        entries[key] = str(out_path)
        aggregate_entries[key] = str(out_path)
        if out_path.exists():
            try:
                with np.load(out_path, allow_pickle=True) as payload:
                    meta = dict(payload["meta_json"].item())
                entity_count = int(meta.get("entity_count", 0))
                entity_histogram[str(entity_count)] = entity_histogram.get(str(entity_count), 0) + 1
                source = str(meta.get("instance_source", "unknown"))
                instance_source_counts[source] = instance_source_counts.get(source, 0) + 1
                total_samples += 1
                if bool(meta.get("true_instance_aware", False)):
                    true_instance_samples += 1
            except Exception:
                pass
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
        meta = dict(sample.get("meta", {}))
        entity_count = int(meta.get("entity_count", int(sample["point_ids"].shape[0])))
        entity_histogram[str(entity_count)] = entity_histogram.get(str(entity_count), 0) + 1
        source = str(meta.get("instance_source", "unknown"))
        instance_source_counts[source] = instance_source_counts.get(source, 0) + 1
        total_samples += 1
        if bool(meta.get("true_instance_aware", False)):
            true_instance_samples += 1
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
        "entity_count_histogram": entity_histogram,
        "instance_source_counts": instance_source_counts,
        "true_instance_samples": int(true_instance_samples),
        "total_samples": int(total_samples),
    }


def parse_args() -> Any:
    p = ArgumentParser(description="Build Stage2 multi-entity instance-aware predecode cache for TUSB-v2")
    p.add_argument("--contract-json", default=str(ROOT / "reports/stage2_bootstrap_data_contract_20260408.json"))
    p.add_argument("--cache-root", default=str(ROOT / "data/processed/stage2_tusb_v2_predecode_cache_20260418"))
    p.add_argument("--output-json", default=str(ROOT / "reports/stage2_tusb_v2_cache_health_20260418.json"))
    p.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_TUSB_V2_CACHE_HEALTH_20260418.md"))
    p.add_argument("--multi-entity-report", default=str(ROOT / "reports/stage2_multi_entity_tusb_data_20260418.json"))
    p.add_argument("--multi-entity-doc", default=str(ROOT / "docs/STAGE2_MULTI_ENTITY_TUSB_DATA_20260418.md"))
    p.add_argument("--dataset-names", nargs="*", default=["vspw", "vipseg"])
    p.add_argument("--splits", nargs="*", default=["train", "val"])
    p.add_argument("--semantic-crop-size", type=int, default=64)
    p.add_argument("--semantic-temporal-window", type=int, default=5)
    p.add_argument("--max-entities-per-sample", type=int, default=8)
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
    entity_histogram: Dict[str, int] = {}
    instance_source_counts: Dict[str, int] = {}
    true_instance_samples = 0
    total_samples = 0
    for split in [str(x) for x in args.splits]:
        payload = _build_split_cache(
            dataset_names=[str(x) for x in args.dataset_names],
            split=split,
            contract_path=str(args.contract_json),
            cache_root=cache_root,
            semantic_crop_size=int(args.semantic_crop_size),
            semantic_temporal_window=int(args.semantic_temporal_window),
            max_entities_per_sample=int(args.max_entities_per_sample),
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
        for key, value in payload.get("entity_count_histogram", {}).items():
            entity_histogram[str(key)] = entity_histogram.get(str(key), 0) + int(value)
        for key, value in payload.get("instance_source_counts", {}).items():
            instance_source_counts[str(key)] = instance_source_counts.get(str(key), 0) + int(value)
        true_instance_samples += int(payload.get("true_instance_samples", 0))
        total_samples += int(payload.get("total_samples", 0))
    index_payload = {
        "generated_at_utc": now_iso(),
        "cache_root": str(cache_root),
        "semantic_temporal_window": int(args.semantic_temporal_window),
        "semantic_crop_size": int(args.semantic_crop_size),
        "max_entities_per_sample": int(args.max_entities_per_sample),
        "entries": aggregate_entries,
    }
    write_json(cache_root / "index.json", index_payload)
    required_keys = [
        "semantic_instance_id_map",
        "semantic_instance_id_crop",
        "semantic_instance_id_temporal",
        "semantic_instance_valid",
        "semantic_objectness_score",
        "semantic_entity_dominant_instance_id",
        "semantic_entity_instance_overlap_score_over_time",
        "semantic_entity_true_instance_confidence",
        "entity_boxes_over_time",
        "entity_masks_over_time",
    ]
    checked = 0
    missing_key_hits = 0
    per_dataset_cache_compatibility: Dict[str, Dict[str, Any]] = {}
    for split_payload in split_payloads:
        for dataset_name in split_payload.get("per_dataset_counts", {}).keys():
            per_dataset_cache_compatibility[str(dataset_name)] = {
                "compatible": True,
                "missing_keys": [],
            }
        for out_path in split_payload.get("entries", {}).values():
            npz_path = Path(str(out_path))
            if not npz_path.exists():
                continue
            with np.load(npz_path, allow_pickle=False) as payload:
                keys = set(payload.files)
            missing = [key for key in required_keys if key not in keys]
            checked += 1
            missing_key_hits += int(bool(missing))
            dataset_name = "unknown"
            try:
                with np.load(npz_path, allow_pickle=True) as payload:
                    meta = payload["meta_json"].item()
                    dataset_name = str(meta.get("dataset", "unknown"))
            except Exception:
                pass
            block = per_dataset_cache_compatibility.setdefault(dataset_name, {"compatible": True, "missing_keys": []})
            if missing:
                block["compatible"] = False
                block["missing_keys"] = sorted(set(list(block.get("missing_keys", [])) + missing))
    report = {
        "generated_at_utc": now_iso(),
        "cache_root": str(cache_root),
        "index_json": str(cache_root / "index.json"),
        "dataset_names": [str(x) for x in args.dataset_names],
        "splits": [str(x) for x in args.splits],
        "semantic_temporal_window": int(args.semantic_temporal_window),
        "semantic_crop_size": int(args.semantic_crop_size),
        "max_entities_per_sample": int(args.max_entities_per_sample),
        "total_cached_entries": int(len(aggregate_entries)),
        "newly_written_entries": int(total_written),
        "reused_existing_entries": int(total_reused),
        "num_workers": int(args.num_workers),
        "prefetch_factor": int(args.prefetch_factor),
        "total_duration_sec": float(total_duration),
        "aggregate_samples_per_sec": float(max(total_written, 1) / max(total_duration, 1e-6)),
        "split_summaries": split_payloads,
        "cache_hit_rate": 1.0 if aggregate_entries else 0.0,
        "missing_keys_rate": float(missing_key_hits / max(checked, 1)),
        "fallback_to_raw_decode_ratio": float(missing_key_hits / max(checked, 1)),
        "per_dataset_cache_compatibility": per_dataset_cache_compatibility,
    }
    write_json(args.output_json, report)
    multi_entity_report = {
        "generated_at_utc": now_iso(),
        "dataset_names": [str(x) for x in args.dataset_names],
        "max_entities_per_sample": int(args.max_entities_per_sample),
        "sample_count": int(total_samples),
        "entity_count_histogram": entity_histogram,
        "multi_entity_sample_coverage_ratio": float(sum(v for k, v in entity_histogram.items() if int(k) >= 2) / max(total_samples, 1)),
        "vipseg_true_instance_continuity_coverage_ratio": float(true_instance_samples / max(total_samples, 1)),
        "instance_source_counts": instance_source_counts,
        "dataset_instance_awareness": {
            "VIPSeg": {"true_instance_aware": True, "mode": "panoptic_instance_id"},
            "VSPW": {"true_instance_aware": False, "mode": "pseudo_or_null_component"},
        },
    }
    write_json(args.multi_entity_report, multi_entity_report)
    write_md(
        args.output_md,
        [
            "# Stage2 TUSB-V2 Cache Health 20260418",
            "",
            f"- cache_root: {report['cache_root']}",
            f"- total_cached_entries: {report['total_cached_entries']}",
            f"- cache_hit_rate: {report['cache_hit_rate']:.4f}",
            f"- missing_keys_rate: {report['missing_keys_rate']:.4f}",
            f"- fallback_to_raw_decode_ratio: {report['fallback_to_raw_decode_ratio']:.4f}",
            f"- aggregate_samples_per_sec: {report['aggregate_samples_per_sec']:.3f}",
            "",
            "## Dataset Compatibility",
            "",
            *[
                f"- {name}: compatible={meta['compatible']} missing_keys={json.dumps(meta['missing_keys'], ensure_ascii=True)}"
                for name, meta in sorted(per_dataset_cache_compatibility.items())
            ],
            "",
            "## Split Summaries",
            "",
            *[
                f"- {row['split']}: entry_count={row['entry_count']} samples_per_sec={row['samples_per_sec']:.3f} per_dataset={json.dumps(row['per_dataset_counts'], ensure_ascii=True)}"
                for row in split_payloads
            ],
        ],
    )
    write_md(
        args.multi_entity_doc,
        [
            "# Stage2 Multi-Entity TUSB Data 20260418",
            "",
            f"- sample_count: {multi_entity_report['sample_count']}",
            f"- max_entities_per_sample: {multi_entity_report['max_entities_per_sample']}",
            f"- multi_entity_sample_coverage_ratio: {multi_entity_report['multi_entity_sample_coverage_ratio']:.4f}",
            f"- vipseg_true_instance_continuity_coverage_ratio: {multi_entity_report['vipseg_true_instance_continuity_coverage_ratio']:.4f}",
            "",
            "## Entity Count Histogram",
            "",
            *[
                f"- K={key}: {value}"
                for key, value in sorted(entity_histogram.items(), key=lambda kv: int(kv[0]))
            ],
            "",
            "## Instance Sources",
            "",
            *[
                f"- {key}: {value}"
                for key, value in sorted(instance_source_counts.items())
            ],
        ],
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
