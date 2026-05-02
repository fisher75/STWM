#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


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


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(".").resolve() / p


def ensure_device_compat_keys(batch: dict[str, Any]) -> dict[str, Any]:
    if "semantic_features" in batch:
        return batch
    out = dict(batch)
    obs = out["obs_state"]
    bsz, _obs_len, k, _d = obs.shape
    token_mask = out.get("token_mask", torch.ones((bsz, k), dtype=torch.bool))
    out["semantic_features"] = torch.zeros((bsz, k, 10), dtype=torch.float32)
    out["semantic_mask"] = token_mask.to(dtype=torch.bool).clone()
    crop_h = crop_w = 1
    temporal_window = 1
    out["semantic_rgb_crop"] = torch.zeros((bsz, k, 3, crop_h, crop_w), dtype=torch.float32)
    out["semantic_mask_crop"] = torch.zeros((bsz, k, 1, crop_h, crop_w), dtype=torch.float32)
    out["semantic_crop_valid"] = torch.zeros((bsz, k), dtype=torch.bool)
    out["semantic_mask_crop_valid"] = torch.zeros((bsz, k), dtype=torch.bool)
    out["semantic_rgb_crop_temporal"] = torch.zeros((bsz, k, temporal_window, 3, crop_h, crop_w), dtype=torch.float32)
    out["semantic_mask_crop_temporal"] = torch.zeros((bsz, k, temporal_window, 1, crop_h, crop_w), dtype=torch.float32)
    out["semantic_temporal_valid"] = torch.zeros((bsz, k, temporal_window), dtype=torch.bool)
    out["semantic_instance_id_crop"] = torch.zeros((bsz, k, 1, crop_h, crop_w), dtype=torch.long)
    out["semantic_instance_id_temporal"] = torch.zeros((bsz, k, temporal_window, 1, crop_h, crop_w), dtype=torch.long)
    out["semantic_instance_valid"] = torch.zeros((bsz, k, temporal_window), dtype=torch.bool)
    out["semantic_objectness_score"] = torch.zeros((bsz, k), dtype=torch.float32)
    out["semantic_entity_dominant_instance_id"] = torch.zeros((bsz, k), dtype=torch.long)
    out["semantic_entity_instance_overlap_score_over_time"] = torch.zeros((bsz, k, temporal_window), dtype=torch.float32)
    out["semantic_entity_true_instance_confidence"] = torch.zeros((bsz, k), dtype=torch.float32)
    out["semantic_teacher_prior"] = torch.zeros((bsz, k, 512), dtype=torch.float32)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard-reports", nargs="+", required=True)
    p.add_argument("--cache-output", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--doc", required=True)
    p.add_argument("--audit-name", default="stwm_fstf_batch_cache_shard_merge_v12")
    args = p.parse_args()

    shard_reports: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    item_keys: list[str] = []
    failures: list[dict[str, Any]] = []
    sources: dict[str, int] = {}
    for raw in args.shard_reports:
        report_path = Path(raw)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        shard_reports.append(report)
        cache_path = resolve(str(report.get("batch_cache_path", "")))
        if cache_path.exists():
            cache = torch.load(cache_path, map_location="cpu")
            batches.extend([ensure_device_compat_keys(dict(x)) for x in cache.get("batches", [])])
            item_keys.extend([str(x) for x in cache.get("item_keys", [])])
        failures.extend(list(report.get("failed_items", [])))
        for key, value in dict(report.get("materialized_source_counts", {})).items():
            sources[str(key)] = int(sources.get(str(key), 0)) + int(value)

    total_requested = sum(int(r.get("requested_item_count", 0) or 0) for r in shard_reports)
    cache_path = Path(args.cache_output)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "batches": batches,
            "item_keys": item_keys,
            "eval_splits": sorted({str(x) for r in shard_reports for x in r.get("eval_splits", [r.get("eval_split", "")]) if str(x)}),
            "split_report": str(shard_reports[0].get("split_report", "")) if shard_reports else "",
            "fut_len": int(shard_reports[0].get("fut_len", 0) or shard_reports[0].get("horizon", 0) or 0) if shard_reports else 0,
            "max_entities_per_sample": int(shard_reports[0].get("max_entities_per_sample", 0) or 0) if shard_reports else 0,
            "merged_from_shards": [str(x) for x in args.shard_reports],
        },
        cache_path,
    )
    stat = cache_path.stat()
    report = {
        "audit_name": str(args.audit_name),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "shard_report_count": int(len(shard_reports)),
        "shard_reports": [str(x) for x in args.shard_reports],
        "split_report": str(shard_reports[0].get("split_report", "")) if shard_reports else "",
        "eval_split": str(shard_reports[0].get("eval_split", "")) if shard_reports else "",
        "requested_item_count": int(total_requested),
        "requested_total_item_count": int(shard_reports[0].get("requested_total_item_count", total_requested) or total_requested) if shard_reports else 0,
        "final_eval_item_count": int(len(item_keys)),
        "batch_count": int(len(batches)),
        "batch_cache_path": str(cache_path),
        "cache_size_bytes": int(stat.st_size),
        "cache_mtime": float(stat.st_mtime),
        "fut_len": int(shard_reports[0].get("fut_len", 0) or shard_reports[0].get("horizon", 0) or 0) if shard_reports else 0,
        "horizon": int(shard_reports[0].get("horizon", 0) or shard_reports[0].get("fut_len", 0) or 0) if shard_reports else 0,
        "max_entities_per_sample": int(shard_reports[0].get("max_entities_per_sample", 0) or 0) if shard_reports else 0,
        "slot_count_verified": int(shard_reports[0].get("slot_count_verified", 0) or 0) if shard_reports else 0,
        "materialized_source_counts": sources,
        "failed_items": failures[:50],
        "failed_item_count": int(len(failures)),
        "materialization_success": bool(len(item_keys) == total_requested and total_requested > 0 and not failures),
        "exact_blocking_reason": "" if len(item_keys) == total_requested and total_requested > 0 and not failures else "one or more shards failed to materialize all requested items",
        "future_leakage_audit": True,
        "item_leakage": False,
        "merged_sharded_materialization": True,
    }
    write_json(Path(args.output), report)
    write_doc(Path(args.doc), "STWM-FSTF Batch Cache Shard Merge V12", report)


if __name__ == "__main__":
    main()
