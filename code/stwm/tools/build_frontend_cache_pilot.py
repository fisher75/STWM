from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from stwm.datasets.stwm_dataset import STWMDataset
from stwm.modules.semantic_adapter import SemanticAdapter
from stwm.modules.trace_adapter import TraceAdapter


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build STWM frontend cache pilot shards from a fixed manifest slice")
    parser.add_argument("--data-root", default="/home/chen034/workspace/stwm/data/external")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slice-offset", type=int, default=0)
    parser.add_argument("--slice-size", type=int, default=256)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trace-cache-dir", default="")
    parser.add_argument("--semantic-cache-dir", default="")
    return parser


def _read_mask_ratio(mask_path: str, target_label_id: int | None) -> float:
    p = Path(mask_path)
    if not p.exists():
        return 0.0
    arr = np.array(Image.open(p))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if target_label_id is None:
        return float((arr > 0).mean())
    tgt = arr == int(target_label_id)
    if tgt.any():
        return float(tgt.mean())
    return float((arr > 0).mean())


def _reappearance_indices(visibility: torch.Tensor, min_gap: int = 1) -> list[int]:
    flags = (visibility >= 0.5).to(dtype=torch.int32).detach().cpu().tolist()
    out: list[int] = []
    min_gap = max(1, int(min_gap))
    for i in range(1, len(flags)):
        if flags[i] != 1:
            continue
        j = i - 1
        gap = 0
        while j >= 0 and flags[j] == 0:
            gap += 1
            j -= 1
        had_visible_before = j >= 0 and flags[j] == 1
        if had_visible_before and gap >= min_gap:
            out.append(i)
    return out


def _flush_shard(
    *,
    shard_records: list[dict[str, Any]],
    shard_idx: int,
    shards_dir: Path,
    entries: list[dict[str, Any]],
) -> int:
    if not shard_records:
        return shard_idx

    shard_name = f"shard_{shard_idx:05d}.pt"
    shard_rel_path = f"shards/{shard_name}"
    shard_path = shards_dir / shard_name
    payload = {
        "schema_version": "frontend_cache_pilot_v1",
        "records": shard_records,
    }
    torch.save(payload, shard_path)

    for offset, record in enumerate(shard_records):
        entries.append(
            {
                "clip_id": str(record.get("clip_id", "")),
                "shard": shard_rel_path,
                "offset": int(offset),
                "seq_len": int(record.get("seq_len", 0)),
            }
        )

    shard_records.clear()
    return shard_idx + 1


def main() -> None:
    args = build_parser().parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    sample_limit = int(args.sample_limit) if int(args.sample_limit) > 0 else None
    dataset = STWMDataset(args.data_root, manifest=args.manifest, limit=sample_limit)
    samples = [sample for sample in dataset.samples if len(sample.frame_paths) >= 4]

    offset = max(0, int(args.slice_offset))
    slice_size = max(1, int(args.slice_size))
    selected = samples[offset : offset + slice_size]
    if not selected:
        raise RuntimeError("no samples selected for frontend cache pilot")

    trace_cache_dir = str(args.trace_cache_dir).strip() or "/home/chen034/workspace/stwm/data/cache/trace_summaries"
    semantic_cache_dir = str(args.semantic_cache_dir).strip() or "/home/chen034/workspace/stwm/data/cache/semantic_summaries"

    trace_adapter = TraceAdapter(cache_dir=trace_cache_dir, use_cache=True)
    semantic_adapter = SemanticAdapter(cache_dir=semantic_cache_dir, use_cache=True)

    started = time.perf_counter()
    entries: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    shard_idx = 0

    shard_size = max(1, int(args.shard_size))

    for sample in selected:
        trace_summary = trace_adapter.encode(sample.frame_paths, metadata=sample.metadata, clip_id=sample.clip_id)
        semantic_summary = semantic_adapter.encode(
            sample.text_labels,
            len(sample.frame_paths),
            metadata=sample.metadata,
            clip_id=sample.clip_id,
        )

        seq_len = int(min(trace_summary.centers.shape[0], semantic_summary.class_scores.shape[0]))
        if seq_len <= 1:
            continue

        centers = trace_summary.centers[:seq_len].float().cpu()
        velocities = trace_summary.velocities[:seq_len].float().cpu()
        visibility = trace_summary.visibility[:seq_len].float().cpu()
        trace_features = torch.cat([centers, velocities, visibility], dim=-1)

        sem_text = semantic_summary.text_embeddings[:seq_len].mean(dim=1).float().cpu()
        sem_scores = semantic_summary.class_scores[:seq_len].mean(dim=1).float().cpu()
        semantic_features = torch.cat([sem_text, sem_scores], dim=-1)

        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        mask_paths = metadata.get("mask_paths") if isinstance(metadata.get("mask_paths"), list) else []
        target_label_id = metadata.get("target_label_id")
        try:
            target_label_id = int(target_label_id) if target_label_id is not None else None
        except (TypeError, ValueError):
            target_label_id = None

        mask_ratios: list[float] = []
        for i in range(seq_len):
            if i < len(mask_paths):
                mask_ratios.append(_read_mask_ratio(str(mask_paths[i]), target_label_id))
            else:
                mask_ratios.append(mask_ratios[-1] if mask_ratios else 0.0)
        mask_ratio_t = torch.tensor(mask_ratios, dtype=torch.float32)

        teacher_objectness_prior = (0.6 * mask_ratio_t + 0.4 * visibility[:, 0]).clamp(0.0, 1.0)
        query_target_frame_idx = int(torch.argmax(teacher_objectness_prior).item())
        reappearance = _reappearance_indices(visibility[:, 0], min_gap=1)

        record = {
            "clip_id": str(sample.clip_id),
            "seq_len": int(seq_len),
            "trace_features": trace_features,
            "semantic_features": semantic_features,
            "target_trajectory": centers,
            "target_visibility": visibility,
            "target_semantic_probs": sem_scores,
            "mask_ratios": mask_ratio_t,
            "teacher_objectness_prior": teacher_objectness_prior,
            "metadata": {
                "dataset": str(metadata.get("dataset", "unknown")),
                "manifest_path": str(metadata.get("manifest_path", "")),
                "manifest_hash": str(metadata.get("manifest_hash", "")),
                "target_label_id": target_label_id,
                "text_labels": [str(x) for x in sample.text_labels],
                "num_frames": int(len(sample.frame_paths)),
                "num_masks": int(len(mask_paths)),
                "query_target_frame_idx": int(query_target_frame_idx),
                "reappearance_indices": [int(x) for x in reappearance],
                "frame_paths_head": [str(x) for x in sample.frame_paths[:2]],
                "frame_paths_tail": [str(x) for x in sample.frame_paths[-2:]],
                "mask_paths_head": [str(x) for x in mask_paths[:2]],
                "mask_paths_tail": [str(x) for x in mask_paths[-2:]],
            },
        }
        shard_records.append(record)

        manifest_rows.append(
            {
                "clip_id": str(sample.clip_id),
                "frame_paths": [str(x) for x in sample.frame_paths],
                "text_labels": [str(x) for x in sample.text_labels],
                "metadata": metadata,
            }
        )

        if len(shard_records) >= shard_size:
            shard_idx = _flush_shard(
                shard_records=shard_records,
                shard_idx=shard_idx,
                shards_dir=shards_dir,
                entries=entries,
            )

    shard_idx = _flush_shard(
        shard_records=shard_records,
        shard_idx=shard_idx,
        shards_dir=shards_dir,
        entries=entries,
    )

    if not entries:
        raise RuntimeError("no frontend cache records were created")

    index_payload = {
        "schema_version": "frontend_cache_pilot_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(args.manifest),
        "slice_offset": int(offset),
        "slice_size": int(slice_size),
        "selected_count": int(len(entries)),
        "shard_size": int(shard_size),
        "shard_count": int(shard_idx),
        "entries": entries,
    }
    (output_dir / "index.json").write_text(json.dumps(index_payload, indent=2))

    pilot_manifest_path = output_dir / "pilot_manifest.json"
    pilot_manifest_path.write_text(json.dumps(manifest_rows, indent=2))

    summary_payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": float(time.perf_counter() - started),
        "output_dir": str(output_dir),
        "manifest": str(args.manifest),
        "pilot_manifest": str(pilot_manifest_path),
        "selected_count": int(len(entries)),
        "shard_count": int(shard_idx),
        "schema_version": "frontend_cache_pilot_v1",
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print(str(output_dir / "index.json"))


if __name__ == "__main__":
    main()
