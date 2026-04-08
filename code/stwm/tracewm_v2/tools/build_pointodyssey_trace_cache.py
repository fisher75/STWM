#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
from PIL import Image

from stwm.tracewm_v2.constants import DATA_ROOT, DATE_TAG, TRACE_CACHE_ROOT
from stwm.tracewm_v2.tools.cache_build_utils import now_iso, select_clip_starts


REQUIRED_KEYS = ["trajs_2d", "trajs_3d", "valids", "visibs", "intrinsics", "extrinsics"]
TRACK_SOURCE = "pointodyssey_anno_real_trajs"


def parse_args() -> Any:
    parser = ArgumentParser(description="Build Stage1 v2 real trace cache from PointOdyssey anno.npz")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--source-root", default="")
    parser.add_argument("--cache-root", default=str(TRACE_CACHE_ROOT / "pointodyssey"))
    parser.add_argument(
        "--index-json",
        default=str(DATA_ROOT / "_manifests" / f"stage1_v2_pointodyssey_cache_index_{DATE_TAG}.json"),
    )
    parser.add_argument(
        "--skipped-manifest-json",
        default=str(DATA_ROOT / "_manifests" / f"stage1_v2_pointodyssey_skipped_{DATE_TAG}.json"),
    )
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--fut-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-clips-per-scene", type=int, default=3)
    parser.add_argument("--max-tracks", type=int, default=128)
    parser.add_argument("--min-valid-frames", type=int, default=12)
    parser.add_argument("--max-anno-bytes", type=int, default=900000000)
    parser.add_argument("--max-scenes-per-split", type=int, default=0)
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    return parser.parse_args()


def _discover_frame_paths(scene_dir: Path) -> Tuple[List[str], int, int]:
    rgb_dir = scene_dir / "rgbs"
    rgb_files = sorted([*rgb_dir.glob("*.jpg"), *rgb_dir.glob("*.jpeg"), *rgb_dir.glob("*.png")])
    if not rgb_files:
        return [], 960, 540

    try:
        size = Image.open(rgb_files[0]).size
        width, height = int(size[0]), int(size[1])
    except Exception:
        width, height = 960, 540

    return [str(p) for p in rgb_files], max(width, 2), max(height, 2)


def _valid_or_empty(arr: np.ndarray, shape_tail: Tuple[int, ...]) -> np.ndarray:
    if arr.ndim != len(shape_tail) + 1:
        return np.zeros((0,) + shape_tail, dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _skip_record(split: str, scene_dir: Path, anno_path: Path, anno_size_bytes: int, reason: str) -> Dict[str, Any]:
    return {
        "scene_id": scene_dir.name,
        "split": split,
        "anno_path": str(anno_path),
        "anno_size_bytes": int(anno_size_bytes),
        "skip_reason": reason,
    }


def _process_scene(
    split: str,
    scene_dir: Path,
    cache_root: Path,
    total_len: int,
    stride: int,
    max_clips_per_scene: int,
    max_tracks: int,
    min_valid_frames: int,
    max_anno_bytes: int,
) -> Tuple[List[Dict[str, Any]], str | None, Dict[str, Any] | None]:
    anno_path = scene_dir / "anno.npz"
    anno_size_bytes = 0
    if not anno_path.exists():
        return [], "missing_anno", _skip_record(split, scene_dir, anno_path, 0, "missing_anno")

    try:
        anno_size_bytes = int(anno_path.stat().st_size)
    except Exception:
        return [], "anno_stat_error", _skip_record(split, scene_dir, anno_path, -1, "anno_stat_error")

    if max_anno_bytes > 0:
        if anno_size_bytes > int(max_anno_bytes):
            return [], "anno_too_large", _skip_record(split, scene_dir, anno_path, anno_size_bytes, "anno_too_large")

    try:
        payload = np.load(anno_path, allow_pickle=True)
    except Exception:
        return [], "anno_load_error", _skip_record(split, scene_dir, anno_path, anno_size_bytes, "anno_load_error")

    for key in REQUIRED_KEYS:
        if key not in payload.files:
            reason = f"missing_key_{key}"
            return [], reason, _skip_record(split, scene_dir, anno_path, anno_size_bytes, reason)

    tracks_2d_raw = np.asarray(payload["trajs_2d"], dtype=np.float32)
    valids_raw = np.asarray(payload["valids"], dtype=bool)
    visibs_raw = np.asarray(payload["visibs"], dtype=bool)

    if tracks_2d_raw.ndim != 3 or tracks_2d_raw.shape[-1] != 2:
        return [], "bad_trajs_2d_shape", _skip_record(split, scene_dir, anno_path, anno_size_bytes, "bad_trajs_2d_shape")

    t_len, k_len, _ = tracks_2d_raw.shape
    if valids_raw.shape != (t_len, k_len) or visibs_raw.shape != (t_len, k_len):
        return [], "bad_mask_shape", _skip_record(split, scene_dir, anno_path, anno_size_bytes, "bad_mask_shape")

    tracks_3d_raw = np.asarray(payload["trajs_3d"])
    if tracks_3d_raw.ndim == 3 and tracks_3d_raw.shape == (t_len, k_len, 3):
        tracks_3d = tracks_3d_raw.astype(np.float32, copy=False)
    else:
        tracks_3d = np.zeros((t_len, k_len, 3), dtype=np.float32)

    intrinsics_raw = np.asarray(payload["intrinsics"])
    extrinsics_raw = np.asarray(payload["extrinsics"])

    frame_paths, width, height = _discover_frame_paths(scene_dir)

    tracks_2d = tracks_2d_raw.copy()
    tracks_2d[..., 0] = tracks_2d[..., 0] / float(width - 1)
    tracks_2d[..., 1] = tracks_2d[..., 1] / float(height - 1)

    finite_2d = np.isfinite(tracks_2d).all(axis=-1)
    finite_3d = np.isfinite(tracks_3d).all(axis=-1)
    valid = valids_raw & finite_2d & finite_3d
    visibility = visibs_raw & valid

    valid_counts = valid.sum(axis=0)
    vis_counts = visibility.sum(axis=0)
    candidate = np.where(valid_counts >= int(min_valid_frames))[0]
    if candidate.size == 0:
        return [], "no_valid_tracks", _skip_record(split, scene_dir, anno_path, anno_size_bytes, "no_valid_tracks")

    order = np.argsort(-vis_counts[candidate])
    select = candidate[order][: max_tracks]
    if select.size == 0:
        return [], "empty_track_selection", _skip_record(split, scene_dir, anno_path, anno_size_bytes, "empty_track_selection")

    starts = select_clip_starts(
        num_frames=t_len,
        total_len=total_len,
        stride=max(stride, 1),
        max_clips=max(max_clips_per_scene, 1),
    )
    if not starts:
        return [], "too_short_for_clip", _skip_record(split, scene_dir, anno_path, anno_size_bytes, "too_short_for_clip")

    tracks_2d = np.where(valid[..., None], tracks_2d, 0.0)
    tracks_3d = np.where(valid[..., None], tracks_3d, 0.0)

    entries: List[Dict[str, Any]] = []
    for st in starts:
        ed = st + total_len
        c2d = tracks_2d[st:ed, select]
        c3d = tracks_3d[st:ed, select]
        cvalid = valid[st:ed, select]
        cvis = visibility[st:ed, select]

        if not bool(cvalid.any()):
            continue

        cintr = np.zeros((0, 3, 3), dtype=np.float32)
        if intrinsics_raw.ndim == 3 and intrinsics_raw.shape[0] >= ed:
            cintr = intrinsics_raw[st:ed].astype(np.float32, copy=False)

        cextr = np.zeros((0, 4, 4), dtype=np.float32)
        if extrinsics_raw.ndim == 3 and extrinsics_raw.shape[0] >= ed:
            cextr = extrinsics_raw[st:ed].astype(np.float32, copy=False)

        if frame_paths and len(frame_paths) >= ed:
            cframes = frame_paths[st:ed]
        else:
            cframes = [f"pointodyssey://{scene_dir.name}/frame_{i:05d}" for i in range(st, ed)]

        clip_id = f"pointodyssey_{split}_{scene_dir.name}_{st:05d}"
        out_path = cache_root / split / scene_dir.name / f"clip_{st:05d}.npz"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            out_path,
            dataset=np.array("pointodyssey"),
            split=np.array(split),
            clip_id=np.array(clip_id),
            source_ref=np.array(str(anno_path)),
            track_source=np.array(TRACK_SOURCE),
            tracks_2d=c2d.astype(np.float32, copy=False),
            tracks_3d=c3d.astype(np.float32, copy=False),
            valid=cvalid.astype(bool, copy=False),
            visibility=cvis.astype(bool, copy=False),
            point_ids=select.astype(np.int64, copy=False),
            intrinsics=cintr,
            extrinsics=cextr,
            frame_paths=np.asarray(cframes, dtype=object),
        )

        entries.append(
            {
                "dataset": "pointodyssey",
                "split": split,
                "scene_name": scene_dir.name,
                "clip_id": clip_id,
                "cache_path": str(out_path),
                "source_ref": str(anno_path),
                "track_source": TRACK_SOURCE,
                "num_frames": int(total_len),
                "num_tokens": int(select.size),
            }
        )

    if not entries:
        return [], "no_written_clips", _skip_record(split, scene_dir, anno_path, anno_size_bytes, "no_written_clips")
    return entries, None, None


def _scene_dirs(source_root: Path, split: str, max_scenes_per_split: int) -> List[Path]:
    split_dir = source_root / split
    if not split_dir.exists() or not split_dir.is_dir():
        return []

    seq_dirs = [
        p
        for p in sorted(split_dir.iterdir(), key=lambda q: q.name)
        if p.is_dir()
    ]
    if max_scenes_per_split > 0:
        seq_dirs = seq_dirs[:max_scenes_per_split]
    return seq_dirs


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    source_root = Path(args.source_root) if args.source_root else data_root / "pointodyssey"
    cache_root = Path(args.cache_root)
    index_json = Path(args.index_json)
    skipped_manifest_json = Path(args.skipped_manifest_json)

    total_len = int(args.obs_len) + int(args.fut_len)
    all_entries: List[Dict[str, Any]] = []
    skipped = Counter()
    skipped_records: List[Dict[str, Any]] = []
    split_scene_counts: Dict[str, int] = defaultdict(int)
    split_clip_counts: Dict[str, int] = defaultdict(int)

    for split in args.splits:
        for scene_dir in _scene_dirs(source_root, split, int(args.max_scenes_per_split)):
            split_scene_counts[split] += 1
            entries, reason, skip_info = _process_scene(
                split=split,
                scene_dir=scene_dir,
                cache_root=cache_root,
                total_len=total_len,
                stride=int(args.stride),
                max_clips_per_scene=int(args.max_clips_per_scene),
                max_tracks=int(args.max_tracks),
                min_valid_frames=int(args.min_valid_frames),
                max_anno_bytes=int(args.max_anno_bytes),
            )
            if reason is not None:
                skipped[reason] += 1
                if skip_info is None:
                    skip_info = _skip_record(split, scene_dir, scene_dir / "anno.npz", 0, reason)
                skipped_records.append(skip_info)
                continue

            all_entries.extend(entries)
            split_clip_counts[split] += len(entries)

    payload = {
        "generated_at_utc": now_iso(),
        "dataset": "pointodyssey",
        "source_root": str(source_root),
        "cache_root": str(cache_root),
        "index_path": str(index_json),
        "skipped_manifest_path": str(skipped_manifest_json),
        "track_source": TRACK_SOURCE,
        "obs_len": int(args.obs_len),
        "fut_len": int(args.fut_len),
        "clip_len": int(total_len),
        "config": {
            "stride": int(args.stride),
            "max_clips_per_scene": int(args.max_clips_per_scene),
            "max_tracks": int(args.max_tracks),
            "min_valid_frames": int(args.min_valid_frames),
            "max_anno_bytes": int(args.max_anno_bytes),
            "max_scenes_per_split": int(args.max_scenes_per_split),
            "splits": [str(s) for s in args.splits],
        },
        "stats": {
            "scene_counts": {k: int(v) for k, v in sorted(split_scene_counts.items())},
            "clip_counts": {k: int(v) for k, v in sorted(split_clip_counts.items())},
            "total_scenes": int(sum(split_scene_counts.values())),
            "total_clips": int(len(all_entries)),
            "skipped_reasons": {k: int(v) for k, v in sorted(skipped.items())},
        },
        "entries": all_entries,
    }

    index_json.parent.mkdir(parents=True, exist_ok=True)
    index_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    skipped_payload = {
        "generated_at_utc": now_iso(),
        "dataset": "pointodyssey",
        "index_path": str(index_json),
        "skipped_scene_count": int(len(skipped_records)),
        "skipped_reasons": {k: int(v) for k, v in sorted(skipped.items())},
        "records": skipped_records,
    }
    skipped_manifest_json.parent.mkdir(parents=True, exist_ok=True)
    skipped_manifest_json.write_text(json.dumps(skipped_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[pointodyssey-cache] wrote index: {index_json}")
    print(f"[pointodyssey-cache] wrote skipped_manifest: {skipped_manifest_json}")
    print(f"[pointodyssey-cache] total_clips={len(all_entries)} total_scenes={sum(split_scene_counts.values())}")


if __name__ == "__main__":
    main()
