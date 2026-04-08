#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np

from stwm.tracewm_v2.constants import DATA_ROOT, DATE_TAG, TRACE_CACHE_ROOT
from stwm.tracewm_v2.tools.cache_build_utils import (
    deterministic_split_from_scene_name,
    now_iso,
    select_clip_starts,
)


TRACK_SOURCE = "kubric_metadata_instance_tracks_real"


def parse_args() -> Any:
    parser = ArgumentParser(description="Build Stage1 v2 real trace cache from Kubric metadata.json")
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--raw-root", default="")
    parser.add_argument("--cache-root", default=str(TRACE_CACHE_ROOT / "kubric"))
    parser.add_argument(
        "--index-json",
        default=str(DATA_ROOT / "_manifests" / f"stage1_v2_kubric_cache_index_{DATE_TAG}.json"),
    )
    parser.add_argument(
        "--first-wave-mode",
        default="panning_raw_first_wave",
        choices=["panning_raw_first_wave"],
        help="First-wave mode declaration. Only one mode is allowed in first-wave runs.",
    )
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--fut-len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-clips-per-scene", type=int, default=1)
    parser.add_argument("--max-scenes", type=int, default=512)
    parser.add_argument("--max-instances", type=int, default=32)
    parser.add_argument("--min-valid-frames", type=int, default=6)
    parser.add_argument("--min-visible-pixels", type=float, default=1.0)
    return parser.parse_args()


def _infer_num_frames(meta: Dict[str, Any]) -> int:
    m = meta.get("metadata", {}) if isinstance(meta, dict) else {}
    num_frames = int(m.get("num_frames", 0)) if isinstance(m, dict) else 0
    if num_frames > 0:
        return num_frames

    instances = meta.get("instances", []) if isinstance(meta, dict) else []
    if instances and isinstance(instances[0], dict):
        return int(len(instances[0].get("image_positions", [])))
    return 0


def _load_scene_tracks(
    scene_dir: Path,
    num_frames: int,
    max_instances: int,
    min_valid_frames: int,
    min_visible_pixels: float,
) -> Tuple[Dict[str, np.ndarray] | None, str | None]:
    meta_path = scene_dir / "metadata.json"
    if not meta_path.exists():
        return None, "missing_metadata"

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None, "metadata_load_error"

    if num_frames <= 0:
        num_frames = _infer_num_frames(payload)
    if num_frames <= 0:
        return None, "bad_num_frames"

    instances = payload.get("instances", [])
    if not isinstance(instances, list) or not instances:
        return None, "empty_instances"

    token_rows: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    for idx, inst in enumerate(instances):
        if not isinstance(inst, dict):
            continue

        ip = np.asarray(inst.get("image_positions", []), dtype=np.float32)
        p3 = np.asarray(inst.get("positions", []), dtype=np.float32)
        vv = np.asarray(inst.get("visibility", []), dtype=np.float32)

        if ip.shape != (num_frames, 2):
            continue
        if p3.shape != (num_frames, 3):
            continue
        if vv.shape != (num_frames,):
            continue

        finite = np.isfinite(ip).all(axis=-1) & np.isfinite(p3).all(axis=-1)
        valid = finite & (vv > float(min_visible_pixels))
        if int(valid.sum()) < int(min_valid_frames):
            continue

        token_rows.append((idx + 1, ip, p3, valid))

    if not token_rows:
        return None, "no_valid_instances"

    token_rows.sort(key=lambda row: int(row[3].sum()), reverse=True)
    token_rows = token_rows[: max(max_instances, 1)]

    point_ids = np.asarray([row[0] for row in token_rows], dtype=np.int64)
    tracks_2d = np.stack([row[1] for row in token_rows], axis=1)
    tracks_3d = np.stack([row[2] for row in token_rows], axis=1)
    valid = np.stack([row[3] for row in token_rows], axis=1)
    visibility = valid.copy()

    tracks_2d = np.where(valid[..., None], tracks_2d, 0.0)
    tracks_3d = np.where(valid[..., None], tracks_3d, 0.0)

    cam = payload.get("camera", {}) if isinstance(payload, dict) else {}
    kmat = np.asarray(cam.get("K", []), dtype=np.float32)
    intrinsics = np.zeros((0, 3, 3), dtype=np.float32)
    if kmat.shape == (3, 3):
        intrinsics = np.repeat(kmat[None, ...], repeats=num_frames, axis=0)

    extrinsics = np.zeros((0, 4, 4), dtype=np.float32)

    rgba_files = sorted(scene_dir.glob("rgba_*.png"))
    if len(rgba_files) >= num_frames:
        frame_paths = [str(p) for p in rgba_files[:num_frames]]
    else:
        frame_paths = [f"kubric://{scene_dir.name}/frame_{i:05d}" for i in range(num_frames)]

    return {
        "tracks_2d": tracks_2d.astype(np.float32, copy=False),
        "tracks_3d": tracks_3d.astype(np.float32, copy=False),
        "valid": valid.astype(bool, copy=False),
        "visibility": visibility.astype(bool, copy=False),
        "point_ids": point_ids,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "frame_paths": np.asarray(frame_paths, dtype=object),
        "meta_path": str(meta_path),
    }, None


def main() -> None:
    args = parse_args()

    data_root = Path(args.data_root)
    raw_root = Path(args.raw_root) if args.raw_root else data_root / "kubric" / "panning_movi_e_full_20260407" / "raw"
    cache_root = Path(args.cache_root)
    index_json = Path(args.index_json)

    total_len = int(args.obs_len) + int(args.fut_len)

    scene_dirs = [p for p in sorted(raw_root.glob("scene_*")) if p.is_dir()]
    if int(args.max_scenes) > 0:
        scene_dirs = scene_dirs[: int(args.max_scenes)]

    all_entries: List[Dict[str, Any]] = []
    skipped = Counter()
    split_scene_counts: Dict[str, int] = defaultdict(int)
    split_clip_counts: Dict[str, int] = defaultdict(int)

    for scene_dir in scene_dirs:
        split = deterministic_split_from_scene_name(scene_dir.name)
        split_scene_counts[split] += 1

        bundle, reason = _load_scene_tracks(
            scene_dir=scene_dir,
            num_frames=0,
            max_instances=int(args.max_instances),
            min_valid_frames=int(args.min_valid_frames),
            min_visible_pixels=float(args.min_visible_pixels),
        )
        if reason is not None or bundle is None:
            skipped[reason or "load_failed"] += 1
            continue

        num_frames = int(bundle["tracks_2d"].shape[0])
        starts = select_clip_starts(
            num_frames=num_frames,
            total_len=total_len,
            stride=max(int(args.stride), 1),
            max_clips=max(int(args.max_clips_per_scene), 1),
        )
        if not starts:
            skipped["too_short_for_clip"] += 1
            continue

        for st in starts:
            ed = st + total_len
            c2d = bundle["tracks_2d"][st:ed]
            c3d = bundle["tracks_3d"][st:ed]
            cvalid = bundle["valid"][st:ed]
            cvis = bundle["visibility"][st:ed]
            cintr = (
                bundle["intrinsics"][st:ed]
                if bundle["intrinsics"].ndim == 3 and bundle["intrinsics"].shape[0] >= ed
                else np.zeros((0, 3, 3), dtype=np.float32)
            )
            cextr = np.zeros((0, 4, 4), dtype=np.float32)

            if not bool(cvalid.any()):
                continue

            cframes = bundle["frame_paths"][st:ed].tolist()
            clip_id = f"kubric_{split}_{scene_dir.name}_{st:05d}"
            out_path = cache_root / split / scene_dir.name / f"clip_{st:05d}.npz"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            np.savez_compressed(
                out_path,
                dataset=np.array("kubric"),
                split=np.array(split),
                clip_id=np.array(clip_id),
                source_ref=np.array(bundle["meta_path"]),
                track_source=np.array(TRACK_SOURCE),
                tracks_2d=c2d.astype(np.float32, copy=False),
                tracks_3d=c3d.astype(np.float32, copy=False),
                valid=cvalid.astype(bool, copy=False),
                visibility=cvis.astype(bool, copy=False),
                point_ids=bundle["point_ids"].astype(np.int64, copy=False),
                intrinsics=cintr.astype(np.float32, copy=False),
                extrinsics=cextr,
                frame_paths=np.asarray(cframes, dtype=object),
            )

            all_entries.append(
                {
                    "dataset": "kubric",
                    "split": split,
                    "scene_name": scene_dir.name,
                    "clip_id": clip_id,
                    "cache_path": str(out_path),
                    "source_ref": str(bundle["meta_path"]),
                    "track_source": TRACK_SOURCE,
                    "num_frames": int(total_len),
                    "num_tokens": int(bundle["point_ids"].shape[0]),
                }
            )
            split_clip_counts[split] += 1

    payload = {
        "generated_at_utc": now_iso(),
        "dataset": "kubric",
        "first_wave_mode": str(args.first_wave_mode),
        "source_root": str(raw_root),
        "cache_root": str(cache_root),
        "index_path": str(index_json),
        "track_source": TRACK_SOURCE,
        "obs_len": int(args.obs_len),
        "fut_len": int(args.fut_len),
        "clip_len": int(total_len),
        "config": {
            "stride": int(args.stride),
            "max_clips_per_scene": int(args.max_clips_per_scene),
            "max_scenes": int(args.max_scenes),
            "max_instances": int(args.max_instances),
            "min_valid_frames": int(args.min_valid_frames),
            "min_visible_pixels": float(args.min_visible_pixels),
            "first_wave_mode": str(args.first_wave_mode),
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

    print(f"[kubric-cache] wrote index: {index_json}")
    print(f"[kubric-cache] total_clips={len(all_entries)} total_scenes={sum(split_scene_counts.values())}")


if __name__ == "__main__":
    main()
