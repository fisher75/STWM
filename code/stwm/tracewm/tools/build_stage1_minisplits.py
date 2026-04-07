#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import pickle
import random

import numpy as np

from stwm.tracewm.data_contract import load_stage1_contract


DATA_ROOT = Path("/home/chen034/workspace/data")
STWM_ROOT = Path("/home/chen034/workspace/stwm")

POINT_AFTER = DATA_ROOT / "_manifests" / "pointodyssey_hard_complete_after_20260407.json"
OUT_MINISPLITS = DATA_ROOT / "_manifests" / "stage1_minisplits_20260408.json"
OUT_PROTOCOL_DOC = STWM_ROOT / "docs" / "STAGE1_MINISPLIT_PROTOCOL_20260408.md"
TAPVID_CACHE_DIR = STWM_ROOT / "outputs" / "stage1_minisplit_cache" / "tapvid_20260408"

SEED = 20260408


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_bucket(text: str, base: int = 10) -> int:
    digest = sha1(text.encode("utf-8")).hexdigest()
    return int(digest, 16) % base


def count_rgb_frames(seq_dir: Path) -> int:
    rgb = seq_dir / "rgbs"
    if not rgb.exists():
        return 0
    return len([*rgb.glob("*.jpg"), *rgb.glob("*.jpeg"), *rgb.glob("*.png")])


def select_diverse_sequences(
    rng: random.Random,
    split_name: str,
    sequence_names: List[str],
    point_root: Path,
    target_n: int,
) -> List[Dict[str, Any]]:
    scored: List[Tuple[str, int]] = []
    for seq in sequence_names:
        seq_dir = point_root / split_name / seq
        scored.append((seq, count_rgb_frames(seq_dir)))

    scored = [x for x in scored if x[1] > 0]
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)

    if not scored_sorted:
        return []

    top_keep = min(max(2, target_n // 3), len(scored_sorted))
    selected = scored_sorted[:top_keep]

    remaining = scored_sorted[top_keep:]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, target_n - len(selected))])

    selected = selected[:target_n]

    out: List[Dict[str, Any]] = []
    for seq, frame_count in selected:
        out.append(
            {
                "sequence_name": seq,
                "start_index": 0,
                "clip_id": f"pointodyssey_{split_name}_{seq}_00000",
                "frame_count": frame_count,
                "selection_reason": "long_motion_plus_seeded_diversity",
            }
        )
    return out


def build_pointodyssey_minisplits(rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    point_root = DATA_ROOT / "pointodyssey"
    payload = load_json(POINT_AFTER)

    train_seq = list(payload.get("split_sequence_names", {}).get("train", []))
    val_seq = list(payload.get("split_sequence_names", {}).get("val", []))

    train_mini = select_diverse_sequences(rng, "train", train_seq, point_root, target_n=12)
    val_mini = select_diverse_sequences(rng, "val", val_seq, point_root, target_n=6)

    return {
        "train_mini": train_mini,
        "val_mini": val_mini,
    }


def build_kubric_minisplits(rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    movi_root = DATA_ROOT / "kubric" / "tfds" / "movi_e"
    tfrecords = sorted([
        *movi_root.rglob("*.tfrecord"),
        *movi_root.rglob("*.tfrecord-*"),
        *movi_root.rglob("*.tfrecord.gz"),
    ])

    train_pool: List[Path] = []
    val_pool: List[Path] = []
    for p in tfrecords:
        bucket = stable_bucket(str(p), base=10)
        if bucket < 8:
            train_pool.append(p)
        else:
            val_pool.append(p)

    rng.shuffle(train_pool)
    rng.shuffle(val_pool)

    train_pick = train_pool[:24]
    val_pick = val_pool[:8]

    train_mini = [
        {
            "clip_id": f"kubric_train_{i:04d}_{p.stem}",
            "tfrecord_path": str(p),
            "selection_reason": "stable_hash_split_seeded_shuffle",
        }
        for i, p in enumerate(train_pick)
    ]
    val_mini = [
        {
            "clip_id": f"kubric_val_{i:04d}_{p.stem}",
            "tfrecord_path": str(p),
            "selection_reason": "stable_hash_split_seeded_shuffle",
        }
        for i, p in enumerate(val_pick)
    ]

    return {"train_mini": train_mini, "val_mini": val_mini}


def _select_tapvid_entries_davis(rng: random.Random, davis_data: Dict[str, Any], k: int) -> List[Tuple[str, Dict[str, Any]]]:
    scored: List[Tuple[str, int, Dict[str, Any]]] = []
    for name, item in davis_data.items():
        if not isinstance(item, dict):
            continue
        points = item.get("points")
        occluded = item.get("occluded")
        if not isinstance(points, np.ndarray) or not isinstance(occluded, np.ndarray):
            continue
        if points.ndim != 3 or occluded.ndim != 2:
            continue
        score = int(points.shape[0] * points.shape[1])
        scored.append((name, score, item))

    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    if not scored_sorted:
        return []

    picks = [scored_sorted[0]]
    if len(scored_sorted) > 2:
        picks.append(scored_sorted[len(scored_sorted) // 2])
    if len(scored_sorted) > 1:
        picks.append(scored_sorted[-1])

    unique: Dict[str, Dict[str, Any]] = {}
    for name, _, item in picks:
        unique[name] = item

    rest = [x for x in scored_sorted if x[0] not in unique]
    rng.shuffle(rest)
    for name, _, item in rest:
        if len(unique) >= k:
            break
        unique[name] = item

    return list(unique.items())[:k]


def _select_tapvid_entries_rgb(rng: random.Random, rgb_data: List[Any], k: int) -> List[Tuple[str, Dict[str, Any]]]:
    valid: List[Tuple[int, Dict[str, Any]]] = []
    for i, item in enumerate(rgb_data):
        if not isinstance(item, dict):
            continue
        points = item.get("points")
        occluded = item.get("occluded")
        if not isinstance(points, np.ndarray) or not isinstance(occluded, np.ndarray):
            continue
        if points.ndim != 3 or occluded.ndim != 2:
            continue
        valid.append((i, item))

    if not valid:
        return []

    candidate_indices = [0, len(valid) // 2, len(valid) - 1]
    picks: Dict[int, Dict[str, Any]] = {}
    for idx in candidate_indices:
        real_idx, item = valid[max(0, min(idx, len(valid) - 1))]
        picks[real_idx] = item

    others = [x for x in valid if x[0] not in picks]
    rng.shuffle(others)
    for real_idx, item in others:
        if len(picks) >= k:
            break
        picks[real_idx] = item

    return [(f"rgb_{idx:04d}", item) for idx, item in sorted(picks.items(), key=lambda x: x[0])][:k]


def build_tapvid_minisplit(rng: random.Random) -> Dict[str, List[Dict[str, Any]]]:
    davis_pkl = DATA_ROOT / "tapvid" / "davis" / "tapvid_davis" / "tapvid_davis.pkl"
    rgb_pkl = DATA_ROOT / "tapvid" / "rgb_stacking" / "tapvid_rgb_stacking" / "tapvid_rgb_stacking.pkl"

    TAPVID_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with davis_pkl.open("rb") as f:
        davis_data = pickle.load(f)
    with rgb_pkl.open("rb") as f:
        rgb_data = pickle.load(f)

    eval_records: List[Dict[str, Any]] = []

    for clip_name, item in _select_tapvid_entries_davis(rng, davis_data, k=3):
        points = np.asarray(item["points"], dtype=np.float32)
        occluded = np.asarray(item["occluded"], dtype=np.bool_)
        cache_path = TAPVID_CACHE_DIR / f"davis_{clip_name}.npz"
        np.savez_compressed(cache_path, points=points, occluded=occluded)
        eval_records.append(
            {
                "clip_id": f"tapvid_davis_{clip_name}",
                "variant": "davis",
                "cache_npz": str(cache_path),
                "num_points": int(points.shape[0]),
                "num_frames": int(points.shape[1]),
                "selection_reason": "real_points_cache_diverse_length",
            }
        )

    for clip_name, item in _select_tapvid_entries_rgb(rng, rgb_data, k=3):
        points = np.asarray(item["points"], dtype=np.float32)
        occluded = np.asarray(item["occluded"], dtype=np.bool_)
        cache_path = TAPVID_CACHE_DIR / f"rgb_{clip_name}.npz"
        np.savez_compressed(cache_path, points=points, occluded=occluded)
        eval_records.append(
            {
                "clip_id": f"tapvid_rgb_{clip_name}",
                "variant": "rgb_stacking",
                "cache_npz": str(cache_path),
                "num_points": int(points.shape[0]),
                "num_frames": int(points.shape[1]),
                "selection_reason": "real_points_cache_diverse_length",
            }
        )

    return {"eval_mini": eval_records}


def build_tapvid3d_minisplit() -> Dict[str, List[Dict[str, Any]]]:
    base = DATA_ROOT / "tapvid3d" / "minival_dataset"
    picks: List[Dict[str, Any]] = []

    source_pick_n = {"pstudio": 4, "adt": 4, "drivetrack": 4}
    for source, n in source_pick_n.items():
        src_dir = base / source
        if not src_dir.exists() or not src_dir.is_dir():
            continue
        paths = sorted(src_dir.glob("*.npz"))[:n]
        for p in paths:
            picks.append(
                {
                    "clip_id": f"tapvid3d_{source}_{p.stem}",
                    "source": source,
                    "npz_path": str(p),
                    "selection_reason": "source_balanced_fixed_order",
                }
            )

    return {"eval_mini": picks}


def write_protocol_doc(payload: Dict[str, Any]) -> None:
    point_train_n = len(payload["datasets"]["pointodyssey"]["train_mini"])
    point_val_n = len(payload["datasets"]["pointodyssey"]["val_mini"])
    kubric_train_n = len(payload["datasets"]["kubric"]["train_mini"])
    kubric_val_n = len(payload["datasets"]["kubric"]["val_mini"])
    tapvid_eval_n = len(payload["datasets"]["tapvid"]["eval_mini"])
    tapvid3d_eval_n = len(payload["datasets"]["tapvid3d"]["eval_mini"])

    lines = [
        "# Stage 1 Minisplit Protocol (2026-04-08)",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- seed: {payload['seed']}",
        f"- contract_path: {payload['contract_path']}",
        "",
        "## Minisplit Design",
        "",
        "- All selections are deterministic under fixed seed.",
        "- PointOdyssey and Kubric are selected for train_mini/val_mini and must each form independent mini-batches.",
        "- TAP-Vid is eval_mini only.",
        "- TAPVid-3D is limited eval_mini only.",
        "",
        "## Coverage Heuristics",
        "",
        "- Long trajectory preference: keep long-frame clips/sequences in each mini pool.",
        "- Visibility/motion diversity: combine top-length picks and seeded random picks.",
        "- Source balance for TAPVid-3D: include pstudio/adt/drivetrack.",
        "",
        "## Sizes",
        "",
        f"- PointOdyssey train_mini: {point_train_n}",
        f"- PointOdyssey val_mini: {point_val_n}",
        f"- Kubric train_mini: {kubric_train_n}",
        f"- Kubric val_mini: {kubric_val_n}",
        f"- TAP-Vid eval_mini: {tapvid_eval_n}",
        f"- TAPVid-3D eval_mini: {tapvid3d_eval_n}",
    ]
    OUT_PROTOCOL_DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT_PROTOCOL_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rng = random.Random(SEED)

    contract = load_stage1_contract()

    point = build_pointodyssey_minisplits(rng)
    kubric = build_kubric_minisplits(rng)
    tapvid = build_tapvid_minisplit(rng)
    tapvid3d = build_tapvid3d_minisplit()

    payload = {
        "generated_at_utc": now_iso(),
        "seed": SEED,
        "contract_path": contract.get("_contract_path", ""),
        "datasets": {
            "pointodyssey": point,
            "kubric": kubric,
            "tapvid": tapvid,
            "tapvid3d": tapvid3d,
        },
    }

    OUT_MINISPLITS.parent.mkdir(parents=True, exist_ok=True)
    OUT_MINISPLITS.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_protocol_doc(payload)

    print(f"[minisplit] wrote: {OUT_MINISPLITS}")
    print(f"[minisplit] wrote: {OUT_PROTOCOL_DOC}")
    print(
        "[minisplit] sizes "
        f"point(train/val)={len(point['train_mini'])}/{len(point['val_mini'])}, "
        f"kubric(train/val)={len(kubric['train_mini'])}/{len(kubric['val_mini'])}, "
        f"tapvid(eval)={len(tapvid['eval_mini'])}, "
        f"tapvid3d(eval)={len(tapvid3d['eval_mini'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
