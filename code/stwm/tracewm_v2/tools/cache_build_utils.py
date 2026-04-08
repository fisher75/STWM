from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List
import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_clip_starts(num_frames: int, total_len: int, stride: int, max_clips: int) -> List[int]:
    if num_frames < total_len:
        return []

    candidates = list(range(0, max(num_frames - total_len + 1, 1), max(stride, 1)))
    if not candidates:
        candidates = [0]

    if len(candidates) <= max_clips:
        return candidates

    idx = np.linspace(0, len(candidates) - 1, num=max_clips)
    picks = sorted({candidates[int(round(i))] for i in idx})
    if not picks:
        return [candidates[0]]
    return picks


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def deterministic_split_from_scene_name(scene_name: str) -> str:
    digits = "".join(ch for ch in scene_name if ch.isdigit())
    if digits:
        bucket = int(digits) % 10
    else:
        bucket = abs(hash(scene_name)) % 10

    if bucket < 8:
        return "train"
    if bucket == 8:
        return "val"
    return "test"
