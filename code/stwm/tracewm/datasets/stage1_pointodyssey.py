from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _PointClipRecord:
    split: str
    sequence_name: str
    clip_id: str
    frame_paths: List[str]


class Stage1PointOdysseyDataset(Dataset):
    """PointOdyssey Stage 1 adapter with a unified sample dict.

    This adapter is trace/state-oriented and emits deterministic trajectory states
    derived from frame index progression for smoke/tiny-train stabilization.
    """

    def __init__(
        self,
        data_root: str | Path = "/home/chen034/workspace/data",
        split: str = "train_mini",
        minisplit_records: List[Dict[str, Any]] | None = None,
        obs_len: int = 8,
        fut_len: int = 8,
        stride: int = 8,
        max_sequences: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset_root = self.data_root / "pointodyssey"
        self.split = split
        self.obs_len = int(obs_len)
        self.fut_len = int(fut_len)
        self.total_len = self.obs_len + self.fut_len
        self.stride = int(stride)
        self.max_sequences = max_sequences

        self._records: List[_PointClipRecord] = []

        if minisplit_records:
            self._records = self._build_from_minisplit(minisplit_records)
        else:
            self._records = self._discover_records()

        if not self._records:
            self._records = [
                _PointClipRecord(
                    split=self.split,
                    sequence_name="synthetic_pointodyssey",
                    clip_id="pointodyssey_synthetic_0000",
                    frame_paths=[f"synthetic://pointodyssey/frame_{i:05d}.jpg" for i in range(self.total_len)],
                )
            ]

    def _base_split(self) -> str:
        if self.split.startswith("train"):
            return "train"
        if self.split.startswith("val"):
            return "val"
        if self.split.startswith("test"):
            return "test"
        if self.split.startswith("eval"):
            return "val"
        return "train"

    def _sequence_dirs(self, split_name: str) -> Iterable[Path]:
        split_dir = self.dataset_root / split_name
        if not split_dir.exists() or not split_dir.is_dir():
            return []

        out: List[Path] = []
        for entry in sorted(split_dir.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            if all((entry / k).is_dir() for k in ["rgbs", "depths", "masks", "normals"]):
                out.append(entry)
        return out

    def _discover_records(self) -> List[_PointClipRecord]:
        base_split = self._base_split()
        seq_dirs = list(self._sequence_dirs(base_split))
        if self.max_sequences is not None:
            seq_dirs = seq_dirs[: self.max_sequences]

        records: List[_PointClipRecord] = []
        for seq_dir in seq_dirs:
            rgb_files = sorted(
                [
                    *seq_dir.joinpath("rgbs").glob("*.jpg"),
                    *seq_dir.joinpath("rgbs").glob("*.jpeg"),
                    *seq_dir.joinpath("rgbs").glob("*.png"),
                ]
            )
            if len(rgb_files) < self.total_len:
                continue

            starts = [0]
            mid = max(0, (len(rgb_files) - self.total_len) // 2)
            if mid not in starts:
                starts.append(mid)
            tail = max(0, len(rgb_files) - self.total_len)
            if tail not in starts:
                starts.append(tail)

            used = 0
            for st in starts:
                clip_frames = [str(p) for p in rgb_files[st : st + self.total_len]]
                clip_id = f"pointodyssey_{base_split}_{seq_dir.name}_{st:05d}"
                records.append(
                    _PointClipRecord(
                        split=self.split,
                        sequence_name=seq_dir.name,
                        clip_id=clip_id,
                        frame_paths=clip_frames,
                    )
                )
                used += 1
                if self.split.endswith("mini") and used >= 1:
                    break
        return records

    def _build_from_minisplit(self, records: List[Dict[str, Any]]) -> List[_PointClipRecord]:
        out: List[_PointClipRecord] = []
        base_split = self._base_split()

        for item in records:
            seq = str(item.get("sequence_name", "")).strip()
            if not seq:
                continue
            seq_dir = self.dataset_root / base_split / seq
            rgb_dir = seq_dir / "rgbs"
            if not rgb_dir.exists():
                continue

            rgb_files = sorted([*rgb_dir.glob("*.jpg"), *rgb_dir.glob("*.jpeg"), *rgb_dir.glob("*.png")])
            if len(rgb_files) < self.total_len:
                continue

            st = int(item.get("start_index", 0))
            st = max(0, min(st, len(rgb_files) - self.total_len))
            clip_frames = [str(p) for p in rgb_files[st : st + self.total_len]]
            clip_id = str(item.get("clip_id", f"pointodyssey_{base_split}_{seq}_{st:05d}"))
            out.append(
                _PointClipRecord(
                    split=self.split,
                    sequence_name=seq,
                    clip_id=clip_id,
                    frame_paths=clip_frames,
                )
            )
        return out

    def __len__(self) -> int:
        return len(self._records)

    @staticmethod
    def _build_tracks_2d(total_len: int, num_points: int = 32) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, total_len, dtype=torch.float32).view(total_len, 1, 1)
        pid = torch.linspace(0.0, 1.0, num_points, dtype=torch.float32).view(1, num_points, 1)

        x = (0.08 + 0.82 * pid + 0.05 * t) % 1.0
        y = (0.12 + 0.74 * (1.0 - pid) + 0.03 * t) % 1.0
        return torch.cat([x, y], dim=-1)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]

        obs_frames = rec.frame_paths[: self.obs_len]
        fut_frames = rec.frame_paths[self.obs_len : self.total_len]

        tracks2d = self._build_tracks_2d(self.total_len, num_points=32)
        obs_tracks_2d = tracks2d[: self.obs_len]
        fut_tracks_2d = tracks2d[self.obs_len :]

        visibility = torch.ones((self.total_len, tracks2d.shape[1]), dtype=torch.bool)
        point_ids = torch.arange(tracks2d.shape[1], dtype=torch.long)

        sample: Dict[str, Any] = {
            "dataset": "pointodyssey",
            "split": rec.split,
            "clip_id": rec.clip_id,
            "obs_frames": obs_frames,
            "fut_frames": fut_frames,
            "obs_valid": torch.ones(self.obs_len, dtype=torch.bool),
            "fut_valid": torch.ones(self.fut_len, dtype=torch.bool),
            "obs_tracks_2d": obs_tracks_2d,
            "fut_tracks_2d": fut_tracks_2d,
            "obs_tracks_3d": None,
            "fut_tracks_3d": None,
            "visibility": visibility,
            "intrinsics": None,
            "extrinsics": None,
            "point_ids": point_ids,
            "meta": {
                "sequence_name": rec.sequence_name,
                "track_source": "deterministic_trace_from_frame_index",
                "modalities": ["rgbs", "depths", "masks", "normals"],
                "obs_len": self.obs_len,
                "fut_len": self.fut_len,
            },
        }
        return sample
