from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _KubricClipRecord:
    split: str
    clip_id: str
    tfrecord_path: str


class Stage1KubricDataset(Dataset):
    """Kubric Stage 1 adapter (movi_e, trace-only synthetic state interface).

    The adapter uses local tfrecord shard references as clip identity and emits
    deterministic trace/state tensors for Stage 1 tiny-train interface validation.
    """

    def __init__(
        self,
        data_root: str | Path = "/home/chen034/workspace/data",
        split: str = "train_mini",
        minisplit_records: List[Dict[str, Any]] | None = None,
        obs_len: int = 8,
        fut_len: int = 8,
        max_records: int | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset_root = self.data_root / "kubric"
        self.movi_root = self.dataset_root / "tfds" / "movi_e"
        self.split = split
        self.obs_len = int(obs_len)
        self.fut_len = int(fut_len)
        self.total_len = self.obs_len + self.fut_len

        if minisplit_records:
            self._records = self._build_from_minisplit(minisplit_records)
        else:
            self._records = self._discover_records()

        if max_records is not None:
            self._records = self._records[:max_records]

        if not self._records:
            self._records = [
                _KubricClipRecord(
                    split=self.split,
                    clip_id="kubric_synthetic_0000",
                    tfrecord_path="synthetic://kubric/movi_e/0000.tfrecord",
                )
            ]

    def _iter_tfrecords(self) -> Iterable[Path]:
        if not self.movi_root.exists() or not self.movi_root.is_dir():
            return []
        return sorted(
            [
                *self.movi_root.rglob("*.tfrecord"),
                *self.movi_root.rglob("*.tfrecord-*"),
                *self.movi_root.rglob("*.tfrecord.gz"),
            ]
        )

    def _discover_records(self) -> List[_KubricClipRecord]:
        tfrecords = list(self._iter_tfrecords())
        out: List[_KubricClipRecord] = []
        for idx, path in enumerate(tfrecords):
            bucket = idx % 10
            inferred_split = "train_mini" if bucket < 8 else "val_mini"
            if self.split.startswith("train") and not inferred_split.startswith("train"):
                continue
            if self.split.startswith("val") and not inferred_split.startswith("val"):
                continue
            clip_id = f"kubric_{path.stem}_{idx:05d}"
            out.append(
                _KubricClipRecord(
                    split=self.split,
                    clip_id=clip_id,
                    tfrecord_path=str(path),
                )
            )
        return out

    def _build_from_minisplit(self, records: List[Dict[str, Any]]) -> List[_KubricClipRecord]:
        out: List[_KubricClipRecord] = []
        for item in records:
            clip_id = str(item.get("clip_id", "")).strip()
            tfrecord_path = str(item.get("tfrecord_path", "")).strip()
            if not clip_id or not tfrecord_path:
                continue
            out.append(
                _KubricClipRecord(
                    split=self.split,
                    clip_id=clip_id,
                    tfrecord_path=tfrecord_path,
                )
            )
        return out

    @staticmethod
    def _build_tracks(total_len: int, num_points: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(0.0, 1.0, total_len, dtype=torch.float32).view(total_len, 1, 1)
        pid = torch.linspace(0.0, 1.0, num_points, dtype=torch.float32).view(1, num_points, 1)

        x = 0.6 * torch.cos(2.0 * torch.pi * (t + pid))
        y = 0.6 * torch.sin(2.0 * torch.pi * (t * 0.5 + pid))
        z = 0.2 + 0.8 * pid + 0.1 * t

        tracks_3d = torch.cat([x, y, z], dim=-1)
        x2 = (x + 1.0) * 0.5
        y2 = (y + 1.0) * 0.5
        tracks_2d = torch.cat([x2, y2], dim=-1)
        return tracks_2d, tracks_3d

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]

        frame_refs = [f"tfrecord://{rec.tfrecord_path}#frame={i:04d}" for i in range(self.total_len)]
        obs_frames = frame_refs[: self.obs_len]
        fut_frames = frame_refs[self.obs_len :]

        tracks_2d, tracks_3d = self._build_tracks(self.total_len, num_points=48)

        visibility = torch.ones((self.total_len, tracks_3d.shape[1]), dtype=torch.bool)
        point_ids = torch.arange(tracks_3d.shape[1], dtype=torch.long)
        intrinsics = torch.tensor([128.0, 128.0, 128.0, 128.0], dtype=torch.float32)

        sample: Dict[str, Any] = {
            "dataset": "kubric",
            "split": rec.split,
            "clip_id": rec.clip_id,
            "obs_frames": obs_frames,
            "fut_frames": fut_frames,
            "obs_valid": torch.ones(self.obs_len, dtype=torch.bool),
            "fut_valid": torch.ones(self.fut_len, dtype=torch.bool),
            "obs_tracks_2d": tracks_2d[: self.obs_len],
            "fut_tracks_2d": tracks_2d[self.obs_len :],
            "obs_tracks_3d": tracks_3d[: self.obs_len],
            "fut_tracks_3d": tracks_3d[self.obs_len :],
            "visibility": visibility,
            "intrinsics": intrinsics,
            "extrinsics": None,
            "point_ids": point_ids,
            "meta": {
                "tfrecord_path": rec.tfrecord_path,
                "track_source": "deterministic_trace_from_tfrecord_identity",
                "subset": "movi_e",
                "obs_len": self.obs_len,
                "fut_len": self.fut_len,
            },
        }
        return sample
