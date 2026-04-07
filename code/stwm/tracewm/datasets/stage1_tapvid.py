from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _TapVidRecord:
    split: str
    clip_id: str
    variant: str
    cache_npz: str
    num_frames: int
    num_points: int


class Stage1TapVidDataset(Dataset):
    """TAP-Vid Stage 1 eval adapter.

    Reads compact per-clip cache produced by build_stage1_minisplits.py and emits
    unified 2D tracking samples for eval smoke.
    """

    def __init__(
        self,
        split: str = "eval_mini",
        minisplit_records: List[Dict[str, Any]] | None = None,
        obs_len: int = 8,
        fut_len: int = 8,
    ) -> None:
        self.split = split
        self.obs_len = int(obs_len)
        self.fut_len = int(fut_len)
        self.total_len = self.obs_len + self.fut_len

        self._records = self._build_from_minisplit(minisplit_records or [])
        if not self._records:
            self._records = [
                _TapVidRecord(
                    split=self.split,
                    clip_id="tapvid_synthetic_0000",
                    variant="synthetic",
                    cache_npz="",
                    num_frames=self.total_len,
                    num_points=16,
                )
            ]

    def _build_from_minisplit(self, records: List[Dict[str, Any]]) -> List[_TapVidRecord]:
        out: List[_TapVidRecord] = []
        for item in records:
            clip_id = str(item.get("clip_id", "")).strip()
            cache_npz = str(item.get("cache_npz", "")).strip()
            variant = str(item.get("variant", "unknown")).strip()
            num_frames = int(item.get("num_frames", 0) or 0)
            num_points = int(item.get("num_points", 0) or 0)
            if not clip_id:
                continue
            out.append(
                _TapVidRecord(
                    split=self.split,
                    clip_id=clip_id,
                    variant=variant,
                    cache_npz=cache_npz,
                    num_frames=max(num_frames, self.total_len),
                    num_points=max(num_points, 1),
                )
            )
        return out

    @staticmethod
    def _synthetic_tracks(total_len: int, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.linspace(0.0, 1.0, total_len, dtype=torch.float32).view(total_len, 1, 1)
        pid = torch.linspace(0.0, 1.0, num_points, dtype=torch.float32).view(1, num_points, 1)
        x = (0.15 + 0.7 * pid + 0.04 * t) % 1.0
        y = (0.2 + 0.6 * (1.0 - pid) + 0.03 * t) % 1.0
        tracks = torch.cat([x, y], dim=-1)
        vis = torch.ones((total_len, num_points), dtype=torch.bool)
        return tracks, vis

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]

        tracks_2d: torch.Tensor
        visibility: torch.Tensor
        frame_count = max(rec.num_frames, self.total_len)

        cache_path = Path(rec.cache_npz)
        if rec.cache_npz and cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as z:
                points = np.asarray(z["points"], dtype=np.float32)  # [N,T,2]
                occluded = np.asarray(z["occluded"], dtype=np.bool_)  # [N,T]
            # Convert to [T,N,2], [T,N]
            points_tn2 = np.transpose(points, (1, 0, 2))
            occ_tn = np.transpose(occluded, (1, 0))

            if points_tn2.shape[0] < self.total_len:
                pad_t = self.total_len - points_tn2.shape[0]
                points_tn2 = np.concatenate([points_tn2, np.repeat(points_tn2[-1:], pad_t, axis=0)], axis=0)
                occ_tn = np.concatenate([occ_tn, np.repeat(occ_tn[-1:], pad_t, axis=0)], axis=0)

            tracks_2d = torch.from_numpy(points_tn2[: self.total_len]).to(torch.float32)
            visibility = torch.from_numpy(~occ_tn[: self.total_len]).to(torch.bool)
            frame_count = int(points_tn2.shape[0])
        else:
            tracks_2d, visibility = self._synthetic_tracks(self.total_len, rec.num_points)

        point_ids = torch.arange(tracks_2d.shape[1], dtype=torch.long)

        obs_frames = [f"tapvid://{rec.variant}/{rec.clip_id}/frame_{i:05d}" for i in range(self.obs_len)]
        fut_frames = [
            f"tapvid://{rec.variant}/{rec.clip_id}/frame_{i:05d}" for i in range(self.obs_len, self.total_len)
        ]

        sample: Dict[str, Any] = {
            "dataset": "tapvid",
            "split": rec.split,
            "clip_id": rec.clip_id,
            "obs_frames": obs_frames,
            "fut_frames": fut_frames,
            "obs_valid": torch.ones(self.obs_len, dtype=torch.bool),
            "fut_valid": torch.ones(self.fut_len, dtype=torch.bool),
            "obs_tracks_2d": tracks_2d[: self.obs_len],
            "fut_tracks_2d": tracks_2d[self.obs_len :],
            "obs_tracks_3d": None,
            "fut_tracks_3d": None,
            "visibility": visibility,
            "intrinsics": None,
            "extrinsics": None,
            "point_ids": point_ids,
            "meta": {
                "variant": rec.variant,
                "cache_npz": rec.cache_npz,
                "frame_count": frame_count,
                "point_count": int(tracks_2d.shape[1]),
                "track_source": "tapvid_points_cache" if rec.cache_npz else "synthetic_fallback",
            },
        }
        return sample
