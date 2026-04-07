from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _TapVid3DRecord:
    split: str
    clip_id: str
    npz_path: str
    source: str


class Stage1TapVid3DDataset(Dataset):
    """TapVid-3D Stage 1 limited-eval adapter."""

    def __init__(
        self,
        data_root: str | Path = "/home/chen034/workspace/data",
        split: str = "eval_mini",
        minisplit_records: List[Dict[str, Any]] | None = None,
        obs_len: int = 8,
        fut_len: int = 8,
    ) -> None:
        self.data_root = Path(data_root)
        self.dataset_root = self.data_root / "tapvid3d"
        self.split = split
        self.obs_len = int(obs_len)
        self.fut_len = int(fut_len)
        self.total_len = self.obs_len + self.fut_len

        if minisplit_records:
            self._records = self._build_from_minisplit(minisplit_records)
        else:
            self._records = self._discover_minival_records()

        if not self._records:
            self._records = [
                _TapVid3DRecord(
                    split=self.split,
                    clip_id="tapvid3d_synthetic_0000",
                    npz_path="",
                    source="synthetic",
                )
            ]

    def _build_from_minisplit(self, records: List[Dict[str, Any]]) -> List[_TapVid3DRecord]:
        out: List[_TapVid3DRecord] = []
        for item in records:
            clip_id = str(item.get("clip_id", "")).strip()
            npz_path = str(item.get("npz_path", "")).strip()
            source = str(item.get("source", "unknown")).strip()
            if not clip_id or not npz_path:
                continue
            out.append(_TapVid3DRecord(split=self.split, clip_id=clip_id, npz_path=npz_path, source=source))
        return out

    def _discover_minival_records(self) -> List[_TapVid3DRecord]:
        out: List[_TapVid3DRecord] = []
        base = self.dataset_root / "minival_dataset"
        for source in ["pstudio", "adt", "drivetrack"]:
            src_dir = base / source
            if not src_dir.exists():
                continue
            for p in sorted(src_dir.glob("*.npz"))[:8]:
                out.append(
                    _TapVid3DRecord(
                        split=self.split,
                        clip_id=f"{source}_{p.stem}",
                        npz_path=str(p),
                        source=source,
                    )
                )
        return out

    @staticmethod
    def _synthetic(total_len: int, num_points: int = 16) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = torch.linspace(0.0, 1.0, total_len, dtype=torch.float32).view(total_len, 1, 1)
        pid = torch.linspace(0.0, 1.0, num_points, dtype=torch.float32).view(1, num_points, 1)
        x = 0.5 * torch.cos(2.0 * torch.pi * (t + pid))
        y = 0.5 * torch.sin(2.0 * torch.pi * (t * 0.75 + pid))
        z = 0.2 + 0.6 * pid + 0.05 * t
        tracks3d = torch.cat([x, y, z], dim=-1)
        visibility = torch.ones((total_len, num_points), dtype=torch.bool)
        intrinsics = torch.tensor([1.0, 1.0, 0.5, 0.5], dtype=torch.float32)
        return tracks3d, visibility, intrinsics

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self._records[index]

        missing_fields: List[str] = []
        if rec.npz_path and Path(rec.npz_path).exists():
            with np.load(rec.npz_path, allow_pickle=False) as z:
                keys = set(z.files)
                tracks_key = "tracks_XYZ" if "tracks_XYZ" in keys else "tracks_xyz"
                if tracks_key not in keys:
                    raise KeyError(f"missing tracks key in {rec.npz_path}, keys={sorted(keys)}")

                tracks_xyz = np.asarray(z[tracks_key], dtype=np.float32)  # [T,N,3]
                visibility_np = np.asarray(z.get("visibility", np.ones(tracks_xyz.shape[:2], dtype=bool)))
                intr_np = np.asarray(z.get("fx_fy_cx_cy", np.array([1.0, 1.0, 0.5, 0.5], dtype=np.float32)))

                extrinsics = None
                if "extrinsics_w2c" in keys:
                    extr_np = np.asarray(z["extrinsics_w2c"], dtype=np.float32)
                    extrinsics = torch.from_numpy(extr_np)
                else:
                    missing_fields.append("extrinsics_w2c")

            total_t = tracks_xyz.shape[0]
            if total_t < self.total_len:
                pad_t = self.total_len - total_t
                tracks_xyz = np.concatenate([tracks_xyz, np.repeat(tracks_xyz[-1:], pad_t, axis=0)], axis=0)
                visibility_np = np.concatenate([visibility_np, np.repeat(visibility_np[-1:], pad_t, axis=0)], axis=0)
                total_t = tracks_xyz.shape[0]

            tracks3d = torch.from_numpy(tracks_xyz[: self.total_len]).to(torch.float32)
            visibility = torch.from_numpy(visibility_np[: self.total_len]).to(torch.bool)
            intrinsics = torch.from_numpy(intr_np).to(torch.float32)
        else:
            tracks3d, visibility, intrinsics = self._synthetic(self.total_len, num_points=16)
            extrinsics = None
            missing_fields.append("npz_missing")

        point_ids = torch.arange(tracks3d.shape[1], dtype=torch.long)
        obs_tracks_3d = tracks3d[: self.obs_len]
        fut_tracks_3d = tracks3d[self.obs_len :]

        obs_frames = [f"tapvid3d://{rec.source}/{rec.clip_id}/frame_{i:05d}" for i in range(self.obs_len)]
        fut_frames = [
            f"tapvid3d://{rec.source}/{rec.clip_id}/frame_{i:05d}" for i in range(self.obs_len, self.total_len)
        ]

        # We keep 2D tracks explicit as unavailable unless a projection path is requested.
        sample: Dict[str, Any] = {
            "dataset": "tapvid3d",
            "split": rec.split,
            "clip_id": rec.clip_id,
            "obs_frames": obs_frames,
            "fut_frames": fut_frames,
            "obs_valid": visibility[: self.obs_len].any(dim=1),
            "fut_valid": visibility[self.obs_len :].any(dim=1),
            "obs_tracks_2d": None,
            "fut_tracks_2d": None,
            "obs_tracks_3d": obs_tracks_3d,
            "fut_tracks_3d": fut_tracks_3d,
            "visibility": visibility,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "point_ids": point_ids,
            "meta": {
                "source": rec.source,
                "npz_path": rec.npz_path,
                "missing_fields": missing_fields,
                "track_source": "tapvid3d_npz",
            },
        }
        return sample
