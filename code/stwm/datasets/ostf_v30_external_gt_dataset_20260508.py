from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path("/raid/chen034/workspace/stwm")
MANIFEST_DIR = ROOT / "manifests/ostf_v30_external_gt"


@dataclass
class OSTFExternalGTItem:
    uid: str
    dataset: str
    split: str
    cache_path: str
    horizon: int
    m_points: int
    coordinate_system: str
    obs_points: np.ndarray
    fut_points: np.ndarray
    obs_vis: np.ndarray
    fut_vis: np.ndarray
    obs_conf: np.ndarray
    fut_conf: np.ndarray
    semantic_id: int
    has_3d: bool
    v30_subset_tags: list[str]


def _load_manifest(split: str) -> list[dict[str, Any]]:
    path = MANIFEST_DIR / f"{split}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("entries", []))


def _scalar(z: Any, key: str, default: Any) -> Any:
    if key not in z:
        return default
    arr = np.asarray(z[key])
    return arr.item() if arr.shape == () else arr


def load_external_gt_item(cache_path: str | Path, tags: list[str] | None = None) -> OSTFExternalGTItem:
    path = Path(cache_path)
    if not path.is_absolute():
        path = ROOT / path
    z = np.load(path, allow_pickle=True)
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    fut_points = np.asarray(z["fut_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"]).astype(bool)
    fut_vis = np.asarray(z["fut_vis"]).astype(bool)
    obs_conf = np.asarray(z["obs_conf"], dtype=np.float32) if "obs_conf" in z else obs_vis.astype(np.float32)
    fut_conf = np.asarray(z["fut_conf"], dtype=np.float32) if "fut_conf" in z else fut_vis.astype(np.float32)
    return OSTFExternalGTItem(
        uid=str(_scalar(z, "video_uid", path.stem)),
        dataset=str(_scalar(z, "dataset", "unknown")),
        split=str(_scalar(z, "split", "unknown")),
        cache_path=str(path.relative_to(ROOT)),
        horizon=int(fut_points.shape[1]),
        m_points=int(obs_points.shape[0]),
        coordinate_system=str(_scalar(z, "coordinate_system", "xy")),
        obs_points=np.nan_to_num(obs_points, nan=0.0, posinf=1e6, neginf=-1e6),
        fut_points=np.nan_to_num(fut_points, nan=0.0, posinf=1e6, neginf=-1e6),
        obs_vis=obs_vis,
        fut_vis=fut_vis,
        obs_conf=np.nan_to_num(obs_conf, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0),
        fut_conf=np.nan_to_num(fut_conf, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0),
        semantic_id=int(_scalar(z, "semantic_id", -1)),
        has_3d=("obs_points_3d" in z and "fut_points_3d" in z),
        v30_subset_tags=list(tags or []),
    )


class OSTFExternalGTDataset(Dataset):
    def __init__(
        self,
        split: str,
        *,
        horizon: int,
        m_points: int,
        manifest_name: str | None = None,
        max_items: int | None = None,
        point_dim: int = 2,
    ) -> None:
        super().__init__()
        manifest = manifest_name or split
        entries = _load_manifest(manifest)
        self.entries = [
            e
            for e in entries
            if int(e.get("H", -1)) == int(horizon) and int(e.get("M", -1)) == int(m_points)
        ]
        if max_items is not None:
            self.entries = self.entries[: int(max_items)]
        self.split = split
        self.horizon = int(horizon)
        self.m_points = int(m_points)
        self.point_dim = int(point_dim)
        if not self.entries:
            raise RuntimeError(f"No V30 external-GT entries for manifest={manifest} H{horizon} M{m_points}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        item = load_external_gt_item(entry["cache_path"], entry.get("v30_subset_tags", []))
        obs_points = item.obs_points[..., : self.point_dim]
        fut_points = item.fut_points[..., : self.point_dim]
        return {
            "uid": item.uid,
            "dataset": item.dataset,
            "split": item.split,
            "cache_path": item.cache_path,
            "coordinate_system": item.coordinate_system,
            "obs_points": torch.from_numpy(obs_points).float(),
            "fut_points": torch.from_numpy(fut_points).float(),
            "obs_vis": torch.from_numpy(item.obs_vis).bool(),
            "fut_vis": torch.from_numpy(item.fut_vis).bool(),
            "obs_conf": torch.from_numpy(item.obs_conf).float(),
            "fut_conf": torch.from_numpy(item.fut_conf).float(),
            "semantic_id": torch.tensor(item.semantic_id, dtype=torch.long),
            "has_3d": torch.tensor(item.has_3d, dtype=torch.bool),
            "v30_subset_tags": item.v30_subset_tags,
        }


def collate_external_gt(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = ["obs_points", "fut_points", "obs_vis", "fut_vis", "obs_conf", "fut_conf", "semantic_id", "has_3d"]
    for key in tensor_keys:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    for key in ["uid", "dataset", "split", "cache_path", "coordinate_system", "v30_subset_tags"]:
        out[key] = [b[key] for b in batch]
    return out


def dataset_summary(horizon: int, m_points: int) -> dict[str, Any]:
    summary = {}
    for split in ("train", "val", "test"):
        ds = OSTFExternalGTDataset(split, horizon=horizon, m_points=m_points)
        datasets: dict[str, int] = {}
        for e in ds.entries:
            datasets[str(e.get("dataset", "unknown"))] = datasets.get(str(e.get("dataset", "unknown")), 0) + 1
        summary[split] = {"item_count": len(ds), "by_dataset": datasets}
    return summary
