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
DEFAULT_V33_SIDECAR_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_semantic_identity_targets/pointodyssey"


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


def _sidecar_path(uid: str, split: str, root: str | Path | None = None) -> Path:
    sidecar_root = Path(root) if root is not None else DEFAULT_V33_SIDECAR_ROOT
    if not sidecar_root.is_absolute():
        sidecar_root = ROOT / sidecar_root
    return sidecar_root / split / f"{uid}.npz"


def _load_semantic_identity_sidecar(
    *,
    uid: str,
    split: str,
    m_points: int,
    horizon: int,
    root: str | Path | None,
    require: bool,
) -> dict[str, Any]:
    path = _sidecar_path(uid, split, root)
    if not path.exists():
        if require:
            raise FileNotFoundError(f"Missing semantic/identity sidecar: {path}")
        return {
            "semantic_identity_sidecar_available": torch.tensor(False, dtype=torch.bool),
            "semantic_identity_sidecar_path": None,
            "leakage_safe": torch.tensor(True, dtype=torch.bool),
            "input_uses_observed_only": torch.tensor(True, dtype=torch.bool),
            "future_targets_supervision_only": torch.tensor(True, dtype=torch.bool),
        }
    s = np.load(path, allow_pickle=True)
    zeros_obs = np.zeros((m_points, 8), dtype=bool)
    zeros_fut = np.zeros((m_points, horizon), dtype=bool)
    def arr(name: str, default: np.ndarray, dtype: Any | None = None) -> np.ndarray:
        val = np.asarray(s[name] if name in s.files else default)
        if dtype is not None:
            val = val.astype(dtype)
        return val
    method = arr("point_to_instance_assignment_method", np.full((m_points,), "unavailable", dtype=object))
    return {
        "semantic_identity_sidecar_available": torch.tensor(True, dtype=torch.bool),
        "semantic_identity_sidecar_path": str(path.relative_to(ROOT)),
        "point_id": torch.from_numpy(arr("point_id", np.arange(m_points), np.int64)).long(),
        "point_to_instance_id": torch.from_numpy(arr("point_to_instance_id", np.full((m_points,), -1), np.int64)).long(),
        "obs_instance_id": torch.from_numpy(arr("obs_instance_id", np.full((m_points, 8), -1), np.int64)).long(),
        "obs_instance_available_mask": torch.from_numpy(arr("obs_instance_available_mask", zeros_obs, bool)).bool(),
        "fut_instance_id": torch.from_numpy(arr("fut_instance_id", np.full((m_points, horizon), -1), np.int64)).long(),
        "fut_instance_available_mask": torch.from_numpy(arr("fut_instance_available_mask", zeros_fut, bool)).bool(),
        "fut_same_instance_as_obs": torch.from_numpy(arr("fut_same_instance_as_obs", zeros_fut, bool)).bool(),
        "fut_point_visible_target": torch.from_numpy(arr("fut_point_visible_target", arr("fut_same_point_valid", zeros_fut, bool), bool)).bool(),
        "fut_point_visible_mask": torch.from_numpy(arr("fut_point_visible_mask", np.ones((m_points, horizon), dtype=bool), bool)).bool(),
        "point_assignment_confidence": torch.from_numpy(arr("point_assignment_confidence", np.zeros((m_points,), dtype=np.float32), np.float32)).float(),
        "point_to_instance_assignment_frame": torch.from_numpy(arr("point_to_instance_assignment_frame", np.full((m_points,), -1), np.int64)).long(),
        "point_to_instance_assignment_method": [str(x) for x in method.tolist()],
        "leakage_safe": torch.tensor(bool(_scalar(s, "leakage_safe", True)), dtype=torch.bool),
        "input_uses_observed_only": torch.tensor(bool(_scalar(s, "input_uses_observed_only", True)), dtype=torch.bool),
        "future_targets_supervision_only": torch.tensor(bool(_scalar(s, "future_targets_supervision_only", True)), dtype=torch.bool),
    }


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
        enable_semantic_identity_sidecar: bool = False,
        semantic_identity_sidecar_root: str | Path | None = None,
        require_semantic_identity_sidecar: bool = False,
        use_observed_instance_context: bool = False,
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
        self.enable_semantic_identity_sidecar = bool(enable_semantic_identity_sidecar)
        self.semantic_identity_sidecar_root = semantic_identity_sidecar_root
        self.require_semantic_identity_sidecar = bool(require_semantic_identity_sidecar)
        self.use_observed_instance_context = bool(use_observed_instance_context)
        if not self.entries:
            raise RuntimeError(f"No V30 external-GT entries for manifest={manifest} H{horizon} M{m_points}")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        item = load_external_gt_item(entry["cache_path"], entry.get("v30_subset_tags", []))
        obs_points = item.obs_points[..., : self.point_dim]
        fut_points = item.fut_points[..., : self.point_dim]
        out = {
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
            "use_observed_instance_context": torch.tensor(self.use_observed_instance_context, dtype=torch.bool),
        }
        if self.enable_semantic_identity_sidecar:
            out.update(
                _load_semantic_identity_sidecar(
                    uid=item.uid,
                    split=item.split,
                    m_points=item.m_points,
                    horizon=item.horizon,
                    root=self.semantic_identity_sidecar_root,
                    require=self.require_semantic_identity_sidecar,
                )
            )
        return out


def collate_external_gt(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    tensor_keys = ["obs_points", "fut_points", "obs_vis", "fut_vis", "obs_conf", "fut_conf", "semantic_id", "has_3d", "use_observed_instance_context"]
    optional_tensor_keys = [
        "semantic_identity_sidecar_available",
        "point_id",
        "point_to_instance_id",
        "obs_instance_id",
        "obs_instance_available_mask",
        "fut_instance_id",
        "fut_instance_available_mask",
        "fut_same_instance_as_obs",
        "fut_point_visible_target",
        "fut_point_visible_mask",
        "point_assignment_confidence",
        "point_to_instance_assignment_frame",
        "leakage_safe",
        "input_uses_observed_only",
        "future_targets_supervision_only",
    ]
    tensor_keys.extend([k for k in optional_tensor_keys if k in batch[0]])
    for key in tensor_keys:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    for key in ["uid", "dataset", "split", "cache_path", "coordinate_system", "v30_subset_tags", "semantic_identity_sidecar_path", "point_to_instance_assignment_method"]:
        if key not in batch[0]:
            continue
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
