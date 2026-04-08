from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from stwm.tracewm_v2.constants import DATASET_ID_MAP, STATE_DIM, TRACE_CONTRACT_PATH
from stwm.tracewm_v2.trace_cache_contract import load_contract


@dataclass(frozen=True)
class _ClipIndex:
    dataset: str
    split: str
    clip_id: str
    cache_path: str


def _canonical_split(split: str) -> str:
    s = str(split).strip().lower()
    if s.startswith("train"):
        return "train"
    if s.startswith("val") or s.startswith("eval"):
        return "val"
    if s.startswith("test"):
        return "test"
    return s


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _select_tokens(valid: np.ndarray, max_tokens: int) -> np.ndarray:
    score = valid.sum(axis=0)
    idx = np.argsort(-score)
    if max_tokens > 0:
        idx = idx[:max_tokens]
    return idx


def _build_state_features(tracks_2d: np.ndarray, tracks_3d: np.ndarray, valid: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    t_len, k_len, _ = tracks_2d.shape
    state = np.zeros((t_len, k_len, STATE_DIM), dtype=np.float32)

    coord = tracks_2d.astype(np.float32, copy=False)
    z = tracks_3d[..., 2].astype(np.float32, copy=False)
    vis = visibility.astype(np.float32, copy=False)

    vel = np.zeros_like(coord, dtype=np.float32)
    vel[1:] = coord[1:] - coord[:-1]

    residual = coord - coord[0:1]

    state[..., 0:2] = coord
    state[..., 2] = z
    state[..., 3] = vis
    state[..., 4:6] = vel
    state[..., 6:8] = residual

    mask = valid.astype(bool)
    state[..., 0:3] = np.where(mask[..., None], state[..., 0:3], 0.0)
    state[..., 4:8] = np.where(mask[..., None], state[..., 4:8], 0.0)
    state[..., 3] = np.where(mask, state[..., 3], 0.0)
    return state


class Stage1V2UnifiedDataset(Dataset):
    def __init__(
        self,
        dataset_names: List[str],
        split: str,
        contract_path: str | Path | None = None,
        obs_len: int = 8,
        fut_len: int = 8,
        max_tokens: int = 64,
        max_samples_per_dataset: int = 0,
    ) -> None:
        self.dataset_names = [str(x) for x in dataset_names]
        self.split = _canonical_split(split)
        self.obs_len = int(obs_len)
        self.fut_len = int(fut_len)
        self.total_len = self.obs_len + self.fut_len
        self.max_tokens = int(max_tokens)

        resolved_contract = Path(contract_path) if contract_path is not None else TRACE_CONTRACT_PATH
        self.contract = load_contract(resolved_contract)

        self.index_map: List[_ClipIndex] = []
        for dataset_name in self.dataset_names:
            ds_entry = None
            for item in self.contract.get("datasets", []):
                if isinstance(item, dict) and str(item.get("dataset_name", "")) == dataset_name:
                    ds_entry = item
                    break
            if ds_entry is None:
                raise KeyError(f"dataset missing in trace cache contract: {dataset_name}")
            if not bool(ds_entry.get("enabled", False)):
                continue

            index_path = Path(str(ds_entry.get("index_path", "")))
            index_payload = _load_json(index_path)
            entries = [e for e in index_payload.get("entries", []) if isinstance(e, dict)]
            filtered = [e for e in entries if _canonical_split(str(e.get("split", ""))) == self.split]

            if int(max_samples_per_dataset) > 0:
                filtered = filtered[: int(max_samples_per_dataset)]

            for rec in filtered:
                self.index_map.append(
                    _ClipIndex(
                        dataset=dataset_name,
                        split=self.split,
                        clip_id=str(rec.get("clip_id", "")),
                        cache_path=str(rec.get("cache_path", "")),
                    )
                )

        if not self.index_map:
            raise RuntimeError(f"no cache samples found for split={self.split} datasets={self.dataset_names}")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rec = self.index_map[index]
        cache_path = Path(rec.cache_path)
        payload = np.load(cache_path, allow_pickle=True)

        tracks_2d = np.asarray(payload["tracks_2d"], dtype=np.float32)
        tracks_3d = np.asarray(payload["tracks_3d"], dtype=np.float32)
        valid = np.asarray(payload["valid"], dtype=bool)
        visibility = np.asarray(payload["visibility"], dtype=bool)
        point_ids = np.asarray(payload["point_ids"], dtype=np.int64)

        if tracks_2d.ndim != 3 or tracks_2d.shape[-1] != 2:
            raise ValueError(f"bad tracks_2d shape in {cache_path}: {tracks_2d.shape}")

        t_len = min(tracks_2d.shape[0], self.total_len)
        tracks_2d = tracks_2d[:t_len]
        tracks_3d = tracks_3d[:t_len]
        valid = valid[:t_len]
        visibility = visibility[:t_len]

        if t_len < self.total_len:
            raise ValueError(f"clip too short in {cache_path}: got {t_len}, need {self.total_len}")

        select = _select_tokens(valid=valid, max_tokens=self.max_tokens)
        tracks_2d = tracks_2d[:, select]
        tracks_3d = tracks_3d[:, select]
        valid = valid[:, select]
        visibility = visibility[:, select]
        point_ids = point_ids[select]

        state = _build_state_features(
            tracks_2d=tracks_2d,
            tracks_3d=tracks_3d,
            valid=valid,
            visibility=visibility,
        )

        obs_state = torch.from_numpy(state[: self.obs_len]).to(torch.float32)
        fut_state = torch.from_numpy(state[self.obs_len : self.total_len]).to(torch.float32)
        obs_valid = torch.from_numpy(valid[: self.obs_len]).to(torch.bool)
        fut_valid = torch.from_numpy(valid[self.obs_len : self.total_len]).to(torch.bool)
        token_mask = torch.ones((point_ids.shape[0],), dtype=torch.bool)

        sample: Dict[str, Any] = {
            "dataset": rec.dataset,
            "split": rec.split,
            "clip_id": rec.clip_id,
            "obs_state": obs_state,
            "fut_state": fut_state,
            "obs_valid": obs_valid,
            "fut_valid": fut_valid,
            "token_mask": token_mask,
            "point_ids": torch.from_numpy(point_ids).to(torch.long),
            "meta": {
                "cache_path": str(cache_path),
                "source_ref": str(payload["source_ref"].item()),
                "track_source": str(payload["track_source"].item()),
            },
        }
        return sample


def stage1_v2_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    bsz = len(batch)
    obs_len = int(batch[0]["obs_state"].shape[0])
    fut_len = int(batch[0]["fut_state"].shape[0])
    state_dim = int(batch[0]["obs_state"].shape[-1])
    max_k = max(int(item["obs_state"].shape[1]) for item in batch)

    obs_state = torch.zeros((bsz, obs_len, max_k, state_dim), dtype=torch.float32)
    fut_state = torch.zeros((bsz, fut_len, max_k, state_dim), dtype=torch.float32)
    obs_valid = torch.zeros((bsz, obs_len, max_k), dtype=torch.bool)
    fut_valid = torch.zeros((bsz, fut_len, max_k), dtype=torch.bool)
    token_mask = torch.zeros((bsz, max_k), dtype=torch.bool)
    point_ids = torch.full((bsz, max_k), fill_value=-1, dtype=torch.long)

    datasets: List[str] = []
    splits: List[str] = []
    clip_ids: List[str] = []
    metas: List[Dict[str, Any]] = []

    for i, item in enumerate(batch):
        k = int(item["obs_state"].shape[1])
        obs_state[i, :, :k] = item["obs_state"]
        fut_state[i, :, :k] = item["fut_state"]
        obs_valid[i, :, :k] = item["obs_valid"]
        fut_valid[i, :, :k] = item["fut_valid"]
        token_mask[i, :k] = item["token_mask"]
        point_ids[i, :k] = item["point_ids"]

        datasets.append(str(item["dataset"]))
        splits.append(str(item["split"]))
        clip_ids.append(str(item["clip_id"]))
        metas.append(dict(item.get("meta", {})))

    dataset_ids = torch.tensor([DATASET_ID_MAP.get(x, 7) for x in datasets], dtype=torch.long)

    return {
        "batch_size": bsz,
        "dataset": datasets,
        "dataset_ids": dataset_ids,
        "split": splits,
        "clip_id": clip_ids,
        "obs_state": obs_state,
        "fut_state": fut_state,
        "obs_valid": obs_valid,
        "fut_valid": fut_valid,
        "token_mask": token_mask,
        "point_ids": point_ids,
        "meta": metas,
    }
