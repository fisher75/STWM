from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json

import torch
from torch.utils.data import Dataset

from stwm.tracewm.datasets.stage1_kubric import Stage1KubricDataset
from stwm.tracewm.datasets.stage1_pointodyssey import Stage1PointOdysseyDataset
from stwm.tracewm.datasets.stage1_tapvid import Stage1TapVidDataset
from stwm.tracewm.datasets.stage1_tapvid3d import Stage1TapVid3DDataset


DEFAULT_MINISPLIT_PATH = Path("/home/chen034/workspace/data/_manifests/stage1_minisplits_20260408.json")

SAMPLE_KEYS = [
    "dataset",
    "split",
    "clip_id",
    "obs_frames",
    "fut_frames",
    "obs_valid",
    "fut_valid",
    "obs_tracks_2d",
    "fut_tracks_2d",
    "obs_tracks_3d",
    "fut_tracks_3d",
    "visibility",
    "intrinsics",
    "extrinsics",
    "point_ids",
    "meta",
]


@dataclass(frozen=True)
class _UnifiedIndex:
    dataset_name: str
    local_index: int


def load_stage1_minisplits(path: str | Path | None = None) -> Dict[str, Any]:
    resolved = Path(path) if path is not None else DEFAULT_MINISPLIT_PATH
    if not resolved.exists():
        return {"datasets": {}}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _records_for(minisplits: Dict[str, Any], dataset_name: str, split: str) -> List[Dict[str, Any]]:
    datasets = minisplits.get("datasets", {}) if isinstance(minisplits, dict) else {}
    ds = datasets.get(dataset_name, {}) if isinstance(datasets, dict) else {}
    recs = ds.get(split, []) if isinstance(ds, dict) else []
    if not isinstance(recs, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in recs:
        if isinstance(item, dict):
            out.append(item)
    return out


def ensure_sample_schema(sample: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    fixed = dict(sample)
    for key in SAMPLE_KEYS:
        if key not in fixed:
            fixed[key] = None

    fixed["dataset"] = str(fixed.get("dataset") or dataset_name)
    fixed["split"] = str(fixed.get("split") or "unknown")
    fixed["clip_id"] = str(fixed.get("clip_id") or f"{dataset_name}_unknown")
    if not isinstance(fixed.get("meta"), dict):
        fixed["meta"] = {}

    return fixed


def stage1_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    output: Dict[str, Any] = {}
    for key in SAMPLE_KEYS:
        values = [item.get(key) for item in batch]

        if all(isinstance(v, torch.Tensor) for v in values):
            shapes = [tuple(v.shape) for v in values]
            dtypes = [v.dtype for v in values]
            if len(set(shapes)) == 1 and len(set(dtypes)) == 1:
                output[key] = torch.stack(values, dim=0)
                continue
        output[key] = values

    output["batch_size"] = len(batch)
    return output


class Stage1UnifiedDataset(Dataset):
    """Unified multi-dataset wrapper under Stage 1 trace-only namespace."""

    def __init__(
        self,
        dataset_names: List[str],
        split: str,
        data_root: str | Path = "/home/chen034/workspace/data",
        minisplit_path: str | Path | None = None,
        obs_len: int = 8,
        fut_len: int = 8,
        max_samples_per_dataset: int | None = None,
    ) -> None:
        self.dataset_names = list(dataset_names)
        self.split = split
        self.data_root = Path(data_root)
        self.obs_len = int(obs_len)
        self.fut_len = int(fut_len)

        minisplits = load_stage1_minisplits(minisplit_path)

        self.datasets: Dict[str, Dataset] = {}
        self.index_map: List[_UnifiedIndex] = []

        for dataset_name in self.dataset_names:
            records = _records_for(minisplits, dataset_name, split)
            ds = self._build_dataset(dataset_name, records, max_samples_per_dataset)
            self.datasets[dataset_name] = ds

            for i in range(len(ds)):
                self.index_map.append(_UnifiedIndex(dataset_name=dataset_name, local_index=i))

        if not self.index_map:
            # Fallback keeps downstream smoke scripts alive with explicit empty signal.
            self.index_map.append(_UnifiedIndex(dataset_name=self.dataset_names[0], local_index=0))

    def _build_dataset(
        self,
        dataset_name: str,
        records: List[Dict[str, Any]],
        max_samples_per_dataset: int | None,
    ) -> Dataset:
        if dataset_name == "pointodyssey":
            ds = Stage1PointOdysseyDataset(
                data_root=self.data_root,
                split=self.split,
                minisplit_records=records,
                obs_len=self.obs_len,
                fut_len=self.fut_len,
                max_sequences=max_samples_per_dataset,
            )
        elif dataset_name == "kubric":
            ds = Stage1KubricDataset(
                data_root=self.data_root,
                split=self.split,
                minisplit_records=records,
                obs_len=self.obs_len,
                fut_len=self.fut_len,
                max_records=max_samples_per_dataset,
            )
        elif dataset_name == "tapvid":
            ds = Stage1TapVidDataset(
                split=self.split,
                minisplit_records=records,
                obs_len=self.obs_len,
                fut_len=self.fut_len,
            )
        elif dataset_name == "tapvid3d":
            ds = Stage1TapVid3DDataset(
                data_root=self.data_root,
                split=self.split,
                minisplit_records=records,
                obs_len=self.obs_len,
                fut_len=self.fut_len,
            )
        else:
            raise ValueError(f"unsupported dataset for Stage1UnifiedDataset: {dataset_name}")
        return ds

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        idx = self.index_map[index]
        ds = self.datasets[idx.dataset_name]
        sample = ds[idx.local_index]
        fixed = ensure_sample_schema(sample, idx.dataset_name)
        fixed_meta = dict(fixed.get("meta", {}))
        fixed_meta.setdefault("source_dataset", idx.dataset_name)
        fixed["meta"] = fixed_meta
        return fixed
