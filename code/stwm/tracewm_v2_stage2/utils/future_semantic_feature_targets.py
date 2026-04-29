from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch


@dataclass(frozen=True)
class FutureSemanticFeatureTargetCache:
    cache_path: str
    item_keys: list[str]
    features: torch.Tensor
    mask: torch.Tensor
    visibility: torch.Tensor
    reappearance: torch.Tensor
    identity: torch.Tensor
    extent_box: torch.Tensor
    feature_dim: int
    feature_backbone: str
    feature_source: str
    no_candidate_leakage: bool

    @property
    def index(self) -> dict[str, int]:
        return {str(k): i for i, k in enumerate(self.item_keys)}


def stage2_item_key(meta: dict[str, Any]) -> str:
    dataset = str(meta.get("dataset", "")).strip().upper()
    clip_id = str(meta.get("clip_id", "")).strip()
    return f"{dataset}::{clip_id}"


def load_future_semantic_feature_target_cache(path_value: str | Path) -> FutureSemanticFeatureTargetCache | None:
    target = str(path_value).strip()
    if not target:
        return None
    report_path = Path(target)
    if not report_path.exists():
        raise FileNotFoundError(f"future semantic feature target cache report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"future semantic feature target cache report must be dict: {report_path}")
    cache_path = Path(str(payload.get("cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"future semantic feature target tensor cache not found: {cache_path}")
    with np.load(cache_path, allow_pickle=True) as data:
        item_keys = [str(x) for x in data["item_keys"].tolist()]
        features = torch.from_numpy(np.asarray(data["future_semantic_feature_target"], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(data["target_mask"], dtype=bool))
        visibility = torch.from_numpy(np.asarray(data["future_visibility_target"], dtype=bool))
        reappearance = torch.from_numpy(np.asarray(data["future_reappearance_target"], dtype=bool))
        identity = torch.from_numpy(np.asarray(data["identity_target"], dtype=np.int64))
        extent_box = torch.from_numpy(np.asarray(data["extent_box_target"], dtype=np.float32))
    return FutureSemanticFeatureTargetCache(
        cache_path=str(cache_path),
        item_keys=item_keys,
        features=features,
        mask=mask,
        visibility=visibility,
        reappearance=reappearance,
        identity=identity,
        extent_box=extent_box,
        feature_dim=int(payload.get("feature_dim") or (features.shape[-1] if features.ndim == 4 else 0)),
        feature_backbone=str(payload.get("feature_backbone") or ""),
        feature_source=str(payload.get("feature_source") or ""),
        no_candidate_leakage=bool(payload.get("no_future_candidate_leakage", True)),
    )


def target_tensors_for_batch(
    cache: FutureSemanticFeatureTargetCache | None,
    batch: dict[str, Any],
    *,
    horizon: int,
    slot_count: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    if cache is None:
        return None, None, {
            "cache_available": False,
            "cache_hit_ratio": 0.0,
            "target_valid_ratio": 0.0,
            "feature_backbone": "",
            "feature_dim": 0,
            "no_candidate_leakage": True,
        }
    bsz = int(batch["obs_state"].shape[0])
    dim = int(cache.feature_dim)
    features = torch.zeros((bsz, int(horizon), int(slot_count), dim), dtype=torch.float32, device=device)
    mask = torch.zeros((bsz, int(horizon), int(slot_count)), dtype=torch.bool, device=device)
    index = cache.index
    hits = 0
    keys: list[str] = []
    for b_idx, meta in enumerate(batch.get("meta", [])):
        key = stage2_item_key(meta if isinstance(meta, dict) else {})
        keys.append(key)
        cache_idx = index.get(key)
        if cache_idx is None:
            continue
        hits += 1
        src_feat = cache.features[cache_idx]
        src_mask = cache.mask[cache_idx]
        h = min(int(horizon), int(src_feat.shape[0]))
        k = min(int(slot_count), int(src_feat.shape[1]))
        d = min(dim, int(src_feat.shape[-1]))
        features[b_idx, :h, :k, :d] = src_feat[:h, :k, :d].to(device=device, dtype=torch.float32)
        mask[b_idx, :h, :k] = src_mask[:h, :k].to(device=device, dtype=torch.bool)
    return features, mask, {
        "cache_available": True,
        "cache_path": cache.cache_path,
        "cache_hit_ratio": float(hits / max(bsz, 1)),
        "batch_item_keys": keys,
        "target_valid_ratio": float(mask.to(dtype=torch.float32).mean().detach().cpu().item()),
        "feature_backbone": cache.feature_backbone,
        "feature_source": cache.feature_source,
        "feature_dim": dim,
        "no_candidate_leakage": bool(cache.no_candidate_leakage),
    }
