from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch

from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key


@dataclass(frozen=True)
class FutureSemanticPrototypeTargetCache:
    cache_path: str
    item_keys: list[str]
    proto_target: torch.Tensor
    proto_distribution: torch.Tensor
    mask: torch.Tensor
    visibility: torch.Tensor
    reappearance: torch.Tensor
    identity: torch.Tensor
    extent_box: torch.Tensor
    prototypes: torch.Tensor
    prototype_count: int
    feature_backbone: str
    no_candidate_leakage: bool
    prototype_entropy: float

    @property
    def index(self) -> dict[str, int]:
        return {str(k): i for i, k in enumerate(self.item_keys)}


def load_future_semantic_prototype_target_cache(path_value: str | Path) -> FutureSemanticPrototypeTargetCache | None:
    target = str(path_value).strip()
    if not target:
        return None
    report_path = Path(target)
    if not report_path.exists():
        raise FileNotFoundError(f"future semantic prototype target report not found: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get("target_cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"future semantic prototype target npz not found: {cache_path}")
    with np.load(cache_path, allow_pickle=True) as data:
        item_keys = [str(x) for x in data["item_keys"].tolist()]
        proto_target = torch.from_numpy(np.asarray(data["future_semantic_proto_target"], dtype=np.int64))
        proto_distribution = torch.from_numpy(np.asarray(data["future_semantic_proto_distribution"], dtype=np.float32))
        mask = torch.from_numpy(np.asarray(data["target_mask"], dtype=bool))
        visibility = torch.from_numpy(np.asarray(data["future_visibility_target"], dtype=bool))
        reappearance = torch.from_numpy(np.asarray(data["future_reappearance_target"], dtype=bool))
        identity = torch.from_numpy(np.asarray(data["identity_target"], dtype=np.int64))
        extent_box = torch.from_numpy(np.asarray(data["future_extent_box_target"], dtype=np.float32))
        prototypes = torch.from_numpy(np.asarray(data["prototypes"], dtype=np.float32))
    return FutureSemanticPrototypeTargetCache(
        cache_path=str(cache_path),
        item_keys=item_keys,
        proto_target=proto_target,
        proto_distribution=proto_distribution,
        mask=mask,
        visibility=visibility,
        reappearance=reappearance,
        identity=identity,
        extent_box=extent_box,
        prototypes=prototypes,
        prototype_count=int(payload.get("prototype_count") or (proto_distribution.shape[-1] if proto_distribution.ndim == 4 else 0)),
        feature_backbone=str(payload.get("feature_backbone") or ""),
        no_candidate_leakage=bool(payload.get("no_future_candidate_leakage", True)),
        prototype_entropy=float(payload.get("prototype_entropy") or 0.0),
    )


def prototype_tensors_for_batch(
    cache: FutureSemanticPrototypeTargetCache | None,
    batch: dict[str, Any],
    *,
    horizon: int,
    slot_count: int,
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, dict[str, Any]]:
    if cache is None:
        return None, None, None, {
            "cache_available": False,
            "cache_hit_ratio": 0.0,
            "target_valid_ratio": 0.0,
            "prototype_count": 0,
            "feature_backbone": "",
            "no_candidate_leakage": True,
        }
    bsz = int(batch["obs_state"].shape[0])
    c = int(cache.prototype_count)
    target = torch.full((bsz, int(horizon), int(slot_count)), -1, dtype=torch.long, device=device)
    dist = torch.zeros((bsz, int(horizon), int(slot_count), c), dtype=torch.float32, device=device)
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
        src_target = cache.proto_target[cache_idx]
        src_dist = cache.proto_distribution[cache_idx]
        src_mask = cache.mask[cache_idx]
        h = min(int(horizon), int(src_target.shape[0]))
        k = min(int(slot_count), int(src_target.shape[1]))
        cc = min(c, int(src_dist.shape[-1]))
        target[b_idx, :h, :k] = src_target[:h, :k].to(device=device, dtype=torch.long)
        dist[b_idx, :h, :k, :cc] = src_dist[:h, :k, :cc].to(device=device, dtype=torch.float32)
        mask[b_idx, :h, :k] = src_mask[:h, :k].to(device=device, dtype=torch.bool)
    return target, dist, mask, {
        "cache_available": True,
        "cache_path": cache.cache_path,
        "cache_hit_ratio": float(hits / max(bsz, 1)),
        "batch_item_keys": keys,
        "target_valid_ratio": float(mask.to(dtype=torch.float32).mean().detach().cpu().item()),
        "prototype_count": c,
        "prototype_entropy": float(cache.prototype_entropy),
        "feature_backbone": cache.feature_backbone,
        "no_candidate_leakage": bool(cache.no_candidate_leakage),
    }
