from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from stwm.tools.ostf_v17_common_20260502 import ROOT
from stwm.tools.ostf_v33_9_semantic_gate_utils_20260510 import last_observed_proto
from stwm.tools.train_ostf_v33_7_identity_belief_calibration_20260509 import BeliefDataset, collate_belief


COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
PROTO_ROOT = COMPLETE / "semantic_prototype_targets/pointodyssey/clip_vit_b32_local/K32"
V33_8_MASK_ROOT = ROOT / "manifests/ostf_v33_8_split_matched_hard_identity_semantic"
V33_11_MASK_ROOT = ROOT / "manifests/ostf_v33_11_true_semantic_hard"
BASELINE_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_11_semantic_baseline_bank/pointodyssey/clip_vit_b32_local/K32"
COPY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_11_copy_residual_semantic_targets/pointodyssey/clip_vit_b32_local/K32"
V33_9_CKPT = ROOT / "outputs/checkpoints/stwm_ostf_v33_9_fresh_expanded_h32_m128/v33_9_v33_6_global_contrastive_fresh_seed42_best.pt"

BASELINE_NAMES = [
    "last_observed_copy",
    "observed_prototype_frequency",
    "sample_level_prototype_frequency",
    "train_global_prototype_frequency",
    "nearest_observed_teacher_embedding",
]
SUBSETS = ["global", "stable", "changed", "semantic_hard"]


def np_scalar(x: Any) -> Any:
    arr = np.asarray(x)
    return arr.item() if arr.shape == () else arr.reshape(-1)[0]


def topk_from_scores(scores: np.ndarray, target: np.ndarray, mask: np.ndarray, k: int) -> float | None:
    valid = mask.astype(bool) & (target >= 0)
    if not bool(valid.any()):
        return None
    rank = np.argsort(-scores, axis=-1)[..., : min(k, scores.shape[-1])]
    hit = np.zeros_like(valid, dtype=bool)
    for j in range(rank.shape[-1]):
        hit |= rank[..., j] == target
    return float((hit & valid).sum() / max(int(valid.sum()), 1))


def onehot_distribution(ids: np.ndarray, k: int, eps: float = 1e-4) -> np.ndarray:
    out = np.full((*ids.shape, k), eps / max(k - 1, 1), dtype=np.float32)
    valid = ids >= 0
    safe = ids.clip(0, k - 1)
    np.put_along_axis(out, safe[..., None], 1.0, axis=-1)
    out[~valid] = 1.0 / k
    return out


def observed_frequency_distribution(obs: np.ndarray, obs_mask: np.ndarray, h: int, k: int) -> np.ndarray:
    out = np.zeros((obs.shape[0], h, k), dtype=np.float32)
    for m in range(obs.shape[0]):
        valid = obs_mask[m] & (obs[m] >= 0)
        if valid.any():
            counts = np.bincount(obs[m, valid], minlength=k).astype(np.float32)
            dist = counts / max(float(counts.sum()), 1.0)
        else:
            dist = np.ones(k, dtype=np.float32) / k
        out[m] = dist[None, :]
    return out


def sample_frequency_distribution(obs: np.ndarray, obs_mask: np.ndarray, h: int, k: int) -> np.ndarray:
    counts = np.ones(k, dtype=np.float32) * 1e-3
    valid = obs_mask & (obs >= 0)
    if valid.any():
        counts += np.bincount(obs[valid], minlength=k).astype(np.float32)
    dist = counts / counts.sum()
    return np.broadcast_to(dist[None, None, :], (obs.shape[0], h, k)).copy()


def train_global_distribution(k: int) -> np.ndarray:
    counts = np.ones(k, dtype=np.float32) * 1e-3
    for path in sorted((PROTO_ROOT / "train").glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
        obs_mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
        valid = obs_mask & (obs >= 0)
        if valid.any():
            counts += np.bincount(obs[valid], minlength=k).astype(np.float32)
    return counts / counts.sum()


def load_manifest_mask(root: Path, seed: int, split: str, uid: str, shape: tuple[int, int], key: str) -> np.ndarray:
    path = root / f"H32_M128_seed{seed}.json"
    if not path.exists():
        return np.zeros(shape, dtype=bool)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for entry in payload.get("splits", {}).get(split, []):
        if entry.get("sample_uid") == uid:
            z = np.load(ROOT / entry["mask_path"], allow_pickle=True)
            use_key = key if key in z.files else key.replace("_train_", "_eval_")
            return np.asarray(z[use_key]).astype(bool)
    return np.zeros(shape, dtype=bool)


def baseline_arrays_for_sample(z: np.lib.npyio.NpzFile, train_global: np.ndarray) -> dict[str, np.ndarray]:
    target = np.asarray(z["semantic_prototype_id"], dtype=np.int64)
    obs = np.asarray(z["obs_semantic_prototype_id"], dtype=np.int64)
    obs_mask = np.asarray(z["obs_semantic_prototype_available_mask"]).astype(bool)
    k = int(train_global.shape[0])
    last = last_observed_proto(obs[None], obs_mask[None])[0]
    copy = np.broadcast_to(last[:, None], target.shape).copy()
    sample_freq = sample_frequency_distribution(obs, obs_mask, target.shape[1], k)
    obs_freq = observed_frequency_distribution(obs, obs_mask, target.shape[1], k)
    return {
        "last_observed_copy": onehot_distribution(copy, k),
        "observed_prototype_frequency": obs_freq,
        "sample_level_prototype_frequency": sample_freq,
        "train_global_prototype_frequency": np.broadcast_to(train_global[None, None, :], (*target.shape, k)).copy(),
        "nearest_observed_teacher_embedding": obs_freq.copy(),
        "copy_semantic_prototype_id": copy,
        "last_observed_semantic_prototype_id": last,
    }


def load_baseline_selection(path: Path | None = None) -> dict[str, str]:
    p = path or (ROOT / "reports/stwm_ostf_v33_11_semantic_baseline_bank_20260510.json")
    if not p.exists():
        return {s: "sample_level_prototype_frequency" for s in SUBSETS}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload.get("strongest_baseline_by_subset_selected_on_val", {s: "sample_level_prototype_frequency" for s in SUBSETS})


class CopyResidualDatasetV3311(BeliefDataset):
    def __init__(self, split: str, args: Any, *, max_items: int | None = None) -> None:
        super().__init__(split, args, max_items=max_items)
        self.copy_root = Path(args.copy_residual_semantic_target_root)
        self.baseline_root = Path(args.semantic_baseline_bank_root)
        if not self.copy_root.is_absolute():
            self.copy_root = ROOT / self.copy_root
        if not self.baseline_root.is_absolute():
            self.baseline_root = ROOT / self.baseline_root
        self._copy_cache: dict[str, dict[str, torch.Tensor]] = {}

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = super().__getitem__(idx)
        uid = str(item["uid"])
        cached = self._copy_cache.get(uid)
        if cached is None:
            z = np.load(self.copy_root / self.split / f"{uid}.npz", allow_pickle=True)
            b = np.load(self.baseline_root / self.split / f"{uid}.npz", allow_pickle=True)
            cached = {}
            for key in [
                "last_observed_semantic_prototype_id",
                "copy_semantic_prototype_id",
                "semantic_stable_mask",
                "semantic_changed_mask",
                "semantic_update_target",
                "semantic_update_available_mask",
            ]:
                arr = np.asarray(z[key])
                if key.endswith("_mask") or (key.endswith("_target") and arr.dtype == bool):
                    cached[key] = torch.from_numpy(arr.astype(bool)).bool()
                else:
                    cached[key] = torch.from_numpy(arr.astype(np.int64)).long()
            for key in [
                "copy_prior_distribution",
                "observed_frequency_prior_distribution",
                "sample_level_frequency_prior_distribution",
                "train_global_prior_distribution",
            ]:
                cached[key] = torch.from_numpy(np.asarray(z[key], dtype=np.float32)).float()
            for name in BASELINE_NAMES:
                cached[f"baseline_{name}_distribution"] = torch.from_numpy(np.asarray(b[f"{name}_distribution"], dtype=np.float32)).float()
            cached["copy_residual_leakage_safe"] = torch.tensor(bool(np.asarray(z["leakage_safe"]).item()), dtype=torch.bool)
            self._copy_cache[uid] = cached
        item.update(cached)
        # The semantic hard mask is deliberately dynamic and comes from the
        # manifest set on args.hard_train_mask_manifest for this eval/training seed.
        item["semantic_hard_mask"] = item["semantic_hard_train_mask"]
        return item


def collate_copy_v3311(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = collate_belief(batch)
    keys = [
        "last_observed_semantic_prototype_id",
        "copy_semantic_prototype_id",
        "semantic_stable_mask",
        "semantic_changed_mask",
        "semantic_hard_mask",
        "semantic_update_target",
        "semantic_update_available_mask",
        "copy_prior_distribution",
        "observed_frequency_prior_distribution",
        "sample_level_frequency_prior_distribution",
        "train_global_prior_distribution",
        "copy_residual_leakage_safe",
    ] + [f"baseline_{name}_distribution" for name in BASELINE_NAMES]
    for key in keys:
        out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out


def make_loader_v3311(split: str, args: Any, *, shuffle: bool, max_items: int | None = None) -> DataLoader:
    ds = CopyResidualDatasetV3311(split, args, max_items=max_items)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=collate_copy_v3311)

