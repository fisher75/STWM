from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from stwm.tracewm_v2.constants import STATE_DIM
from stwm.tracewm_v2.datasets.stage1_v2_unified import Stage1V2UnifiedDataset, stage1_v2_collate_fn


SEMANTIC_FEATURE_DIM = 10


@dataclass
class Stage2SemanticDatasetConfig:
    dataset_names: List[str]
    split: str
    contract_path: str
    obs_len: int
    fut_len: int
    max_tokens: int
    max_samples_per_dataset: int
    semantic_patch_radius: int = 12
    semantic_frame_index: int = 0


def _load_rgb(path: str | Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def _safe_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _mask_candidates(frame_path: Path) -> List[Path]:
    stem = frame_path.stem
    parent = frame_path.parent
    candidates = [
        parent / f"mask_{stem.split('_')[-1]}.png",
        parent / f"mask_{stem.split('_')[-1]}.jpg",
        parent / f"{stem.replace('rgb_', 'mask_')}.png",
        parent / f"{stem.replace('rgba_', 'mask_')}.png",
        parent.parent / "masks" / f"{stem}.png",
        parent.parent / "masks" / f"mask_{stem}.png",
    ]
    out: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _resolve_optional_mask(frame_path: str, source_ref: str) -> Tuple[str, bool]:
    fp = Path(frame_path)
    for cand in _mask_candidates(fp):
        if cand.exists():
            return str(cand), True

    src = Path(source_ref)
    if src.suffix.lower() == ".json" and src.exists():
        payload = _safe_json(src)
        for key in ["mask_paths", "segmentation_paths", "instance_mask_paths"]:
            val = payload.get(key)
            if isinstance(val, list) and val:
                p = Path(str(val[0]))
                if p.exists():
                    return str(p), True
    return "", False


def _safe_crop_bounds(cx: float, cy: float, radius: int, width: int, height: int) -> Tuple[int, int, int, int]:
    r = max(int(radius), 2)
    x0 = max(int(round(cx)) - r, 0)
    y0 = max(int(round(cy)) - r, 0)
    x1 = min(int(round(cx)) + r + 1, width)
    y1 = min(int(round(cy)) + r + 1, height)
    if x1 <= x0:
        x1 = min(x0 + 1, width)
    if y1 <= y0:
        y1 = min(y0 + 1, height)
    return x0, y0, x1, y1


def _extract_region_features(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    tracks_2d: np.ndarray,
    valid: np.ndarray,
    token_index: int,
    obs_len: int,
    radius: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    h, w, _ = rgb.shape

    v = valid[:obs_len, token_index]
    xy = tracks_2d[:obs_len, token_index]

    use = np.where(v.astype(bool))[0]
    if use.size == 0:
        use = np.array([0], dtype=np.int64)

    coords = xy[use]
    coords = np.where(np.isfinite(coords), coords, 0.0)
    cx = float(np.mean(coords[:, 0]))
    cy = float(np.mean(coords[:, 1]))

    x0, y0, x1, y1 = _safe_crop_bounds(cx=cx, cy=cy, radius=radius, width=w, height=h)
    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        patch = np.zeros((1, 1, 3), dtype=np.float32)

    masked_used = False
    if mask is not None and mask.shape[0] == h and mask.shape[1] == w:
        patch_mask = mask[y0:y1, x0:x1] > 0
        if np.any(patch_mask):
            pix = patch[patch_mask]
            masked_used = True
        else:
            pix = patch.reshape(-1, 3)
    else:
        pix = patch.reshape(-1, 3)

    mean_rgb = np.mean(pix, axis=0).astype(np.float32)
    std_rgb = np.std(pix, axis=0).astype(np.float32)

    vis_ratio = float(np.mean(v.astype(np.float32)))
    first_xy = coords[0]
    last_xy = coords[-1]
    motion = float(np.linalg.norm(last_xy - first_xy) / max(np.sqrt(h * h + w * w), 1.0))
    area_ratio = float(((x1 - x0) * (y1 - y0)) / max(float(h * w), 1.0))

    feat = np.array(
        [
            float(mean_rgb[0]),
            float(mean_rgb[1]),
            float(mean_rgb[2]),
            float(std_rgb[0]),
            float(std_rgb[1]),
            float(std_rgb[2]),
            float(vis_ratio),
            float(motion),
            float(1.0 if masked_used else 0.0),
            float(area_ratio),
        ],
        dtype=np.float32,
    )

    box = np.array([float(x0), float(y0), float(x1), float(y1)], dtype=np.float32)
    return feat, box, bool(masked_used)


class Stage2SemanticDataset(Dataset):
    """Stage2 bootstrap dataset built on Stage1-v2 cache with visual semantic regions."""

    def __init__(self, cfg: Stage2SemanticDatasetConfig) -> None:
        self.cfg = cfg
        self.base = Stage1V2UnifiedDataset(
            dataset_names=[str(x) for x in cfg.dataset_names],
            split=str(cfg.split),
            contract_path=str(cfg.contract_path),
            obs_len=int(cfg.obs_len),
            fut_len=int(cfg.fut_len),
            max_tokens=int(cfg.max_tokens),
            max_samples_per_dataset=int(cfg.max_samples_per_dataset),
        )

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.base[index]
        cache_path = Path(str(sample["meta"]["cache_path"]))
        cache = np.load(cache_path, allow_pickle=True)

        tracks_2d = np.asarray(cache["tracks_2d"], dtype=np.float32)
        valid = np.asarray(cache["valid"], dtype=bool)
        point_ids_full = np.asarray(cache["point_ids"], dtype=np.int64)

        frame_paths = [str(x) for x in np.asarray(cache["frame_paths"]).tolist()]
        if not frame_paths:
            raise RuntimeError(f"frame_paths missing in cache: {cache_path}")

        frame_index = max(0, min(int(self.cfg.semantic_frame_index), len(frame_paths) - 1))
        frame_path = frame_paths[frame_index]
        rgb = _load_rgb(frame_path)

        source_ref = str(sample["meta"].get("source_ref", ""))
        mask_path, mask_available = _resolve_optional_mask(frame_path=frame_path, source_ref=source_ref)
        mask_arr = None
        if mask_available:
            try:
                m = Image.open(mask_path)
                mask_arr = np.asarray(m)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr[..., 0]
            except Exception:
                mask_arr = None
                mask_available = False

        selected_ids = sample["point_ids"].detach().cpu().numpy().astype(np.int64)
        idx_map = {int(pid): int(i) for i, pid in enumerate(point_ids_full.tolist())}

        semantic_features = np.zeros((selected_ids.shape[0], SEMANTIC_FEATURE_DIM), dtype=np.float32)
        semantic_boxes = np.zeros((selected_ids.shape[0], 4), dtype=np.float32)
        semantic_mask = np.zeros((selected_ids.shape[0],), dtype=bool)
        masked_count = 0

        for i, pid in enumerate(selected_ids.tolist()):
            token_idx = idx_map.get(int(pid), -1)
            if token_idx < 0:
                continue
            feat, box, used_mask = _extract_region_features(
                rgb=rgb,
                mask=mask_arr,
                tracks_2d=tracks_2d,
                valid=valid,
                token_index=token_idx,
                obs_len=int(self.cfg.obs_len),
                radius=int(self.cfg.semantic_patch_radius),
            )
            semantic_features[i] = feat
            semantic_boxes[i] = box
            semantic_mask[i] = True
            if used_mask:
                masked_count += 1

        out = dict(sample)
        out["semantic_features"] = torch.from_numpy(semantic_features).to(torch.float32)
        out["semantic_boxes"] = torch.from_numpy(semantic_boxes).to(torch.float32)
        out["semantic_mask"] = torch.from_numpy(semantic_mask).to(torch.bool)
        out["semantic_frame_path"] = frame_path
        out["semantic_mask_path"] = mask_path
        out["semantic_source_mode"] = "object_region_or_mask_crop_visual_state"
        out["semantic_source_summary"] = {
            "mask_crop_used_tokens": int(masked_count),
            "region_crop_used_tokens": int(int(semantic_mask.sum()) - masked_count),
            "mask_available": bool(mask_available),
            "semantic_feature_dim": int(SEMANTIC_FEATURE_DIM),
        }
        return out


def stage2_semantic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    base = stage1_v2_collate_fn(batch)
    bsz = int(base["batch_size"])
    max_k = int(base["token_mask"].shape[1])
    feat_dim = int(batch[0]["semantic_features"].shape[1])

    semantic_features = torch.zeros((bsz, max_k, feat_dim), dtype=torch.float32)
    semantic_boxes = torch.zeros((bsz, max_k, 4), dtype=torch.float32)
    semantic_mask = torch.zeros((bsz, max_k), dtype=torch.bool)

    frame_paths: List[str] = []
    mask_paths: List[str] = []
    source_summaries: List[Dict[str, Any]] = []

    for i, item in enumerate(batch):
        k = int(item["semantic_features"].shape[0])
        semantic_features[i, :k] = item["semantic_features"]
        semantic_boxes[i, :k] = item["semantic_boxes"]
        semantic_mask[i, :k] = item["semantic_mask"]
        frame_paths.append(str(item.get("semantic_frame_path", "")))
        mask_paths.append(str(item.get("semantic_mask_path", "")))
        source_summaries.append(dict(item.get("semantic_source_summary", {})))

    base["semantic_features"] = semantic_features
    base["semantic_boxes"] = semantic_boxes
    base["semantic_mask"] = semantic_mask
    base["semantic_frame_paths"] = frame_paths
    base["semantic_mask_paths"] = mask_paths
    base["semantic_source_mode"] = "object_region_or_mask_crop_visual_state"
    base["semantic_source_summaries"] = source_summaries
    return base
