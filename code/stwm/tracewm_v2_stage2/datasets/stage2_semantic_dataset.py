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


SEMANTIC_FEATURE_DIM = 10
DEFAULT_SEMANTIC_CROP_SIZE = 64
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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
    semantic_crop_size: int = DEFAULT_SEMANTIC_CROP_SIZE
    semantic_source_mainline: str = "crop_visual_encoder"
    semantic_frame_index: int = 0
    semantic_temporal_window: int = 1
    predecode_cache_path: str = ""


def _safe_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _norm_name(name: str) -> str:
    n = str(name).strip().upper()
    if n == "VIPSEG":
        return "VIPSEG"
    return n


def _list_frames(frame_dir: Path) -> List[str]:
    if not frame_dir.exists():
        return []
    frames = [
        p
        for p in frame_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")
    ]
    return [str(p) for p in sorted(frames)]


def _read_split_ids(path: str | Path) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    ids: List[str] = []
    for raw in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("._"):
            continue
        ids.append(line)
    return ids


def _align_masks_by_name(frame_paths: List[str], mask_dir: Path) -> List[str]:
    if not mask_dir.exists():
        return ["" for _ in frame_paths]

    masks: List[str] = []
    for frame_path in frame_paths:
        fp = Path(frame_path)
        cand = mask_dir / f"{fp.stem}.png"
        if cand.exists():
            masks.append(str(cand))
        else:
            masks.append("")
    return masks


def _temporal_indices(frame_count: int, total_steps: int) -> List[int]:
    if frame_count <= 0:
        return [0 for _ in range(max(total_steps, 1))]
    idx = np.linspace(0, frame_count - 1, num=max(total_steps, 1), dtype=np.int64)
    return [int(x) for x in idx.tolist()]


def _cache_key(dataset_name: str, split: str, clip_id: str) -> str:
    safe_clip = str(clip_id).replace("/", "__")
    return f"{str(dataset_name)}::{str(split)}::{safe_clip}"


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


def _load_mask(path: str) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        arr = np.asarray(Image.open(p))
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _dominant_positive_instance_id(id_map: np.ndarray | None) -> int:
    if id_map is None:
        return 0
    flat = np.asarray(id_map, dtype=np.int64)
    flat = flat[flat > 0]
    if flat.size <= 0:
        return 0
    values, counts = np.unique(flat, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _instance_mask_from_id_map(id_map: np.ndarray | None, instance_id: int) -> np.ndarray | None:
    if id_map is None or int(instance_id) <= 0:
        return None
    mask = (np.asarray(id_map, dtype=np.int64) == int(instance_id))
    if not np.any(mask):
        return None
    return mask.astype(np.uint8) * 255


def _build_instance_id_crop(
    id_map: np.ndarray | None,
    box_xyxy: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    if id_map is None or id_map.ndim != 2:
        return np.zeros((1, int(crop_size), int(crop_size)), dtype=np.int64)
    h, w = id_map.shape
    x0 = max(0, min(int(float(box_xyxy[0])), w - 1))
    y0 = max(0, min(int(float(box_xyxy[1])), h - 1))
    x1 = max(x0 + 1, min(int(float(box_xyxy[2])), w))
    y1 = max(y0 + 1, min(int(float(box_xyxy[3])), h))
    patch = np.asarray(id_map[y0:y1, x0:x1], dtype=np.int32)
    if patch.size == 0:
        patch = np.zeros((1, 1), dtype=np.int32)
    id_img = Image.fromarray(patch, mode="I")
    id_resized = np.asarray(
        id_img.resize((int(crop_size), int(crop_size)), resample=Image.NEAREST),
        dtype=np.int64,
    )
    return id_resized[None, ...]


def _box_from_mask_or_center(mask: np.ndarray | None, width: int, height: int, radius: int) -> Tuple[np.ndarray, bool, float]:
    if mask is not None and mask.shape[0] == height and mask.shape[1] == width:
        fg = mask > 0
        if np.any(fg):
            ys, xs = np.where(fg)
            x0 = int(xs.min())
            y0 = int(ys.min())
            x1 = int(xs.max()) + 1
            y1 = int(ys.max()) + 1
            fg_ratio = float(fg.mean())
            return np.array([x0, y0, x1, y1], dtype=np.float32), True, fg_ratio

    cx = 0.5 * float(width)
    cy = 0.5 * float(height)
    x0, y0, x1, y1 = _safe_crop_bounds(cx=cx, cy=cy, radius=radius, width=width, height=height)
    return np.array([x0, y0, x1, y1], dtype=np.float32), False, 0.0


def _semantic_feature(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    box_xyxy: np.ndarray,
    mask_used: bool,
    fg_ratio: float,
) -> np.ndarray:
    h, w, _ = rgb.shape
    x0 = max(0, min(int(float(box_xyxy[0])), w - 1))
    y0 = max(0, min(int(float(box_xyxy[1])), h - 1))
    x1 = max(x0 + 1, min(int(float(box_xyxy[2])), w))
    y1 = max(y0 + 1, min(int(float(box_xyxy[3])), h))

    patch = rgb[y0:y1, x0:x1]
    if patch.size == 0:
        patch = np.zeros((1, 1, 3), dtype=np.float32)

    if mask is not None and mask.shape[0] == h and mask.shape[1] == w:
        mpatch = mask[y0:y1, x0:x1] > 0
        if mask_used and np.any(mpatch):
            pix = patch[mpatch]
        else:
            pix = patch.reshape(-1, 3)
    else:
        pix = patch.reshape(-1, 3)

    mean_rgb = np.mean(pix, axis=0).astype(np.float32)
    std_rgb = np.std(pix, axis=0).astype(np.float32)
    bw = float((x1 - x0) / max(float(w), 1.0))
    bh = float((y1 - y0) / max(float(h), 1.0))

    return np.array(
        [
            float(mean_rgb[0]),
            float(mean_rgb[1]),
            float(mean_rgb[2]),
            float(std_rgb[0]),
            float(std_rgb[1]),
            float(std_rgb[2]),
            float(bw),
            float(bh),
            float(1.0 if mask_used else 0.0),
            float(fg_ratio),
        ],
        dtype=np.float32,
    )


def _build_semantic_crops(
    rgb: np.ndarray,
    mask: np.ndarray | None,
    box_xyxy: np.ndarray,
    crop_size: int,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    h, w, _ = rgb.shape
    x0 = max(0, min(int(float(box_xyxy[0])), w - 1))
    y0 = max(0, min(int(float(box_xyxy[1])), h - 1))
    x1 = max(x0 + 1, min(int(float(box_xyxy[2])), w))
    y1 = max(y0 + 1, min(int(float(box_xyxy[3])), h))

    rgb_patch = rgb[y0:y1, x0:x1]
    if rgb_patch.size == 0:
        rgb_patch = np.zeros((1, 1, 3), dtype=np.float32)

    rgb_u8 = np.clip(rgb_patch * 255.0, 0.0, 255.0).astype(np.uint8)
    rgb_img = Image.fromarray(rgb_u8, mode="RGB")
    rgb_resized = np.asarray(
        rgb_img.resize((int(crop_size), int(crop_size)), resample=Image.BILINEAR),
        dtype=np.float32,
    )
    rgb_crop = np.transpose(rgb_resized / 255.0, (2, 0, 1)).astype(np.float32)

    mask_available = bool(mask is not None and mask.shape[0] == h and mask.shape[1] == w)
    if mask_available:
        mask_patch = (mask[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    else:
        mask_patch = np.zeros((max(y1 - y0, 1), max(x1 - x0, 1)), dtype=np.uint8)

    mask_img = Image.fromarray(mask_patch, mode="L")
    mask_resized = np.asarray(
        mask_img.resize((int(crop_size), int(crop_size)), resample=Image.NEAREST),
        dtype=np.float32,
    )
    mask_crop = (mask_resized[None, ...] / 255.0).astype(np.float32)
    return rgb_crop, mask_crop, bool(mask_available)


def _build_state_from_boxes(boxes: List[np.ndarray], sizes: List[Tuple[int, int]]) -> np.ndarray:
    t = len(boxes)
    state = np.zeros((t, STATE_DIM), dtype=np.float32)

    prev_x = None
    prev_y = None
    for i in range(t):
        box = boxes[i]
        w, h = sizes[i]
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]

        cx = 0.5 * (x0 + x1) / max(float(w), 1.0)
        cy = 0.5 * (y0 + y1) / max(float(h), 1.0)
        rx = (x1 - x0) / max(float(w), 1.0)
        ry = (y1 - y0) / max(float(h), 1.0)

        vx = 0.0 if prev_x is None else (cx - prev_x)
        vy = 0.0 if prev_y is None else (cy - prev_y)
        prev_x, prev_y = cx, cy

        state[i, 0] = cx
        state[i, 1] = cy
        state[i, 2] = 0.0
        state[i, 3] = 1.0
        state[i, 4] = vx
        state[i, 5] = vy
        state[i, 6] = rx
        state[i, 7] = ry

    return state


class Stage2SemanticDataset(Dataset):
    """Stage2 bootstrap dataset on top of VSPW/VIPSeg/BURST audit-approved inputs."""

    def __init__(self, cfg: Stage2SemanticDatasetConfig) -> None:
        self.cfg = cfg
        self.contract = _safe_json(cfg.contract_path)

        records = self.contract.get("datasets", [])
        if not isinstance(records, list):
            records = []

        by_name: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            key = _norm_name(str(rec.get("dataset_name", "")))
            if key:
                by_name[key] = rec

        requested = [_norm_name(x) for x in cfg.dataset_names]
        self.entries: List[Dict[str, Any]] = []
        self.dataset_summary: Dict[str, Dict[str, Any]] = {}
        self.predecode_index: Dict[str, str] = {}
        cache_root = str(cfg.predecode_cache_path).strip()
        if cache_root:
            cache_root_path = Path(cache_root)
            index_path = cache_root_path if cache_root_path.suffix.lower() == ".json" else cache_root_path / "index.json"
            if index_path.exists():
                payload = _safe_json(index_path)
                raw_entries = payload.get("entries", {}) if isinstance(payload.get("entries", {}), dict) else {}
                self.predecode_index = {str(k): str(v) for k, v in raw_entries.items()}

        for name in requested:
            rec = by_name.get(name)
            if rec is None:
                continue
            if bool(rec.get("not_in_current_bootstrap", False)):
                continue

            if name == "VSPW":
                ds_entries = self._scan_split_file_dataset(rec=rec, split=cfg.split, dataset_display_name="VSPW")
            elif name == "VIPSEG":
                ds_entries = self._scan_split_file_dataset(rec=rec, split=cfg.split, dataset_display_name="VIPSeg")
            elif name == "BURST":
                ds_entries = self._scan_burst_dataset(rec=rec, split=cfg.split)
            else:
                ds_entries = []

            source_count_before_limit = int(len(ds_entries))
            requested_limit = int(cfg.max_samples_per_dataset)
            full_dataset_used = bool(requested_limit < 0)
            if not full_dataset_used:
                max_take = max(int(requested_limit), 1)
                ds_entries = ds_entries[:max_take]
            self.entries.extend(ds_entries)

            display_name = str(rec.get("dataset_name", name))
            self.dataset_summary[display_name] = {
                "sample_count": int(len(ds_entries)),
                "source_count_before_limit": int(source_count_before_limit),
                "requested_max_samples_per_dataset": int(requested_limit),
                "full_dataset_used": bool(full_dataset_used),
                "split": str(cfg.split),
                "used_in_bootstrap_train": bool(rec.get("used_in_bootstrap_train", False)),
                "used_in_bootstrap_eval": bool(rec.get("used_in_bootstrap_eval", False)),
                "local_path": str(rec.get("local_path", "")),
            }

        if not self.entries:
            raise RuntimeError(
                "no Stage2 semantic samples found from requested datasets "
                f"{cfg.dataset_names} split={cfg.split}"
            )

    def _scan_split_file_dataset(self, rec: Dict[str, Any], split: str, dataset_display_name: str) -> List[Dict[str, Any]]:
        mapping = rec.get("split_mapping", {}) if isinstance(rec.get("split_mapping", {}), dict) else {}
        split_cfg = mapping.get(split, {}) if isinstance(mapping.get(split, {}), dict) else {}

        split_file = str(split_cfg.get("split_file", ""))
        frame_root = Path(str(split_cfg.get("frame_root", "")))
        frame_subdir = str(split_cfg.get("frame_subdir", "")).strip()
        mask_root = Path(str(split_cfg.get("mask_root", "")))
        mask_subdir = str(split_cfg.get("mask_subdir", "")).strip()

        entries: List[Dict[str, Any]] = []
        for clip_id in _read_split_ids(split_file):
            frame_dir = frame_root / clip_id
            if frame_subdir:
                frame_dir = frame_dir / frame_subdir
            frames = _list_frames(frame_dir)
            if len(frames) < 2:
                continue

            mask_dir = mask_root / clip_id
            if mask_subdir:
                mask_dir = mask_dir / mask_subdir
            masks = _align_masks_by_name(frames, mask_dir)

            entries.append(
                {
                    "dataset_name": dataset_display_name,
                    "clip_id": clip_id,
                    "frame_paths": frames,
                    "mask_paths": masks,
                    "annotation_source": str(rec.get("annotation_source", "")),
                }
            )
        return entries

    def _scan_burst_dataset(self, rec: Dict[str, Any], split: str) -> List[Dict[str, Any]]:
        mapping = rec.get("split_mapping", {}) if isinstance(rec.get("split_mapping", {}), dict) else {}
        split_cfg = mapping.get(split, {}) if isinstance(mapping.get(split, {}), dict) else {}

        frames_root = Path(str(split_cfg.get("frames_root", "")))
        annotation_file = str(split_cfg.get("annotation_file", ""))
        if not frames_root.exists():
            return []

        entries: List[Dict[str, Any]] = []
        for source_dir in sorted([p for p in frames_root.iterdir() if p.is_dir()]):
            for clip_dir in sorted([p for p in source_dir.iterdir() if p.is_dir()]):
                frames = _list_frames(clip_dir)
                if len(frames) < 2:
                    continue
                entries.append(
                    {
                        "dataset_name": "BURST",
                        "clip_id": f"{source_dir.name}/{clip_dir.name}",
                        "frame_paths": frames,
                        "mask_paths": ["" for _ in frames],
                        "annotation_source": annotation_file,
                    }
                )
        return entries

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        entry = self.entries[index]
        dataset_name = str(entry.get("dataset_name", ""))
        true_instance_aware = bool(str(dataset_name).strip().upper() == "VIPSEG")
        cache_key = _cache_key(str(entry.get("dataset_name", "")), str(self.cfg.split), str(entry.get("clip_id", "")))
        cache_path = self.predecode_index.get(cache_key, "")
        if cache_path:
            cp = Path(cache_path)
            if cp.exists():
                try:
                    with np.load(cp, allow_pickle=True) as payload:
                        required_instance_keys = {
                            "semantic_instance_id_crop",
                            "semantic_instance_id_temporal",
                            "semantic_instance_valid",
                            "semantic_objectness_score",
                        }
                        if not required_instance_keys.issubset(set(payload.files)):
                            raise KeyError("instance_aware_fields_missing_in_predecode_cache")
                        meta_obj = payload["meta_json"].tolist()
                        source_summary_obj = payload["semantic_source_summary_json"].tolist()
                        rgb_temporal = payload["semantic_rgb_crop_temporal"]
                        mask_temporal = payload["semantic_mask_crop_temporal"]
                        valid_temporal = payload["semantic_temporal_valid"]
                        target_window = max(int(self.cfg.semantic_temporal_window), 1)
                        if rgb_temporal.ndim == 4:
                            rgb_temporal = rgb_temporal[None, ...]
                        if mask_temporal.ndim == 4:
                            mask_temporal = mask_temporal[None, ...]
                        if valid_temporal.ndim == 1:
                            valid_temporal = valid_temporal[None, ...]
                        current_window = int(rgb_temporal.shape[1])
                        if current_window > target_window:
                            rgb_temporal = rgb_temporal[:, :target_window, ...]
                            mask_temporal = mask_temporal[:, :target_window, ...]
                            valid_temporal = valid_temporal[:, :target_window, ...]
                        elif current_window < target_window:
                            pad_rgb = np.zeros(
                                (int(rgb_temporal.shape[0]), target_window - current_window, int(rgb_temporal.shape[2]), int(rgb_temporal.shape[3]), int(rgb_temporal.shape[4])),
                                dtype=rgb_temporal.dtype,
                            )
                            pad_mask = np.zeros(
                                (int(mask_temporal.shape[0]), target_window - current_window, int(mask_temporal.shape[2]), int(mask_temporal.shape[3]), int(mask_temporal.shape[4])),
                                dtype=mask_temporal.dtype,
                            )
                            pad_valid = np.zeros(
                                (int(valid_temporal.shape[0]), target_window - current_window),
                                dtype=valid_temporal.dtype,
                            )
                            rgb_temporal = np.concatenate([rgb_temporal, pad_rgb], axis=1)
                            mask_temporal = np.concatenate([mask_temporal, pad_mask], axis=1)
                            valid_temporal = np.concatenate([valid_temporal, pad_valid], axis=1)
                        return {
                            "obs_state": torch.from_numpy(payload["obs_state"]).to(torch.float32),
                            "fut_state": torch.from_numpy(payload["fut_state"]).to(torch.float32),
                            "obs_valid": torch.from_numpy(payload["obs_valid"]).to(torch.bool),
                            "fut_valid": torch.from_numpy(payload["fut_valid"]).to(torch.bool),
                            "point_ids": torch.from_numpy(payload["point_ids"]).to(torch.long),
                            "meta": dict(meta_obj if isinstance(meta_obj, dict) else {}),
                            "semantic_features": torch.from_numpy(payload["semantic_features"]).to(torch.float32),
                            "semantic_boxes": torch.from_numpy(payload["semantic_boxes"]).to(torch.float32),
                            "semantic_mask": torch.from_numpy(payload["semantic_mask"]).to(torch.bool),
                            "semantic_rgb_crop": torch.from_numpy(payload["semantic_rgb_crop"]).to(torch.float32),
                            "semantic_mask_crop": torch.from_numpy(payload["semantic_mask_crop"]).to(torch.float32),
                            "semantic_crop_valid": torch.from_numpy(payload["semantic_crop_valid"]).to(torch.bool),
                            "semantic_mask_crop_valid": torch.from_numpy(payload["semantic_mask_crop_valid"]).to(torch.bool),
                            "semantic_rgb_crop_temporal": torch.from_numpy(rgb_temporal).to(torch.float32),
                            "semantic_mask_crop_temporal": torch.from_numpy(mask_temporal).to(torch.float32),
                            "semantic_temporal_valid": torch.from_numpy(valid_temporal).to(torch.bool),
                            "semantic_instance_id_map": torch.from_numpy(
                                payload["semantic_instance_id_map"] if "semantic_instance_id_map" in payload.files else np.zeros((1, 1), dtype=np.int64)
                            ).to(torch.long),
                            "semantic_instance_id_crop": torch.from_numpy(payload["semantic_instance_id_crop"]).to(torch.long),
                            "semantic_instance_id_temporal": torch.from_numpy(payload["semantic_instance_id_temporal"]).to(torch.long),
                            "semantic_instance_valid": torch.from_numpy(payload["semantic_instance_valid"]).to(torch.bool),
                            "semantic_objectness_score": torch.from_numpy(payload["semantic_objectness_score"]).to(torch.float32),
                            "semantic_frame_path": str(payload["semantic_frame_path"].tolist()),
                            "semantic_mask_path": str(payload["semantic_mask_path"].tolist()),
                            "semantic_source_mode": "object_region_or_mask_crop_visual_state",
                            "current_mainline_semantic_source": str(self.cfg.semantic_source_mainline),
                            "legacy_semantic_source": "hand_crafted_stats",
                            "semantic_source_summary": dict(source_summary_obj if isinstance(source_summary_obj, dict) else {}),
                        }
                except KeyError:
                    pass
                except Exception:
                    try:
                        cp.unlink()
                    except Exception:
                        pass
        frame_paths = [str(x) for x in entry["frame_paths"]]
        mask_paths = [str(x) for x in entry["mask_paths"]]

        total_steps = int(self.cfg.obs_len) + int(self.cfg.fut_len)
        indices = _temporal_indices(frame_count=len(frame_paths), total_steps=total_steps)

        semantic_step = max(0, min(int(self.cfg.semantic_frame_index), total_steps - 1))

        boxes: List[np.ndarray] = []
        sizes: List[Tuple[int, int]] = []

        sem_rgb = None
        sem_mask = None
        sem_box = None
        sem_instance_id_map = None
        sem_instance_crop = None
        sem_instance_valid = False
        sem_frame_path = ""
        sem_mask_path = ""
        sem_mask_used = False
        sem_fg_ratio = 0.0
        semantic_objectness_score = 0.0
        temporal_rgb_crops: List[np.ndarray] = []
        temporal_mask_crops: List[np.ndarray] = []
        temporal_valid_flags: List[bool] = []
        temporal_instance_crops: List[np.ndarray] = []
        temporal_instance_valid_flags: List[bool] = []
        target_instance_id = 0

        for step_i, src_i in enumerate(indices):
            frame_path = frame_paths[src_i]
            mask_path = mask_paths[src_i] if src_i < len(mask_paths) else ""

            with Image.open(frame_path) as img_obj:
                img = img_obj.convert("RGB")
                w, h = img.size
                if step_i == semantic_step:
                    sem_rgb = np.asarray(img, dtype=np.float32) / 255.0

            mask_arr = _load_mask(mask_path)
            raw_instance_map = np.asarray(mask_arr, dtype=np.int64) if mask_arr is not None else None
            if true_instance_aware and step_i == semantic_step:
                target_instance_id = _dominant_positive_instance_id(raw_instance_map)
            instance_mask_arr = _instance_mask_from_id_map(raw_instance_map, target_instance_id) if true_instance_aware else None
            effective_mask = instance_mask_arr if instance_mask_arr is not None else mask_arr
            box, used_mask, fg_ratio = _box_from_mask_or_center(
                mask=effective_mask,
                width=w,
                height=h,
                radius=int(self.cfg.semantic_patch_radius),
            )

            boxes.append(box)
            sizes.append((w, h))

            if step_i == semantic_step:
                sem_mask = effective_mask
                sem_box = box
                sem_instance_id_map = raw_instance_map if true_instance_aware else None
                sem_instance_crop = _build_instance_id_crop(
                    sem_instance_id_map if true_instance_aware else None,
                    box_xyxy=box,
                    crop_size=int(self.cfg.semantic_crop_size),
                )
                sem_instance_valid = bool(true_instance_aware and target_instance_id > 0 and instance_mask_arr is not None)
                sem_frame_path = frame_path
                sem_mask_path = mask_path if mask_path and Path(mask_path).exists() else ""
                sem_mask_used = bool(used_mask)
                sem_fg_ratio = float(fg_ratio)
                semantic_objectness_score = float(fg_ratio if sem_instance_valid else fg_ratio)
            if step_i < max(int(self.cfg.semantic_temporal_window), 1):
                rgb_for_temporal = sem_rgb if step_i == semantic_step and sem_rgb is not None else None
                if rgb_for_temporal is None:
                    with Image.open(frame_path) as img_obj:
                        rgb_for_temporal = np.asarray(img_obj.convert("RGB"), dtype=np.float32) / 255.0
                rgb_crop_t, mask_crop_t, mask_valid_t = _build_semantic_crops(
                    rgb=rgb_for_temporal,
                    mask=effective_mask,
                    box_xyxy=box,
                    crop_size=int(self.cfg.semantic_crop_size),
                )
                temporal_rgb_crops.append(rgb_crop_t)
                temporal_mask_crops.append(mask_crop_t)
                temporal_valid_flags.append(bool(mask_valid_t))
                temporal_instance_crops.append(
                    _build_instance_id_crop(
                        raw_instance_map if true_instance_aware else None,
                        box_xyxy=box,
                        crop_size=int(self.cfg.semantic_crop_size),
                    )
                )
                temporal_instance_valid_flags.append(bool(true_instance_aware and target_instance_id > 0 and instance_mask_arr is not None))

        if sem_rgb is None or sem_box is None:
            with Image.open(frame_paths[0]) as fallback:
                sem_rgb = np.asarray(fallback.convert("RGB"), dtype=np.float32) / 255.0
            sem_mask = _load_mask(mask_paths[0] if mask_paths else "")
            sem_box = boxes[0]

        semantic_rgb_crop, semantic_mask_crop, mask_crop_available = _build_semantic_crops(
            rgb=sem_rgb,
            mask=sem_mask,
            box_xyxy=sem_box,
            crop_size=int(self.cfg.semantic_crop_size),
        )

        semantic_feat = _semantic_feature(
            rgb=sem_rgb,
            mask=sem_mask,
            box_xyxy=sem_box,
            mask_used=sem_mask_used,
            fg_ratio=sem_fg_ratio,
        )
        temporal_window = max(int(self.cfg.semantic_temporal_window), 1)
        while len(temporal_rgb_crops) < temporal_window:
            temporal_rgb_crops.append(np.zeros_like(semantic_rgb_crop))
            temporal_mask_crops.append(np.zeros_like(semantic_mask_crop))
            temporal_valid_flags.append(False)
            temporal_instance_crops.append(np.zeros((1, int(self.cfg.semantic_crop_size), int(self.cfg.semantic_crop_size)), dtype=np.int64))
            temporal_instance_valid_flags.append(False)
        temporal_rgb = np.stack(temporal_rgb_crops[:temporal_window], axis=0).astype(np.float32)
        temporal_mask = np.stack(temporal_mask_crops[:temporal_window], axis=0).astype(np.float32)
        temporal_valid = np.asarray(temporal_valid_flags[:temporal_window], dtype=bool)
        temporal_instance = np.stack(temporal_instance_crops[:temporal_window], axis=0).astype(np.int64)
        temporal_instance_valid = np.asarray(temporal_instance_valid_flags[:temporal_window], dtype=bool)

        state = _build_state_from_boxes(boxes=boxes, sizes=sizes)
        valid = np.ones((state.shape[0], 1), dtype=bool)

        obs = int(self.cfg.obs_len)
        obs_state = state[:obs, None, :]
        fut_state = state[obs:, None, :]
        obs_valid = valid[:obs]
        fut_valid = valid[obs:]

        sample: Dict[str, Any] = {
            "obs_state": torch.from_numpy(obs_state).to(torch.float32),
            "fut_state": torch.from_numpy(fut_state).to(torch.float32),
            "obs_valid": torch.from_numpy(obs_valid).to(torch.bool),
            "fut_valid": torch.from_numpy(fut_valid).to(torch.bool),
            "point_ids": torch.tensor([0], dtype=torch.long),
            "meta": {
                "dataset": str(entry.get("dataset_name", "")),
                "clip_id": str(entry.get("clip_id", "")),
                "annotation_source": str(entry.get("annotation_source", "")),
                "frame_count_total": int(len(frame_paths)),
            },
            "semantic_features": torch.from_numpy(semantic_feat[None, :]).to(torch.float32),
            "semantic_boxes": torch.from_numpy(np.asarray([sem_box], dtype=np.float32)).to(torch.float32),
            "semantic_mask": torch.tensor([True], dtype=torch.bool),
            "semantic_rgb_crop": torch.from_numpy(semantic_rgb_crop[None, ...]).to(torch.float32),
            "semantic_mask_crop": torch.from_numpy(semantic_mask_crop[None, ...]).to(torch.float32),
            "semantic_crop_valid": torch.tensor([True], dtype=torch.bool),
            "semantic_mask_crop_valid": torch.tensor([bool(mask_crop_available)], dtype=torch.bool),
            "semantic_rgb_crop_temporal": torch.from_numpy(temporal_rgb[None, ...]).to(torch.float32),
            "semantic_mask_crop_temporal": torch.from_numpy(temporal_mask[None, ...]).to(torch.float32),
            "semantic_temporal_valid": torch.from_numpy(temporal_valid[None, ...]).to(torch.bool),
            "semantic_instance_id_map": torch.from_numpy(
                np.asarray(sem_instance_id_map, dtype=np.int64) if isinstance(sem_instance_id_map, np.ndarray) else np.zeros((1, 1), dtype=np.int64)
            ).to(torch.long),
            "semantic_instance_id_crop": torch.from_numpy(
                (sem_instance_crop if isinstance(sem_instance_crop, np.ndarray) else np.zeros((1, int(self.cfg.semantic_crop_size), int(self.cfg.semantic_crop_size)), dtype=np.int64))[None, ...]
            ).to(torch.long),
            "semantic_instance_id_temporal": torch.from_numpy(temporal_instance[None, ...]).to(torch.long),
            "semantic_instance_valid": torch.from_numpy(temporal_instance_valid[None, ...]).to(torch.bool),
            "semantic_objectness_score": torch.tensor([float(semantic_objectness_score)], dtype=torch.float32),
            "semantic_frame_path": sem_frame_path,
            "semantic_mask_path": sem_mask_path,
            "semantic_source_mode": "object_region_or_mask_crop_visual_state",
            "current_mainline_semantic_source": str(self.cfg.semantic_source_mainline),
            "legacy_semantic_source": "hand_crafted_stats",
            "semantic_source_summary": {
                "mask_crop_used_tokens": int(1 if sem_mask_used else 0),
                "region_crop_used_tokens": int(0 if sem_mask_used else 1),
                "mask_available": bool(sem_mask_path),
                "instance_aware_source": "true_instance_id" if bool(sem_instance_valid) else ("fallback_null" if str(dataset_name).strip().upper() == "VSPW" else "fallback_binary_or_none"),
                "target_instance_id": int(target_instance_id),
                "semantic_crop_size": int(self.cfg.semantic_crop_size),
                "current_mainline_semantic_source": str(self.cfg.semantic_source_mainline),
                "legacy_semantic_source": "hand_crafted_stats",
                "legacy_semantic_feature_dim": int(SEMANTIC_FEATURE_DIM),
            },
        }
        return sample


def stage2_semantic_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        return {}

    bsz = len(batch)
    obs_len = int(batch[0]["obs_state"].shape[0])
    fut_len = int(batch[0]["fut_state"].shape[0])
    feat_dim = int(batch[0]["semantic_features"].shape[-1])
    crop_h = int(batch[0]["semantic_rgb_crop"].shape[-2])
    crop_w = int(batch[0]["semantic_rgb_crop"].shape[-1])
    max_k = max(int(item["obs_state"].shape[1]) for item in batch)

    obs_state = torch.zeros((bsz, obs_len, max_k, STATE_DIM), dtype=torch.float32)
    fut_state = torch.zeros((bsz, fut_len, max_k, STATE_DIM), dtype=torch.float32)
    obs_valid = torch.zeros((bsz, obs_len, max_k), dtype=torch.bool)
    fut_valid = torch.zeros((bsz, fut_len, max_k), dtype=torch.bool)
    token_mask = torch.zeros((bsz, max_k), dtype=torch.bool)
    point_ids = torch.full((bsz, max_k), fill_value=-1, dtype=torch.long)

    semantic_features = torch.zeros((bsz, max_k, feat_dim), dtype=torch.float32)
    semantic_boxes = torch.zeros((bsz, max_k, 4), dtype=torch.float32)
    semantic_mask = torch.zeros((bsz, max_k), dtype=torch.bool)
    semantic_rgb_crop = torch.zeros((bsz, max_k, 3, crop_h, crop_w), dtype=torch.float32)
    semantic_mask_crop = torch.zeros((bsz, max_k, 1, crop_h, crop_w), dtype=torch.float32)
    semantic_crop_valid = torch.zeros((bsz, max_k), dtype=torch.bool)
    semantic_mask_crop_valid = torch.zeros((bsz, max_k), dtype=torch.bool)
    temporal_window = int(batch[0]["semantic_rgb_crop_temporal"].shape[1])
    semantic_rgb_crop_temporal = torch.zeros((bsz, max_k, temporal_window, 3, crop_h, crop_w), dtype=torch.float32)
    semantic_mask_crop_temporal = torch.zeros((bsz, max_k, temporal_window, 1, crop_h, crop_w), dtype=torch.float32)
    semantic_temporal_valid = torch.zeros((bsz, max_k, temporal_window), dtype=torch.bool)
    semantic_instance_id_crop = torch.zeros((bsz, max_k, 1, crop_h, crop_w), dtype=torch.long)
    semantic_instance_id_temporal = torch.zeros((bsz, max_k, temporal_window, 1, crop_h, crop_w), dtype=torch.long)
    semantic_instance_valid = torch.zeros((bsz, max_k, temporal_window), dtype=torch.bool)
    semantic_objectness_score = torch.zeros((bsz, max_k), dtype=torch.float32)

    semantic_frame_paths: List[str] = []
    semantic_mask_paths: List[str] = []
    semantic_source_summaries: List[Dict[str, Any]] = []
    semantic_instance_id_map: List[torch.Tensor] = []
    meta: List[Dict[str, Any]] = []

    for i, item in enumerate(batch):
        k = int(item["obs_state"].shape[1])

        obs_state[i, :, :k] = item["obs_state"]
        fut_state[i, :, :k] = item["fut_state"]
        obs_valid[i, :, :k] = item["obs_valid"]
        fut_valid[i, :, :k] = item["fut_valid"]
        token_mask[i, :k] = True
        point_ids[i, :k] = item["point_ids"]

        semantic_features[i, :k] = item["semantic_features"]
        semantic_boxes[i, :k] = item["semantic_boxes"]
        semantic_mask[i, :k] = item["semantic_mask"]
        semantic_rgb_crop[i, :k] = item["semantic_rgb_crop"]
        semantic_mask_crop[i, :k] = item["semantic_mask_crop"]
        semantic_crop_valid[i, :k] = item["semantic_crop_valid"]
        semantic_mask_crop_valid[i, :k] = item["semantic_mask_crop_valid"]
        semantic_rgb_crop_temporal[i, :k] = item["semantic_rgb_crop_temporal"]
        semantic_mask_crop_temporal[i, :k] = item["semantic_mask_crop_temporal"]
        semantic_temporal_valid[i, :k] = item["semantic_temporal_valid"]
        semantic_instance_id_crop[i, :k] = item["semantic_instance_id_crop"]
        semantic_instance_id_temporal[i, :k] = item["semantic_instance_id_temporal"]
        semantic_instance_valid[i, :k] = item["semantic_instance_valid"]
        semantic_objectness_score[i, :k] = item["semantic_objectness_score"]

        semantic_frame_paths.append(str(item.get("semantic_frame_path", "")))
        semantic_mask_paths.append(str(item.get("semantic_mask_path", "")))
        semantic_source_summaries.append(dict(item.get("semantic_source_summary", {})))
        semantic_instance_id_map.append(item.get("semantic_instance_id_map"))
        meta.append(dict(item.get("meta", {})))

    return {
        "batch_size": int(bsz),
        "obs_state": obs_state,
        "fut_state": fut_state,
        "obs_valid": obs_valid,
        "fut_valid": fut_valid,
        "token_mask": token_mask,
        "point_ids": point_ids,
        "meta": meta,
        "semantic_features": semantic_features,
        "semantic_boxes": semantic_boxes,
        "semantic_mask": semantic_mask,
        "semantic_rgb_crop": semantic_rgb_crop,
        "semantic_mask_crop": semantic_mask_crop,
        "semantic_crop_valid": semantic_crop_valid,
        "semantic_mask_crop_valid": semantic_mask_crop_valid,
        "semantic_rgb_crop_temporal": semantic_rgb_crop_temporal,
        "semantic_mask_crop_temporal": semantic_mask_crop_temporal,
        "semantic_temporal_valid": semantic_temporal_valid,
        "semantic_instance_id_map": semantic_instance_id_map,
        "semantic_instance_id_crop": semantic_instance_id_crop,
        "semantic_instance_id_temporal": semantic_instance_id_temporal,
        "semantic_instance_valid": semantic_instance_valid,
        "semantic_objectness_score": semantic_objectness_score,
        "semantic_frame_paths": semantic_frame_paths,
        "semantic_mask_paths": semantic_mask_paths,
        "semantic_source_mode": "object_region_or_mask_crop_visual_state",
        "current_mainline_semantic_source": str(batch[0].get("current_mainline_semantic_source", "crop_visual_encoder")),
        "legacy_semantic_source": str(batch[0].get("legacy_semantic_source", "hand_crafted_stats")),
        "semantic_source_summaries": semantic_source_summaries,
    }
