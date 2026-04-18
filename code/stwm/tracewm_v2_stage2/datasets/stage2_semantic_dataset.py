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
    teacher_semantic_cache_path: str = ""
    max_entities_per_sample: int = 8
    include_entity_masks_over_time: bool = False
    include_full_instance_id_map: bool = False


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


def _largest_positive_labels(id_map: np.ndarray | None, limit: int) -> List[int]:
    if id_map is None:
        return []
    arr = np.asarray(id_map, dtype=np.int64)
    vals, counts = np.unique(arr[arr > 0], return_counts=True)
    if vals.size <= 0:
        return []
    order = np.argsort(-counts)
    return [int(vals[idx]) for idx in order[: max(int(limit), 1)].tolist()]


def _vipseg_is_instance_id(label_id: int) -> bool:
    return int(label_id) >= 125


def _vipseg_semantic_id(label_id: int) -> int:
    if int(label_id) < 125:
        return int(label_id) - 1
    return int(label_id) // 100 - 1


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


def _select_entity_ids_for_sample(
    *,
    dataset_name: str,
    raw_id_maps: List[np.ndarray | None],
    semantic_step: int,
    max_entities: int,
) -> Tuple[List[int], str]:
    max_entities = max(int(max_entities), 1)
    name = str(dataset_name).strip().upper()
    anchor = raw_id_maps[min(max(int(semantic_step), 0), max(len(raw_id_maps) - 1, 0))] if raw_id_maps else None
    if name == "VIPSEG":
        candidate_ids: List[int] = []
        if isinstance(anchor, np.ndarray):
            for label_id in _largest_positive_labels(anchor, limit=max_entities * 4):
                if _vipseg_is_instance_id(int(label_id)):
                    candidate_ids.append(int(label_id))
        scored: List[Tuple[int, float, int]] = []
        for entity_id in candidate_ids:
            presence = 0
            mean_area = 0.0
            for id_map in raw_id_maps:
                if id_map is None:
                    continue
                mask = np.asarray(id_map, dtype=np.int64) == int(entity_id)
                if bool(np.any(mask)):
                    presence += 1
                    mean_area += float(mask.mean())
            if presence > 0:
                scored.append((-presence, -mean_area, int(entity_id)))
        scored.sort()
        return [int(x[2]) for x in scored[:max_entities]], "true_instance_id"

    if name == "VSPW":
        labels = _largest_positive_labels(anchor, limit=max_entities)
        return [int(x) for x in labels[:max_entities]], "pseudo_semantic_region"

    labels = _largest_positive_labels(anchor, limit=max_entities)
    return [int(x) for x in labels[:max_entities]], "fallback_label_region"


def _entity_mask_for_dataset(
    *,
    dataset_name: str,
    raw_id_map: np.ndarray | None,
    entity_id: int,
) -> np.ndarray | None:
    if raw_id_map is None or int(entity_id) <= 0:
        return None
    arr = np.asarray(raw_id_map, dtype=np.int64)
    mask = arr == int(entity_id)
    if not np.any(mask):
        return None
    return mask.astype(np.uint8) * 255


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
        self.teacher_cache_index: Dict[str, str] = {}
        cache_root = str(cfg.predecode_cache_path).strip()
        if cache_root:
            cache_root_path = Path(cache_root)
            index_path = cache_root_path if cache_root_path.suffix.lower() == ".json" else cache_root_path / "index.json"
            if index_path.exists():
                payload = _safe_json(index_path)
                raw_entries = payload.get("entries", {}) if isinstance(payload.get("entries", {}), dict) else {}
                self.predecode_index = {str(k): str(v) for k, v in raw_entries.items()}
        teacher_cache_root = str(cfg.teacher_semantic_cache_path).strip()
        if teacher_cache_root:
            cache_root_path = Path(teacher_cache_root)
            index_path = cache_root_path if cache_root_path.suffix.lower() == ".json" else cache_root_path / "index.json"
            if index_path.exists():
                payload = _safe_json(index_path)
                raw_entries = payload.get("entries", {}) if isinstance(payload.get("entries", {}), dict) else {}
                self.teacher_cache_index = {str(k): str(v) for k, v in raw_entries.items()}

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
        dataset_upper = str(dataset_name).strip().upper()
        true_instance_aware = bool(dataset_upper == "VIPSEG")
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
                        if "entity_boxes_over_time" not in payload.files:
                            raise KeyError("multi_entity_fields_missing_in_predecode_cache")
                        meta_obj = payload["meta_json"].tolist()
                        source_summary_obj = payload["semantic_source_summary_json"].tolist()
                        rgb_temporal = payload["semantic_rgb_crop_temporal"]
                        mask_temporal = payload["semantic_mask_crop_temporal"]
                        valid_temporal = payload["semantic_temporal_valid"]
                        instance_id_temporal = payload["semantic_instance_id_temporal"]
                        instance_valid_temporal = payload["semantic_instance_valid"]
                        target_window = max(int(self.cfg.semantic_temporal_window), 1)
                        if rgb_temporal.ndim == 4:
                            rgb_temporal = rgb_temporal[None, ...]
                        if mask_temporal.ndim == 4:
                            mask_temporal = mask_temporal[None, ...]
                        if valid_temporal.ndim == 1:
                            valid_temporal = valid_temporal[None, ...]
                        if instance_id_temporal.ndim == 4:
                            instance_id_temporal = instance_id_temporal[None, ...]
                        if instance_valid_temporal.ndim == 1:
                            instance_valid_temporal = instance_valid_temporal[None, ...]
                        current_window = int(rgb_temporal.shape[1])
                        if current_window > target_window:
                            rgb_temporal = rgb_temporal[:, :target_window, ...]
                            mask_temporal = mask_temporal[:, :target_window, ...]
                            valid_temporal = valid_temporal[:, :target_window, ...]
                            instance_id_temporal = instance_id_temporal[:, :target_window, ...]
                            instance_valid_temporal = instance_valid_temporal[:, :target_window, ...]
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
                            pad_instance_ids = np.zeros(
                                (
                                    int(instance_id_temporal.shape[0]),
                                    target_window - current_window,
                                    int(instance_id_temporal.shape[2]),
                                    int(instance_id_temporal.shape[3]),
                                    int(instance_id_temporal.shape[4]),
                                ),
                                dtype=instance_id_temporal.dtype,
                            )
                            pad_instance_valid = np.zeros(
                                (int(instance_valid_temporal.shape[0]), target_window - current_window),
                                dtype=instance_valid_temporal.dtype,
                            )
                            rgb_temporal = np.concatenate([rgb_temporal, pad_rgb], axis=1)
                            mask_temporal = np.concatenate([mask_temporal, pad_mask], axis=1)
                            valid_temporal = np.concatenate([valid_temporal, pad_valid], axis=1)
                            instance_id_temporal = np.concatenate([instance_id_temporal, pad_instance_ids], axis=1)
                            instance_valid_temporal = np.concatenate([instance_valid_temporal, pad_instance_valid], axis=1)
                        teacher_prior = None
                        teacher_cache_path = self.teacher_cache_index.get(cache_key, "")
                        if teacher_cache_path:
                            tp = Path(teacher_cache_path)
                            if tp.exists():
                                try:
                                    with np.load(tp, allow_pickle=True) as teacher_payload:
                                        teacher_prior = np.asarray(
                                            teacher_payload["semantic_teacher_prior"],
                                            dtype=np.float32,
                                        )
                                except Exception:
                                    teacher_prior = None
                        if teacher_prior is None:
                            teacher_prior = np.zeros(
                                (int(payload["point_ids"].shape[0]), 512),
                                dtype=np.float32,
                            )
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
                                payload["semantic_instance_id_map"]
                                if bool(self.cfg.include_full_instance_id_map) and "semantic_instance_id_map" in payload.files
                                else np.zeros((1, 1), dtype=np.int64)
                            ).to(torch.long),
                            "semantic_instance_id_crop": torch.from_numpy(payload["semantic_instance_id_crop"]).to(torch.long),
                            "semantic_instance_id_temporal": torch.from_numpy(instance_id_temporal).to(torch.long),
                            "semantic_instance_valid": torch.from_numpy(instance_valid_temporal).to(torch.bool),
                            "semantic_objectness_score": torch.from_numpy(payload["semantic_objectness_score"]).to(torch.float32),
                            "semantic_teacher_prior": torch.from_numpy(teacher_prior).to(torch.float32),
                            "entity_boxes_over_time": torch.from_numpy(payload["entity_boxes_over_time"]).to(torch.float32),
                            # Full per-entity raw mask sequences are not consumed by the trainer/eval path.
                            # Keep the field lightweight by default to avoid dataloader IPC stalls.
                            "entity_masks_over_time": (
                                payload["entity_masks_over_time"].tolist()
                                if bool(self.cfg.include_entity_masks_over_time) and "entity_masks_over_time" in payload.files
                                else []
                            ),
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

        sizes: List[Tuple[int, int]] = []
        raw_id_maps: List[np.ndarray | None] = []
        for step_i, src_i in enumerate(indices):
            frame_path = frame_paths[src_i]
            mask_path = mask_paths[src_i] if src_i < len(mask_paths) else ""
            with Image.open(frame_path) as img_obj:
                img = img_obj.convert("RGB")
                w, h = img.size
            sizes.append((w, h))
            mask_arr = _load_mask(mask_path)
            raw_id_maps.append(np.asarray(mask_arr, dtype=np.int64) if mask_arr is not None else None)

        entity_ids, instance_source = _select_entity_ids_for_sample(
            dataset_name=dataset_name,
            raw_id_maps=raw_id_maps,
            semantic_step=semantic_step,
            max_entities=int(self.cfg.max_entities_per_sample),
        )
        if not entity_ids:
            fallback_map = raw_id_maps[min(max(semantic_step, 0), max(len(raw_id_maps) - 1, 0))] if raw_id_maps else None
            entity_ids = _largest_positive_labels(fallback_map, limit=1) or [1]
            instance_source = "fallback_single_region"

        crop_size = int(self.cfg.semantic_crop_size)
        temporal_window = max(int(self.cfg.semantic_temporal_window), 1)
        total_steps = int(self.cfg.obs_len) + int(self.cfg.fut_len)
        entity_count = min(len(entity_ids), max(int(self.cfg.max_entities_per_sample), 1))
        obs_state = np.zeros((int(self.cfg.obs_len), entity_count, STATE_DIM), dtype=np.float32)
        fut_state = np.zeros((int(self.cfg.fut_len), entity_count, STATE_DIM), dtype=np.float32)
        obs_valid = np.zeros((int(self.cfg.obs_len), entity_count), dtype=bool)
        fut_valid = np.zeros((int(self.cfg.fut_len), entity_count), dtype=bool)
        semantic_features = np.zeros((entity_count, SEMANTIC_FEATURE_DIM), dtype=np.float32)
        semantic_boxes = np.zeros((entity_count, 4), dtype=np.float32)
        semantic_mask_flags = np.ones((entity_count,), dtype=bool)
        semantic_rgb_crop = np.zeros((entity_count, 3, crop_size, crop_size), dtype=np.float32)
        semantic_mask_crop = np.zeros((entity_count, 1, crop_size, crop_size), dtype=np.float32)
        semantic_crop_valid = np.ones((entity_count,), dtype=bool)
        semantic_mask_crop_valid = np.zeros((entity_count,), dtype=bool)
        semantic_rgb_crop_temporal = np.zeros((entity_count, temporal_window, 3, crop_size, crop_size), dtype=np.float32)
        semantic_mask_crop_temporal = np.zeros((entity_count, temporal_window, 1, crop_size, crop_size), dtype=np.float32)
        semantic_temporal_valid = np.zeros((entity_count, temporal_window), dtype=bool)
        semantic_instance_id_crop = np.zeros((entity_count, 1, crop_size, crop_size), dtype=np.int64)
        semantic_instance_id_temporal = np.zeros((entity_count, temporal_window, 1, crop_size, crop_size), dtype=np.int64)
        semantic_instance_valid = np.zeros((entity_count, temporal_window), dtype=bool)
        semantic_objectness_score = np.zeros((entity_count,), dtype=np.float32)
        entity_boxes_over_time = np.zeros((total_steps, entity_count, 4), dtype=np.float32)
        entity_masks_over_time: List[List[np.ndarray | None]] = []
        semantic_frame_path = frame_paths[min(max(semantic_step, 0), max(len(frame_paths) - 1, 0))]
        semantic_mask_path = mask_paths[min(max(semantic_step, 0), max(len(mask_paths) - 1, 0))] if mask_paths else ""
        semantic_instance_id_map = raw_id_maps[min(max(semantic_step, 0), max(len(raw_id_maps) - 1, 0))] if raw_id_maps else None

        for ent_idx, entity_id in enumerate(entity_ids[:entity_count]):
            boxes: List[np.ndarray] = []
            present_flags: List[bool] = []
            mask_seq: List[np.ndarray | None] = []
            last_box: np.ndarray | None = None
            for step_i, src_i in enumerate(indices):
                frame_path = frame_paths[src_i]
                raw_id_map = raw_id_maps[step_i]
                mask = _entity_mask_for_dataset(
                    dataset_name=dataset_name,
                    raw_id_map=raw_id_map,
                    entity_id=int(entity_id),
                )
                with Image.open(frame_path) as img_obj:
                    img = img_obj.convert("RGB")
                    w, h = img.size
                    rgb_arr = np.asarray(img, dtype=np.float32) / 255.0
                if mask is not None and np.any(mask > 0):
                    box, used_mask, fg_ratio = _box_from_mask_or_center(
                        mask=mask,
                        width=w,
                        height=h,
                        radius=int(self.cfg.semantic_patch_radius),
                    )
                    last_box = box
                    present = True
                else:
                    box = np.asarray(last_box, dtype=np.float32) if isinstance(last_box, np.ndarray) else _box_from_mask_or_center(None, width=w, height=h, radius=int(self.cfg.semantic_patch_radius))[0]
                    used_mask = False
                    fg_ratio = 0.0
                    present = False
                boxes.append(np.asarray(box, dtype=np.float32))
                present_flags.append(bool(present))
                mask_seq.append(mask.copy() if isinstance(mask, np.ndarray) else None)
                entity_boxes_over_time[step_i, ent_idx] = np.asarray(box, dtype=np.float32)

                if step_i == semantic_step:
                    semantic_boxes[ent_idx] = np.asarray(box, dtype=np.float32)
                    semantic_features[ent_idx] = _semantic_feature(
                        rgb=rgb_arr,
                        mask=mask,
                        box_xyxy=box,
                        mask_used=bool(used_mask),
                        fg_ratio=float(fg_ratio),
                    )
                    rgb_crop, mask_crop, mask_valid = _build_semantic_crops(
                        rgb=rgb_arr,
                        mask=mask,
                        box_xyxy=box,
                        crop_size=crop_size,
                    )
                    semantic_rgb_crop[ent_idx] = rgb_crop
                    semantic_mask_crop[ent_idx] = mask_crop
                    semantic_mask_crop_valid[ent_idx] = bool(mask_valid)
                    semantic_objectness_score[ent_idx] = float(fg_ratio)
                    semantic_instance_id_crop[ent_idx] = _build_instance_id_crop(
                        raw_id_map,
                        box_xyxy=box,
                        crop_size=crop_size,
                    )
                if step_i < temporal_window:
                    rgb_crop_t, mask_crop_t, mask_valid_t = _build_semantic_crops(
                        rgb=rgb_arr,
                        mask=mask,
                        box_xyxy=box,
                        crop_size=crop_size,
                    )
                    semantic_rgb_crop_temporal[ent_idx, step_i] = rgb_crop_t
                    semantic_mask_crop_temporal[ent_idx, step_i] = mask_crop_t
                    semantic_temporal_valid[ent_idx, step_i] = bool(mask_valid_t and present)
                    semantic_instance_id_temporal[ent_idx, step_i] = _build_instance_id_crop(
                        raw_id_map,
                        box_xyxy=box,
                        crop_size=crop_size,
                    )
                    semantic_instance_valid[ent_idx, step_i] = bool(true_instance_aware and present and int(entity_id) > 0)

            if bool(self.cfg.include_entity_masks_over_time):
                entity_masks_over_time.append(mask_seq)
            state = _build_state_from_boxes(boxes=boxes, sizes=sizes)
            valid_arr = np.asarray(present_flags, dtype=bool)
            obs_state[:, ent_idx, :] = state[: int(self.cfg.obs_len)]
            fut_state[:, ent_idx, :] = state[int(self.cfg.obs_len) :]
            obs_valid[:, ent_idx] = valid_arr[: int(self.cfg.obs_len)]
            fut_valid[:, ent_idx] = valid_arr[int(self.cfg.obs_len) :]

        teacher_prior = None
        teacher_cache_path = self.teacher_cache_index.get(cache_key, "")
        if teacher_cache_path:
            tp = Path(teacher_cache_path)
            if tp.exists():
                try:
                    with np.load(tp, allow_pickle=True) as teacher_payload:
                        teacher_prior = np.asarray(teacher_payload["semantic_teacher_prior"], dtype=np.float32)
                except Exception:
                    teacher_prior = None
        if teacher_prior is None or teacher_prior.shape[0] != entity_count:
            teacher_prior = np.zeros((entity_count, 512), dtype=np.float32)

        sample: Dict[str, Any] = {
            "obs_state": torch.from_numpy(obs_state).to(torch.float32),
            "fut_state": torch.from_numpy(fut_state).to(torch.float32),
            "obs_valid": torch.from_numpy(obs_valid).to(torch.bool),
            "fut_valid": torch.from_numpy(fut_valid).to(torch.bool),
            "point_ids": torch.from_numpy(np.asarray(entity_ids[:entity_count], dtype=np.int64)).to(torch.long),
            "meta": {
                "dataset": str(entry.get("dataset_name", "")),
                "clip_id": str(entry.get("clip_id", "")),
                "annotation_source": str(entry.get("annotation_source", "")),
                "frame_count_total": int(len(frame_paths)),
                "entity_count": int(entity_count),
                "instance_source": str(instance_source),
                "true_instance_aware": bool(true_instance_aware),
            },
            "semantic_features": torch.from_numpy(semantic_features).to(torch.float32),
            "semantic_boxes": torch.from_numpy(semantic_boxes).to(torch.float32),
            "semantic_mask": torch.from_numpy(semantic_mask_flags).to(torch.bool),
            "semantic_rgb_crop": torch.from_numpy(semantic_rgb_crop).to(torch.float32),
            "semantic_mask_crop": torch.from_numpy(semantic_mask_crop).to(torch.float32),
            "semantic_crop_valid": torch.from_numpy(semantic_crop_valid).to(torch.bool),
            "semantic_mask_crop_valid": torch.from_numpy(semantic_mask_crop_valid).to(torch.bool),
            "semantic_rgb_crop_temporal": torch.from_numpy(semantic_rgb_crop_temporal).to(torch.float32),
            "semantic_mask_crop_temporal": torch.from_numpy(semantic_mask_crop_temporal).to(torch.float32),
            "semantic_temporal_valid": torch.from_numpy(semantic_temporal_valid).to(torch.bool),
            "semantic_instance_id_map": torch.from_numpy(
                np.asarray(semantic_instance_id_map, dtype=np.int64)
                if bool(self.cfg.include_full_instance_id_map) and isinstance(semantic_instance_id_map, np.ndarray)
                else np.zeros((1, 1), dtype=np.int64)
            ).to(torch.long),
            "semantic_instance_id_crop": torch.from_numpy(semantic_instance_id_crop).to(torch.long),
            "semantic_instance_id_temporal": torch.from_numpy(semantic_instance_id_temporal).to(torch.long),
            "semantic_instance_valid": torch.from_numpy(semantic_instance_valid).to(torch.bool),
            "semantic_objectness_score": torch.from_numpy(semantic_objectness_score).to(torch.float32),
            "semantic_teacher_prior": torch.from_numpy(teacher_prior).to(torch.float32),
            "entity_boxes_over_time": torch.from_numpy(entity_boxes_over_time).to(torch.float32),
            "entity_masks_over_time": entity_masks_over_time if bool(self.cfg.include_entity_masks_over_time) else [],
            "semantic_frame_path": semantic_frame_path,
            "semantic_mask_path": semantic_mask_path,
            "semantic_source_mode": "object_region_or_mask_crop_visual_state",
            "current_mainline_semantic_source": str(self.cfg.semantic_source_mainline),
            "legacy_semantic_source": "hand_crafted_stats",
            "semantic_source_summary": {
                "mask_crop_used_tokens": int(sum(int(x > 0.0) for x in semantic_objectness_score.tolist())),
                "region_crop_used_tokens": int(entity_count),
                "mask_available": bool(semantic_mask_path),
                "instance_aware_source": str(instance_source),
                "target_instance_ids": [int(x) for x in entity_ids[:entity_count]],
                "semantic_crop_size": int(self.cfg.semantic_crop_size),
                "current_mainline_semantic_source": str(self.cfg.semantic_source_mainline),
                "legacy_semantic_source": "hand_crafted_stats",
                "legacy_semantic_feature_dim": int(SEMANTIC_FEATURE_DIM),
                "entity_count": int(entity_count),
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
    temporal_window = max(int(item["semantic_rgb_crop_temporal"].shape[1]) for item in batch)
    semantic_rgb_crop_temporal = torch.zeros((bsz, max_k, temporal_window, 3, crop_h, crop_w), dtype=torch.float32)
    semantic_mask_crop_temporal = torch.zeros((bsz, max_k, temporal_window, 1, crop_h, crop_w), dtype=torch.float32)
    semantic_temporal_valid = torch.zeros((bsz, max_k, temporal_window), dtype=torch.bool)
    semantic_instance_id_crop = torch.zeros((bsz, max_k, 1, crop_h, crop_w), dtype=torch.long)
    semantic_instance_id_temporal = torch.zeros((bsz, max_k, temporal_window, 1, crop_h, crop_w), dtype=torch.long)
    semantic_instance_valid = torch.zeros((bsz, max_k, temporal_window), dtype=torch.bool)
    semantic_objectness_score = torch.zeros((bsz, max_k), dtype=torch.float32)
    teacher_prior_dim = int(batch[0].get("semantic_teacher_prior", torch.zeros((1, 512))).shape[-1])
    semantic_teacher_prior = torch.zeros((bsz, max_k, teacher_prior_dim), dtype=torch.float32)

    semantic_frame_paths: List[str] = []
    semantic_mask_paths: List[str] = []
    semantic_source_summaries: List[Dict[str, Any]] = []
    semantic_instance_id_map: List[torch.Tensor] = []
    entity_boxes_over_time: List[torch.Tensor] = []
    entity_masks_over_time: List[Any] = []
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
        item_temporal_window = int(item["semantic_rgb_crop_temporal"].shape[1])
        semantic_rgb_crop_temporal[i, :k, :item_temporal_window] = item["semantic_rgb_crop_temporal"]
        semantic_mask_crop_temporal[i, :k, :item_temporal_window] = item["semantic_mask_crop_temporal"]
        semantic_temporal_valid[i, :k, :item_temporal_window] = item["semantic_temporal_valid"]
        semantic_instance_id_crop[i, :k] = item["semantic_instance_id_crop"]
        semantic_instance_id_temporal[i, :k, :item_temporal_window] = item["semantic_instance_id_temporal"]
        semantic_instance_valid[i, :k, :item_temporal_window] = item["semantic_instance_valid"]
        semantic_objectness_score[i, :k] = item["semantic_objectness_score"]
        semantic_teacher_prior[i, :k] = item["semantic_teacher_prior"]

        semantic_frame_paths.append(str(item.get("semantic_frame_path", "")))
        semantic_mask_paths.append(str(item.get("semantic_mask_path", "")))
        semantic_source_summaries.append(dict(item.get("semantic_source_summary", {})))
        semantic_instance_id_map.append(item.get("semantic_instance_id_map"))
        entity_boxes_over_time.append(item.get("entity_boxes_over_time"))
        entity_masks_over_time.append(item.get("entity_masks_over_time"))
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
        "semantic_teacher_prior": semantic_teacher_prior,
        "entity_boxes_over_time": entity_boxes_over_time,
        "entity_masks_over_time": entity_masks_over_time,
        "semantic_frame_paths": semantic_frame_paths,
        "semantic_mask_paths": semantic_mask_paths,
        "semantic_source_mode": "object_region_or_mask_crop_visual_state",
        "current_mainline_semantic_source": str(batch[0].get("current_mainline_semantic_source", "crop_visual_encoder")),
        "legacy_semantic_source": str(batch[0].get("legacy_semantic_source", "hand_crafted_stats")),
        "semantic_source_summaries": semantic_source_summaries,
    }
