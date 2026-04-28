from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any
import math

import numpy as np
from PIL import Image
import torch

from stwm.tracewm_v2.constants import STATE_DIM


SEMANTIC_FEATURE_DIM = 10
_IMAGE_CACHE: OrderedDict[str, Image.Image] = OrderedDict()
_IMAGE_CACHE_MAX = 4


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except Exception:
        return None


def _image_size(item: dict[str, Any]) -> tuple[int, int]:
    size = item.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return int(size[0]), int(size[1])
    if isinstance(size, dict):
        width = size.get("width")
        height = size.get("height")
        if width is not None and height is not None:
            return int(width), int(height)
    frames = item.get("frame_paths")
    if isinstance(frames, list) and frames:
        try:
            with Image.open(frames[0]) as im:
                return int(im.width), int(im.height)
        except Exception:
            pass
    return 1, 1


def _center_bbox_state(bbox: list[float] | None, image_w: int, image_h: int) -> torch.Tensor:
    state = torch.zeros((STATE_DIM,), dtype=torch.float32)
    if bbox is None or image_w <= 0 or image_h <= 0:
        return state
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) * 0.5) / float(image_w)
    cy = ((y1 + y2) * 0.5) / float(image_h)
    rw = max((x2 - x1) / float(image_w), 0.0)
    rh = max((y2 - y1) / float(image_h), 0.0)
    state[0] = float(max(0.0, min(1.0, cx)))
    state[1] = float(max(0.0, min(1.0, cy)))
    state[2] = 0.0
    state[3] = 1.0
    state[4] = 0.0
    state[5] = 0.0
    state[6] = float(max(0.0, min(1.0, rw)))
    state[7] = float(max(0.0, min(1.0, rh)))
    return state


def _crop_rgb(frame_path: str, bbox: list[float] | None, crop_size: int) -> torch.Tensor:
    out = torch.zeros((3, int(crop_size), int(crop_size)), dtype=torch.float32)
    if not frame_path or bbox is None:
        return out
    try:
        im = _load_rgb_image_cached(frame_path)
        w, h = im.size
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(x1 + 1, min(int(round(x2)), w))
        y2 = max(y1 + 1, min(int(round(y2)), h))
        crop = im.crop((x1, y1, x2, y2)).resize((int(crop_size), int(crop_size)), Image.BILINEAR)
        arr = np.asarray(crop, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    except Exception:
        return out


def _load_rgb_image_cached(frame_path: str) -> Image.Image:
    cached = _IMAGE_CACHE.get(frame_path)
    if cached is not None:
        _IMAGE_CACHE.move_to_end(frame_path)
        return cached
    with Image.open(frame_path) as im:
        rgb = im.convert("RGB").copy()
    _IMAGE_CACHE[frame_path] = rgb
    _IMAGE_CACHE.move_to_end(frame_path)
    while len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
        _IMAGE_CACHE.popitem(last=False)
    return rgb


def _semantic_features_from_bbox(bbox: list[float] | None, image_w: int, image_h: int) -> torch.Tensor:
    feat = torch.zeros((SEMANTIC_FEATURE_DIM,), dtype=torch.float32)
    if bbox is None:
        return feat
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) * 0.5) / max(float(image_w), 1.0)
    cy = ((y1 + y2) * 0.5) / max(float(image_h), 1.0)
    bw = max((x2 - x1) / max(float(image_w), 1.0), 0.0)
    bh = max((y2 - y1) / max(float(image_h), 1.0), 0.0)
    area = bw * bh
    feat[:10] = torch.tensor([cx, cy, bw, bh, area, math.sqrt(max(area, 0.0)), 1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    return feat


def _frame_path_for_index(frame_paths: list[Any], frame_index: Any) -> str | None:
    if not frame_paths:
        return None
    try:
        idx = int(frame_index)
    except Exception:
        idx = len(frame_paths) - 1
    idx = max(0, min(idx, len(frame_paths) - 1))
    return str(frame_paths[idx])


def build_candidate_measurement_features(
    item: dict[str, Any],
    candidate: dict[str, Any],
    frame_paths: list[Any] | None = None,
    future_frame_index: Any | None = None,
    crop_size: int = 64,
) -> dict[str, Any]:
    """Build deterministic candidate measurement features for scoring only.

    The returned feature vector is intentionally kept outside the model batch:
    it is a posterior measurement likelihood term, not rollout input.
    """

    paths = frame_paths if isinstance(frame_paths, list) else item.get("frame_paths")
    paths = paths if isinstance(paths, list) else []
    frame_idx = item.get("future_frame_index") if future_frame_index is None else future_frame_index
    frame_path = _frame_path_for_index(paths, frame_idx)
    image_w, image_h = _image_size(item)
    bbox = _bbox(candidate.get("bbox") if isinstance(candidate, dict) else None)
    if bbox is None:
        return {
            "feature_source": "missing_candidate_bbox",
            "feature_vector": [0.0] * 18,
            "future_frame_path": frame_path,
            "candidate_feature_used_for_scoring": True,
            "candidate_feature_used_for_rollout": False,
        }
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) * 0.5) / max(float(image_w), 1.0)
    cy = ((y1 + y2) * 0.5) / max(float(image_h), 1.0)
    bw = max((x2 - x1) / max(float(image_w), 1.0), 0.0)
    bh = max((y2 - y1) / max(float(image_h), 1.0), 0.0)
    area = bw * bh
    aspect = bw / max(bh, 1e-6)
    mean_rgb = [0.0, 0.0, 0.0]
    std_rgb = [0.0, 0.0, 0.0]
    feature_source = "bbox_stats_only"
    if frame_path:
        crop = _crop_rgb(frame_path, bbox, int(crop_size))
        if torch.isfinite(crop).all() and float(crop.abs().sum().item()) > 0.0:
            flat = crop.view(3, -1)
            mean_rgb = [float(x) for x in flat.mean(dim=1).tolist()]
            std_rgb = [float(x) for x in flat.std(dim=1, unbiased=False).tolist()]
            feature_source = "weak_rgb_bbox_stats"
    mask_area = 0.0
    rle = candidate.get("mask_rle") if isinstance(candidate, dict) else None
    if isinstance(rle, dict) and isinstance(rle.get("size"), list) and len(rle.get("size")) >= 2:
        # RLE decoding is intentionally avoided here; bbox area is the safe
        # deterministic proxy used for this V1 measurement feature.
        mask_area = area
    feat = [
        float(max(0.0, min(1.0, cx))),
        float(max(0.0, min(1.0, cy))),
        float(max(0.0, min(1.0, bw))),
        float(max(0.0, min(1.0, bh))),
        float(max(0.0, min(1.0, area))),
        float(max(0.0, min(10.0, aspect)) / 10.0),
        float(max(0.0, min(1.0, math.sqrt(max(area, 0.0))))),
        float(max(0.0, min(1.0, mask_area))),
        *mean_rgb,
        *std_rgb,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    return {
        "feature_source": feature_source,
        "feature_vector": [float(x) for x in feat],
        "future_frame_path": frame_path,
        "bbox_stats": {"cx": feat[0], "cy": feat[1], "w": feat[2], "h": feat[3], "area": feat[4], "aspect_scaled": feat[5]},
        "mean_rgb": mean_rgb,
        "std_rgb": std_rgb,
        "candidate_feature_used_for_scoring": True,
        "candidate_feature_used_for_rollout": False,
    }


def _feature_stats_vector(candidate: dict[str, Any], item: dict[str, Any], frame_path: str | None, crop_size: int) -> tuple[list[float], dict[str, Any]]:
    image_w, image_h = _image_size(item)
    bbox = _bbox(candidate.get("bbox") if isinstance(candidate, dict) else None)
    if bbox is None:
        return [0.0] * 18, {"feature_source": "missing_candidate_bbox", "bbox_used": False, "mask_used": False}
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) * 0.5) / max(float(image_w), 1.0)
    cy = ((y1 + y2) * 0.5) / max(float(image_h), 1.0)
    bw = max((x2 - x1) / max(float(image_w), 1.0), 0.0)
    bh = max((y2 - y1) / max(float(image_h), 1.0), 0.0)
    area = bw * bh
    aspect = bw / max(bh, 1e-6)
    mean_rgb = [0.0, 0.0, 0.0]
    std_rgb = [0.0, 0.0, 0.0]
    source = "bbox_stats_only"
    if frame_path:
        crop = _crop_rgb(frame_path, bbox, int(crop_size))
        if torch.isfinite(crop).all() and float(crop.abs().sum().item()) > 0.0:
            flat = crop.view(3, -1)
            mean_rgb = [float(x) for x in flat.mean(dim=1).tolist()]
            std_rgb = [float(x) for x in flat.std(dim=1, unbiased=False).tolist()]
            source = "weak_rgb_bbox_stats"
    mask_used = isinstance(candidate.get("mask_rle") if isinstance(candidate, dict) else None, dict) or bool(candidate.get("mask_path") if isinstance(candidate, dict) else None)
    vec = [
        float(max(0.0, min(1.0, cx))),
        float(max(0.0, min(1.0, cy))),
        float(max(0.0, min(1.0, bw))),
        float(max(0.0, min(1.0, bh))),
        float(max(0.0, min(1.0, area))),
        float(max(0.0, min(10.0, aspect)) / 10.0),
        float(max(0.0, min(1.0, math.sqrt(max(area, 0.0))))),
        float(max(0.0, min(1.0, area if mask_used else 0.0))),
        *mean_rgb,
        *std_rgb,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    return [float(x) for x in vec], {"feature_source": source, "bbox_used": True, "mask_used": bool(mask_used)}


def build_item_candidate_measurement_cache(
    item: dict[str, Any],
    *,
    semantic_encoder: Any | None = None,
    device: Any | None = None,
    crop_size: int = 64,
    feature_mode: str = "crop_encoder_feature",
) -> dict[str, Any]:
    """Build candidate measurement features for one item.

    Candidate features are posterior measurement observations only. They are
    deliberately not returned in the model batch and must not be fed to rollout.
    """

    frame_paths = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
    future_frame_path = _frame_path_for_index(frame_paths, item.get("future_frame_index"))
    observed_indices = item.get("observed_frame_indices") if isinstance(item.get("observed_frame_indices"), list) else [0]
    observed_frame_path = _frame_path_for_index(frame_paths, observed_indices[0] if observed_indices else 0)
    observed_target = item.get("observed_target") if isinstance(item.get("observed_target"), dict) else {}
    candidates = item.get("future_candidates") if isinstance(item.get("future_candidates"), list) else []
    crop_items: list[tuple[str, dict[str, Any], str | None]] = [("__observed_target__", observed_target, observed_frame_path)]
    for cand in candidates:
        if isinstance(cand, dict):
            crop_items.append((str(cand.get("candidate_id")), cand, future_frame_path))

    stats: dict[str, list[float]] = {}
    meta: dict[str, dict[str, Any]] = {}
    rgb_crops = []
    mask_crops = []
    for cid, obj, frame_path in crop_items:
        vec, info = _feature_stats_vector(obj, item, frame_path, int(crop_size))
        stats[cid] = vec
        meta[cid] = info
        bbox = _bbox(obj.get("bbox") if isinstance(obj, dict) else None)
        rgb_crops.append(_crop_rgb(frame_path or "", bbox, int(crop_size)))
        mask_crops.append(torch.ones((1, int(crop_size), int(crop_size)), dtype=torch.float32))

    encoded: dict[str, list[float]] = {}
    feature_source = "weak_rgb_bbox_stats"
    if str(feature_mode) in {"crop_encoder_feature", "hybrid_crop_bbox_feature"} and semantic_encoder is not None:
        try:
            rgb = torch.stack(rgb_crops, dim=0).unsqueeze(0)
            mask = torch.stack(mask_crops, dim=0).unsqueeze(0)
            if device is not None:
                rgb = rgb.to(device)
                mask = mask.to(device)
            with torch.no_grad():
                token = semantic_encoder(
                    semantic_rgb_crop=rgb,
                    semantic_mask_crop=mask,
                    source_mode="crop_visual_encoder",
                )[0].detach().float().cpu()
            for idx, (cid, _, _) in enumerate(crop_items):
                encoded[cid] = [float(x) for x in token[idx].tolist()]
            feature_source = "crop_encoder_feature" if str(feature_mode) == "crop_encoder_feature" else "hybrid_crop_bbox_feature"
        except Exception as exc:
            feature_source = f"crop_encoder_blocked_fallback_weak_rgb_bbox_stats:{type(exc).__name__}"

    observed_feature = encoded.get("__observed_target__", stats.get("__observed_target__", []))
    features: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        cid = str(cand.get("candidate_id"))
        crop_vec = encoded.get(cid)
        stats_vec = stats.get(cid, [0.0] * 18)
        if str(feature_mode) == "hybrid_crop_bbox_feature" and crop_vec is not None:
            vec = [float(x) for x in crop_vec] + [float(x) for x in stats_vec]
        elif crop_vec is not None:
            vec = [float(x) for x in crop_vec]
        else:
            vec = [float(x) for x in stats_vec]
        features[cid] = {
            "feature_source": feature_source if crop_vec is not None else meta.get(cid, {}).get("feature_source", "weak_rgb_bbox_stats"),
            "feature_vector": vec,
            "observed_target_feature_vector": observed_feature,
            "feature_dim": len(vec),
            "observed_target_feature_available": bool(observed_feature),
            "future_candidate_feature_available": bool(vec),
            "mask_used": bool(meta.get(cid, {}).get("mask_used", False)),
            "bbox_used": bool(meta.get(cid, {}).get("bbox_used", False)),
            "candidate_feature_used_for_scoring": True,
            "candidate_feature_used_for_rollout": False,
            "future_candidate_used_for_scoring": True,
            "future_candidate_used_for_rollout": False,
        }
    return {
        "feature_mode": str(feature_mode),
        "candidate_feature_source": feature_source,
        "observed_target_feature_available": bool(observed_feature),
        "candidate_features": features,
        "future_candidate_used_for_rollout": False,
        "future_candidate_used_for_scoring": True,
    }


def group_candidate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        grouped[str(rec.get("item_id"))].append(rec)
    items: list[dict[str, Any]] = []
    for item_id, recs in grouped.items():
        first = recs[0]
        candidates = []
        labels = []
        for rec in recs:
            candidates.append(rec.get("candidate"))
            labels.append(int(rec.get("label_same_identity", 0)))
        items.append(
            {
                "item_id": item_id,
                "records": recs,
                "source_dataset": first.get("source_dataset"),
                "video_id": first.get("video_id"),
                "frame_paths": first.get("frame_paths"),
                "observed_frame_indices": first.get("observed_frame_indices"),
                "future_frame_index": first.get("future_frame_index"),
                "observed_target": first.get("observed_target"),
                "future_candidates": candidates,
                "candidate_labels": labels,
                "subset_tags": first.get("subset_tags") if isinstance(first.get("subset_tags"), dict) else {},
                "image_size": first.get("image_size"),
            }
        )
    return items


def build_query_batch_from_item(
    item: dict[str, Any],
    *,
    obs_len: int,
    fut_len: int,
    crop_size: int = 64,
    semantic_temporal_window: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frame_paths = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
    if not frame_paths:
        raise RuntimeError("missing_frame_paths")
    observed_target = item.get("observed_target") if isinstance(item.get("observed_target"), dict) else {}
    bbox = _bbox(observed_target.get("bbox"))
    if bbox is None:
        raise RuntimeError("observed_target_bbox_required_for_query_batch_v1")
    image_w, image_h = _image_size(item)
    state = _center_bbox_state(bbox, image_w=image_w, image_h=image_h)
    obs_state = torch.zeros((1, int(obs_len), 1, STATE_DIM), dtype=torch.float32)
    obs_state[:, :, 0] = state
    fut_state = torch.zeros((1, int(fut_len), 1, STATE_DIM), dtype=torch.float32)
    obs_valid = torch.zeros((1, int(obs_len), 1), dtype=torch.bool)
    observed_indices = item.get("observed_frame_indices") if isinstance(item.get("observed_frame_indices"), list) else [0]
    # External hard-case items usually expose only one observed prompt. Repeat
    # the state for model context, but mark only observed frames as supervised.
    obs_valid[:, : max(1, min(len(observed_indices), int(obs_len))), 0] = True
    fut_valid = torch.zeros((1, int(fut_len), 1), dtype=torch.bool)
    token_mask = torch.ones((1, 1), dtype=torch.bool)
    semantic_features = _semantic_features_from_bbox(bbox, image_w, image_h).view(1, 1, -1)
    first_frame_idx = int(observed_indices[0]) if observed_indices else 0
    first_frame_idx = max(0, min(first_frame_idx, len(frame_paths) - 1))
    rgb = _crop_rgb(str(frame_paths[first_frame_idx]), bbox, crop_size).view(1, 1, 3, int(crop_size), int(crop_size))
    mask_crop = torch.ones((1, 1, 1, int(crop_size), int(crop_size)), dtype=torch.float32)
    rgb_temporal = rgb[:, :, None].repeat(1, 1, int(semantic_temporal_window), 1, 1, 1)
    mask_temporal = mask_crop[:, :, None].repeat(1, 1, int(semantic_temporal_window), 1, 1, 1)
    temporal_valid = torch.zeros((1, 1, int(semantic_temporal_window)), dtype=torch.bool)
    temporal_valid[:, :, 0] = True
    batch = {
        "batch_size": 1,
        "obs_state": obs_state,
        "fut_state": fut_state,
        "obs_valid": obs_valid,
        "fut_valid": fut_valid,
        "token_mask": token_mask,
        "point_ids": torch.zeros((1, 1), dtype=torch.long),
        "meta": [{"dataset": item.get("source_dataset"), "clip_id": item.get("item_id"), "external_hardcase_query": True}],
        "semantic_features": semantic_features,
        "semantic_boxes": torch.tensor([[[bbox[0], bbox[1], bbox[2], bbox[3]]]], dtype=torch.float32),
        "semantic_mask": torch.ones((1, 1), dtype=torch.bool),
        "semantic_rgb_crop": rgb,
        "semantic_mask_crop": mask_crop,
        "semantic_crop_valid": torch.ones((1, 1), dtype=torch.bool),
        "semantic_mask_crop_valid": torch.ones((1, 1), dtype=torch.bool),
        "semantic_rgb_crop_temporal": rgb_temporal,
        "semantic_mask_crop_temporal": mask_temporal,
        "semantic_temporal_valid": temporal_valid,
        "semantic_instance_id_map": [torch.zeros((1, 1), dtype=torch.long)],
        "semantic_instance_id_crop": torch.zeros((1, 1, 1, int(crop_size), int(crop_size)), dtype=torch.long),
        "semantic_instance_id_temporal": torch.zeros((1, 1, int(semantic_temporal_window), 1, int(crop_size), int(crop_size)), dtype=torch.long),
        "semantic_instance_valid": temporal_valid.clone(),
        "semantic_objectness_score": torch.ones((1, 1), dtype=torch.float32),
        "semantic_entity_dominant_instance_id": torch.zeros((1, 1), dtype=torch.long),
        "semantic_entity_instance_overlap_score_over_time": torch.zeros((1, 1, int(semantic_temporal_window)), dtype=torch.float32),
        "semantic_entity_true_instance_confidence": torch.ones((1, 1), dtype=torch.float32),
        "semantic_teacher_prior": torch.zeros((1, 1, 512), dtype=torch.float32),
        "entity_boxes_over_time": [torch.zeros((int(obs_len) + int(fut_len), 1, 4), dtype=torch.float32)],
        "entity_masks_over_time": [[]],
        "semantic_frame_paths": [str(frame_paths[first_frame_idx])],
        "semantic_mask_paths": [""],
        "semantic_source_mode": "external_observed_target_crop",
        "current_mainline_semantic_source": "crop_visual_encoder",
        "legacy_semantic_source": "hand_crafted_stats",
        "semantic_source_summaries": [{"semantic_feature_source": "observed_target_crop_and_bbox_stats"}],
    }
    meta = {
        "future_candidate_used_as_input": False,
        "candidate_bbox_used_for_rollout_input": False,
        "candidate_bbox_used_for_eval_scoring": True,
        "observed_target_used_for_input": True,
        "future_target_leakage": False,
        "padded_obs_repeated": True,
        "semantic_feature_source": "observed_target_crop_and_bbox_stats",
        "query_slot_count": 1,
    }
    return batch, meta
