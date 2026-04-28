from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import math

import numpy as np
from PIL import Image
import torch

from stwm.tracewm_v2.constants import STATE_DIM


SEMANTIC_FEATURE_DIM = 10


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
        with Image.open(frame_path) as im:
            im = im.convert("RGB")
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
