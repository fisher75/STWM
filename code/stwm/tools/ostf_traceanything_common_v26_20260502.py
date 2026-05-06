#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tools.ostf_v17_common_20260502 import (
    ROOT,
    assign_semantic_prototypes,
    dump_json,
    kmeans_semantic_prototypes,
    set_seed,
    write_doc,
)
from stwm.tools.ostf_v18_common_20260502 import (
    analytic_affine_motion_predict,
    analytic_constant_velocity_predict,
    semantic_logits_from_observed_memory,
)


V25_CACHE_ROOT = ROOT / "outputs/cache/stwm_traceanything_hardbench_v25"
V25_DECISION_PATH = ROOT / "reports/stwm_ostf_v25_traceanything_hardbench_decision_20260502.json"
V24_BRIDGE_PATH = ROOT / "reports/stwm_ostf_semantic_identity_bridge_hardbench_v24_20260502.json"
HARDBENCH_PATH = ROOT / "reports/stwm_ostf_hard_benchmark_v2_20260502.json"


def _scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _object_rel(points_xy: np.ndarray) -> np.ndarray:
    pts = points_xy.astype(np.float32)
    mn = pts.min(axis=0, keepdims=True)
    mx = pts.max(axis=0, keepdims=True)
    return ((pts - mn) / np.maximum(mx - mn, 1.0)).astype(np.float32)


def _anchor_from_points(points: np.ndarray, vis: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    weight = vis.astype(np.float32)
    if conf is not None:
        weight = weight * np.clip(conf.astype(np.float32), 0.0, 4.0)
    denom = np.maximum(weight.sum(axis=0), 1.0)
    return ((points * weight[..., None]).sum(axis=0) / denom[:, None]).astype(np.float32)


def _velocity(xy_t_2: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(xy_t_2, dtype=np.float32)
    if xy_t_2.shape[0] >= 2:
        vel[1:] = xy_t_2[1:] - xy_t_2[:-1]
    return vel


def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.maximum((x * x).sum(axis=axis), 0.0))


def _load_hardbench_reason_tags() -> dict[str, list[str]]:
    if not HARDBENCH_PATH.exists():
        return {}
    payload = json.loads(HARDBENCH_PATH.read_text(encoding="utf-8"))
    return {str(k): list(v.get("reason_tags", [])) for k, v in payload.get("per_item", {}).items()}


def _reason_flags(reason_tags: list[str]) -> dict[str, bool]:
    tags = set(reason_tags)
    return {
        "top20_cv_hard": ("top20_cv_hard" in tags) or ("top10_cv_hard" in tags),
        "top30_cv_hard": ("top30_cv_hard" in tags) or ("top20_cv_hard" in tags) or ("top10_cv_hard" in tags),
        "occlusion_hard": "occlusion_hard" in tags,
        "nonlinear_hard": "nonlinear_hard" in tags,
        "interaction_hard": "interaction_hard" in tags,
    }


def _observed_box_features(pre: np.lib.npyio.NpzFile, object_index: int, obs_len: int) -> np.ndarray:
    if "entity_boxes_over_time" not in pre.files:
        return np.zeros((14,), dtype=np.float32)
    boxes = np.asarray(pre["entity_boxes_over_time"], dtype=np.float32)
    if object_index >= boxes.shape[1]:
        return np.zeros((14,), dtype=np.float32)
    obs = boxes[:obs_len, object_index]
    x0, y0, x1, y1 = [obs[:, i] for i in range(4)]
    w = np.maximum(x1 - x0, 1.0)
    h = np.maximum(y1 - y0, 1.0)
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    area = w * h
    aspect = w / np.maximum(h, 1.0)
    dcx = np.diff(cx, prepend=cx[:1])
    dcy = np.diff(cy, prepend=cy[:1])
    dlog_area = np.diff(np.log(np.maximum(area, 1.0)), prepend=np.log(np.maximum(area[:1], 1.0)))
    feats = np.asarray(
        [
            cx[-1],
            cy[-1],
            w[-1],
            h[-1],
            area.mean(),
            area.std(),
            aspect.mean(),
            aspect.std(),
            dcx.mean(),
            dcy.mean(),
            dcx.std(),
            dcy.std(),
            dlog_area.mean(),
            dlog_area.std(),
        ],
        dtype=np.float32,
    )
    scale = max(float(np.abs(feats).max()), 1.0)
    return feats / scale


def _neighbor_features(tracks: np.ndarray, vis: np.ndarray, object_index: int, obs_len: int) -> np.ndarray:
    if object_index >= tracks.shape[0]:
        return np.zeros((10,), dtype=np.float32)
    mask = vis[object_index, :, :obs_len]
    if not np.any(mask):
        return np.zeros((10,), dtype=np.float32)
    denom = np.maximum(mask.sum(axis=0), 1.0)
    cur_anchor = (tracks[object_index, :, :obs_len] * mask[..., None]).sum(axis=0) / denom[:, None]
    cur_vel = np.zeros_like(cur_anchor)
    if cur_anchor.shape[0] > 1:
        cur_vel[1:] = cur_anchor[1:] - cur_anchor[:-1]
    other = []
    for j in range(tracks.shape[0]):
        if j == object_index:
            continue
        omask = vis[j, :, :obs_len]
        if not np.any(omask):
            continue
        oden = np.maximum(omask.sum(axis=0), 1.0)
        anchor = (tracks[j, :, :obs_len] * omask[..., None]).sum(axis=0) / oden[:, None]
        vel = np.zeros_like(anchor)
        if anchor.shape[0] > 1:
            vel[1:] = anchor[1:] - anchor[:-1]
        other.append((anchor[-1], vel[-1]))
    if not other:
        return np.zeros((10,), dtype=np.float32)
    pos = np.stack([x[0] for x in other], axis=0)
    vel = np.stack([x[1] for x in other], axis=0)
    rel = pos - cur_anchor[-1][None]
    dist = _safe_norm(rel, axis=-1)
    near_idx = int(dist.argmin())
    near_rel = rel[near_idx]
    near_vel = vel[near_idx] - cur_vel[-1]
    feats = np.asarray(
        [
            dist.min(),
            dist.mean(),
            float((dist < 0.05).mean()),
            float((dist < 0.10).mean()),
            near_rel[0],
            near_rel[1],
            near_vel[0],
            near_vel[1],
            _safe_norm(vel, axis=-1).mean(),
            _safe_norm(vel, axis=-1).std(),
        ],
        dtype=np.float32,
    )
    scale = max(float(np.abs(feats).max()), 1.0)
    return feats / scale


def _global_motion_features(tracks: np.ndarray, vis: np.ndarray, obs_len: int) -> np.ndarray:
    anchors = []
    vels = []
    for j in range(tracks.shape[0]):
        mask = vis[j, :, :obs_len]
        if not np.any(mask):
            continue
        denom = np.maximum(mask.sum(axis=0), 1.0)
        anchor = (tracks[j, :, :obs_len] * mask[..., None]).sum(axis=0) / denom[:, None]
        vel = np.zeros_like(anchor)
        if anchor.shape[0] > 1:
            vel[1:] = anchor[1:] - anchor[:-1]
        anchors.append(anchor[-1])
        vels.append(vel[-1])
    if not anchors:
        return np.zeros((8,), dtype=np.float32)
    anchors_np = np.stack(anchors, axis=0)
    vels_np = np.stack(vels, axis=0)
    feats = np.asarray(
        [
            np.median(vels_np[:, 0]),
            np.median(vels_np[:, 1]),
            vels_np[:, 0].std(),
            vels_np[:, 1].std(),
            _safe_norm(vels_np, axis=-1).mean(),
            _safe_norm(vels_np, axis=-1).std(),
            anchors_np[:, 0].std(),
            anchors_np[:, 1].std(),
        ],
        dtype=np.float32,
    )
    scale = max(float(np.abs(feats).max()), 1.0)
    return feats / scale


def _semantic_token_from_obs_state(pre: np.lib.npyio.NpzFile) -> np.ndarray:
    if "obs_state" not in pre.files:
        return np.zeros((8,), dtype=np.float32)
    obs_state = np.asarray(pre["obs_state"], dtype=np.float32)
    if obs_state.ndim == 3:
        token = obs_state.mean(axis=(0, 1))
    elif obs_state.ndim == 2:
        token = obs_state.mean(axis=0)
    else:
        token = obs_state.reshape(-1)[:8]
    token = np.asarray(token, dtype=np.float32).reshape(-1)
    if token.size < 8:
        token = np.pad(token, (0, 8 - token.size))
    return token[:8]


def _reappearance_flag(fut_vis: np.ndarray) -> float:
    vis_any = fut_vis.any(axis=0)
    dropped = False
    for flag in vis_any:
        if not flag:
            dropped = True
        elif dropped and flag:
            return 1.0
    return 0.0


@dataclass
class TraceAnythingOSTFSample:
    item_key: str
    dataset: str
    split: str
    source_cache_path: str
    frame_paths: list[str]
    object_index: int
    object_id: int
    semantic_id: int
    m: int
    h: int
    obs_points: np.ndarray
    fut_points: np.ndarray
    obs_vis: np.ndarray
    fut_vis: np.ndarray
    obs_conf: np.ndarray
    fut_conf: np.ndarray
    rel_xy: np.ndarray
    anchor_obs: np.ndarray
    anchor_fut: np.ndarray
    anchor_obs_vel: np.ndarray
    semantic_feat: np.ndarray
    semantic_valid: bool
    proto_target: int
    teacher_source: str
    box_feat: np.ndarray
    neighbor_feat: np.ndarray
    global_feat: np.ndarray
    tusb_token: np.ndarray
    reason_tags: tuple[str, ...]
    subset_flags: dict[str, bool]
    hardness_score: float
    occlusion_ratio: float
    reappearance_flag: float


def load_traceanything_v25_samples(
    combo: str,
    *,
    limit_per_split: dict[str, int] | None = None,
) -> dict[str, list[TraceAnythingOSTFSample]]:
    combo_root = V25_CACHE_ROOT / combo
    out: dict[str, list[TraceAnythingOSTFSample]] = {"train": [], "val": [], "test": []}
    if not combo_root.exists():
        return out
    reason_map = _load_hardbench_reason_tags()
    for cache_path in sorted(combo_root.glob("*/*.npz")):
        z = np.load(cache_path, allow_pickle=True)
        split = str(_scalar(z["split"]))
        if split not in out:
            continue
        if limit_per_split and len(out[split]) >= limit_per_split.get(split, 10**12):
            continue
        item_key = str(_scalar(z["item_key"]))
        tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
        vis = np.asarray(z["visibility"]).astype(bool)
        conf = np.asarray(z["confidence"], dtype=np.float32)
        query = np.asarray(z["query_points_xy"], dtype=np.float32)
        object_ids = np.asarray(z["object_id"], dtype=np.int64)
        semantic_ids = np.asarray(z["semantic_id"], dtype=np.int64)
        raw_size = np.asarray(z["raw_size"], dtype=np.float32)
        scale = float(max(raw_size.tolist()))
        obs_len = int(_scalar(z["obs_len"]))
        horizon = int(_scalar(z["horizon"]))
        frame_paths = [str(x) for x in np.asarray(z["frame_paths"], dtype=object).tolist()]
        teacher_source = str(_scalar(z["teacher_source"]))
        pre = np.load(str(_scalar(z["predecode_path"])), allow_pickle=True)
        sem_feat_all = np.asarray(pre["semantic_features"], dtype=np.float32)
        reason_tags = tuple(reason_map.get(item_key, []))
        subset_flags = _reason_flags(list(reason_tags))
        tusb_token = _semantic_token_from_obs_state(pre)
        for obj in range(tracks.shape[0]):
            sem_idx = obj if obj < sem_feat_all.shape[0] else max(sem_feat_all.shape[0] - 1, 0)
            points = (tracks[obj] / scale).astype(np.float32)
            visibility = vis[obj]
            confidence = np.clip(conf[obj] / 4.0, 0.0, 1.0).astype(np.float32)
            anchor_obs = _anchor_from_points(points[:, :obs_len], visibility[:, :obs_len], confidence[:, :obs_len])
            anchor_fut = _anchor_from_points(points[:, obs_len : obs_len + horizon], visibility[:, obs_len : obs_len + horizon], confidence[:, obs_len : obs_len + horizon])
            box_feat = _observed_box_features(pre, obj, obs_len)
            neighbor_feat = _neighbor_features(points[None] if points.ndim == 2 else tracks / scale, visibility[None] if visibility.ndim == 1 else vis, obj, obs_len)
            # Recompute using full scene for better neighbor/global context.
            neighbor_feat = _neighbor_features((tracks / scale).astype(np.float32), vis, obj, obs_len)
            global_feat = _global_motion_features((tracks / scale).astype(np.float32), vis, obs_len)
            occlusion_ratio = float(1.0 - visibility[:, obs_len : obs_len + horizon].mean())
            hardness_score = float(
                1.0
                + 1.0 * subset_flags["top20_cv_hard"]
                + 0.75 * subset_flags["top30_cv_hard"]
                + 0.50 * subset_flags["occlusion_hard"]
                + 0.35 * subset_flags["nonlinear_hard"]
                + 0.25 * subset_flags["interaction_hard"]
                + 0.25 * occlusion_ratio
            )
            out[split].append(
                TraceAnythingOSTFSample(
                    item_key=item_key,
                    dataset=str(_scalar(z["dataset"])),
                    split=split,
                    source_cache_path=str(cache_path.relative_to(ROOT)),
                    frame_paths=frame_paths,
                    object_index=int(obj),
                    object_id=int(object_ids[obj]) if obj < object_ids.shape[0] else int(obj),
                    semantic_id=int(semantic_ids[obj]) if obj < semantic_ids.shape[0] else -1,
                    m=int(points.shape[0]),
                    h=horizon,
                    obs_points=points[:, :obs_len].astype(np.float32),
                    fut_points=points[:, obs_len : obs_len + horizon].astype(np.float32),
                    obs_vis=visibility[:, :obs_len].astype(bool),
                    fut_vis=visibility[:, obs_len : obs_len + horizon].astype(bool),
                    obs_conf=confidence[:, :obs_len].astype(np.float32),
                    fut_conf=confidence[:, obs_len : obs_len + horizon].astype(np.float32),
                    rel_xy=_object_rel(query[obj]).astype(np.float32),
                    anchor_obs=anchor_obs.astype(np.float32),
                    anchor_fut=anchor_fut.astype(np.float32),
                    anchor_obs_vel=_velocity(anchor_obs.astype(np.float32)),
                    semantic_feat=sem_feat_all[sem_idx].astype(np.float32) if sem_feat_all.size else np.zeros((10,), dtype=np.float32),
                    semantic_valid=bool(sem_feat_all.size and np.linalg.norm(sem_feat_all[sem_idx]) > 1e-6),
                    proto_target=0,
                    teacher_source=teacher_source,
                    box_feat=box_feat.astype(np.float32),
                    neighbor_feat=neighbor_feat.astype(np.float32),
                    global_feat=global_feat.astype(np.float32),
                    tusb_token=tusb_token.astype(np.float32),
                    reason_tags=reason_tags,
                    subset_flags=subset_flags,
                    hardness_score=hardness_score,
                    occlusion_ratio=occlusion_ratio,
                    reappearance_flag=_reappearance_flag(visibility[:, obs_len : obs_len + horizon]),
                )
            )
    return out


def build_v26_rows(combo: str, seed: int = 42) -> tuple[dict[str, list[TraceAnythingOSTFSample]], np.ndarray]:
    set_seed(seed)
    rows = load_traceanything_v25_samples(combo)
    train_samples = rows["train"]
    if not train_samples:
        return rows, np.zeros((32, 10), dtype=np.float32)
    centers = kmeans_semantic_prototypes(train_samples, k=32, iters=20, seed=seed)
    assign_semantic_prototypes(rows, centers)
    return rows, centers


def batch_from_samples_v26(samples: list[TraceAnythingOSTFSample], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "obs_points": torch.tensor(np.stack([s.obs_points for s in samples]), device=device, dtype=torch.float32),
        "fut_points": torch.tensor(np.stack([s.fut_points for s in samples]), device=device, dtype=torch.float32),
        "obs_vis": torch.tensor(np.stack([s.obs_vis for s in samples]), device=device, dtype=torch.bool),
        "fut_vis": torch.tensor(np.stack([s.fut_vis for s in samples]), device=device, dtype=torch.bool),
        "obs_conf": torch.tensor(np.stack([s.obs_conf for s in samples]), device=device, dtype=torch.float32),
        "fut_conf": torch.tensor(np.stack([s.fut_conf for s in samples]), device=device, dtype=torch.float32),
        "rel_xy": torch.tensor(np.stack([s.rel_xy for s in samples]), device=device, dtype=torch.float32),
        "anchor_obs": torch.tensor(np.stack([s.anchor_obs for s in samples]), device=device, dtype=torch.float32),
        "anchor_fut": torch.tensor(np.stack([s.anchor_fut for s in samples]), device=device, dtype=torch.float32),
        "anchor_obs_vel": torch.tensor(np.stack([s.anchor_obs_vel for s in samples]), device=device, dtype=torch.float32),
        "semantic_feat": torch.tensor(np.stack([s.semantic_feat for s in samples]), device=device, dtype=torch.float32),
        "semantic_id": torch.tensor(np.asarray([max(int(s.semantic_id), 0) % 8192 for s in samples]), device=device, dtype=torch.long),
        "proto_target": torch.tensor(np.asarray([int(s.proto_target) for s in samples]), device=device, dtype=torch.long),
        "box_feat": torch.tensor(np.stack([s.box_feat for s in samples]), device=device, dtype=torch.float32),
        "neighbor_feat": torch.tensor(np.stack([s.neighbor_feat for s in samples]), device=device, dtype=torch.float32),
        "global_feat": torch.tensor(np.stack([s.global_feat for s in samples]), device=device, dtype=torch.float32),
        "tusb_token": torch.tensor(np.stack([s.tusb_token for s in samples]), device=device, dtype=torch.float32),
        "hardness_score": torch.tensor(np.asarray([float(s.hardness_score) for s in samples]), device=device, dtype=torch.float32),
        "occlusion_ratio": torch.tensor(np.asarray([float(s.occlusion_ratio) for s in samples]), device=device, dtype=torch.float32),
        "reappearance_flag": torch.tensor(np.asarray([float(s.reappearance_flag) for s in samples]), device=device, dtype=torch.float32),
    }


def analytic_last_observed_copy_predict(
    samples: list[TraceAnythingOSTFSample],
    *,
    proto_count: int = 32,
    proto_centers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_preds = []
    vis_logits = []
    sem_logits = semantic_logits_from_observed_memory(
        samples,
        proto_centers=proto_centers,
        proto_count=proto_count,
    )
    for s in samples:
        last = s.obs_points[:, -1]
        pred = np.repeat(last[:, None, :], s.h, axis=1)
        point_preds.append(pred.astype(np.float32))
        vis_logits.append(np.where(np.repeat(s.obs_vis[:, -1:], s.h, axis=1), 8.0, -8.0).astype(np.float32))
    return np.stack(point_preds), np.stack(vis_logits), sem_logits


def cache_verification_payload() -> dict[str, Any]:
    combos = ["M128_H32", "M512_H32", "M128_H64", "M512_H64"]
    combo_stats: dict[str, Any] = {}
    processed = 0
    valid_ratios = []
    teacher_sources = set()
    model_input_observed_only = True
    target_side_box_search_used = []
    for combo in combos:
        combo_root = V25_CACHE_ROOT / combo
        split_counts = {}
        for split in ["train", "val", "test"]:
            files = sorted(combo_root.joinpath(split).glob("*.npz"))
            split_counts[split] = len(files)
            processed += len(files)
            if files:
                for path in files[: min(10, len(files))]:
                    z = np.load(path, allow_pickle=True)
                    valid_ratios.append(float(_scalar(z["valid_point_ratio"])))
                    teacher_sources.add(str(_scalar(z["teacher_source"])))
                    model_input_observed_only = model_input_observed_only and bool(_scalar(z["model_input_observed_only"]))
                    target_side_box_search_used.append(bool(_scalar(z["target_side_object_box_search_used"])))
        combo_stats[combo] = {
            "exists": combo_root.exists(),
            "split_counts": split_counts,
            "ready": all(split_counts.get(s, 0) > 0 for s in ["train", "val", "test"]),
        }
    bridge_ready = False
    if V24_BRIDGE_PATH.exists():
        bridge = json.loads(V24_BRIDGE_PATH.read_text(encoding="utf-8"))
        bridge_ready = bool(
            bridge.get("object_points_bind_to_semantic_id_instance_id", False)
            and bridge.get("future_semantic_prototype_target_available", False)
            and bridge.get("false_confuser_reacquisition_evaluable", False)
            and bridge.get("no_future_semantic_leakage", False)
            and bridge.get("tusb_fstf_semantic_memory_attachable_as_observed_token", False)
        )
    payload = {
        "audit_name": "stwm_ostf_v26_cache_verification",
        "generated_from": str(V25_CACHE_ROOT.relative_to(ROOT)),
        "combo_stats": combo_stats,
        "processed_clip_count_estimate": processed,
        "valid_point_ratio_sample_mean": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "teacher_sources": sorted(teacher_sources),
        "teacher_source_traceanything_only": teacher_sources == {"traceanything_official_trajectory_field"},
        "semantic_identity_bridge_ready": bridge_ready,
        "model_input_observed_only": bool(model_input_observed_only),
        "target_side_box_search_teacher_only_note": "target_side_object_box_search_used is permitted only for teacher target extraction; model input remains observed-only.",
        "target_side_box_search_used_sample_ratio": float(np.mean(target_side_box_search_used)) if target_side_box_search_used else 0.0,
        "h32_ready": bool(combo_stats["M128_H32"]["ready"] and combo_stats["M512_H32"]["ready"]),
        "h64_ready": bool(combo_stats["M128_H64"]["ready"] and combo_stats["M512_H64"]["ready"]),
        "h64_limited_coverage": int(combo_stats["M128_H64"]["split_counts"]["train"]) < int(combo_stats["M128_H32"]["split_counts"]["train"]),
        "cache_verified": bool(
            combo_stats["M128_H32"]["ready"]
            and combo_stats["M512_H32"]["ready"]
            and combo_stats["M128_H64"]["ready"]
            and combo_stats["M512_H64"]["ready"]
            and teacher_sources == {"traceanything_official_trajectory_field"}
            and bridge_ready
            and (float(np.mean(valid_ratios)) if valid_ratios else 0.0) >= 0.4
            and model_input_observed_only
        ),
    }
    return payload


def write_cache_verification_outputs(report_path: Path, doc_path: Path) -> dict[str, Any]:
    payload = cache_verification_payload()
    dump_json(report_path, payload)
    write_doc(
        doc_path,
        "STWM OSTF V26 Cache Verification",
        payload,
        [
            "cache_verified",
            "processed_clip_count_estimate",
            "h32_ready",
            "h64_ready",
            "h64_limited_coverage",
            "valid_point_ratio_sample_mean",
            "teacher_source_traceanything_only",
            "semantic_identity_bridge_ready",
            "model_input_observed_only",
        ],
    )
    return payload
