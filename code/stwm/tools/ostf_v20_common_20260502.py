#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tools.ostf_v17_common_20260502 import ROOT, OSTFObjectSample, dump_json, load_v16_samples, set_seed, write_doc
from stwm.tools.ostf_v18_common_20260502 import (
    analytic_affine_motion_predict,
    analytic_constant_velocity_predict,
    build_v18_rows,
    eval_metrics_extended,
    item_scores_from_predictions,
    metrics_by_dataset,
)


def key_tuple(item_key: str, object_id: int) -> tuple[str, int]:
    return (str(item_key), int(object_id))


def sample_key(sample: OSTFObjectSample) -> tuple[str, int]:
    return key_tuple(sample.item_key, sample.object_id)


def _safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.sqrt(np.maximum((x * x).sum(axis=axis), 0.0))


def _safe_mean(x: np.ndarray) -> float:
    return float(np.asarray(x, dtype=np.float64).mean()) if np.asarray(x).size else 0.0


def _rank_flag(values: np.ndarray, frac: float) -> np.ndarray:
    assert 0.0 < frac < 1.0
    n = len(values)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    k = max(1, int(math.ceil(frac * n)))
    idx = np.argsort(values)[-k:]
    out = np.zeros((n,), dtype=bool)
    out[idx] = True
    return out


def load_combo_rows(combo: str, seed: int = 42) -> tuple[dict[str, list[OSTFObjectSample]], np.ndarray]:
    return build_v18_rows(combo, seed=seed)


def load_cache_npz(sample: OSTFObjectSample) -> np.lib.npyio.NpzFile:
    return np.load(ROOT / sample.source_cache_path, allow_pickle=True)


def load_predecode_npz(sample: OSTFObjectSample) -> np.lib.npyio.NpzFile:
    z = load_cache_npz(sample)
    return np.load(str(z["predecode_path"].item()), allow_pickle=True)


def observed_box_features(sample: OSTFObjectSample, pre: np.lib.npyio.NpzFile) -> np.ndarray:
    if "entity_boxes_over_time" not in pre.files:
        return np.zeros((14,), dtype=np.float32)
    boxes = np.asarray(pre["entity_boxes_over_time"], dtype=np.float32)
    obj = int(sample.object_index)
    if obj >= boxes.shape[1]:
        return np.zeros((14,), dtype=np.float32)
    obs = boxes[: sample.obs_points.shape[1], obj]
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


def neighbor_features(sample: OSTFObjectSample, cache: np.lib.npyio.NpzFile) -> np.ndarray:
    tracks = np.asarray(cache["tracks_xy"], dtype=np.float32)
    vis = np.asarray(cache["visibility"]).astype(bool)
    obs_len = int(cache["obs_len"].item())
    obj = int(sample.object_index)
    if obj >= tracks.shape[0]:
        return np.zeros((10,), dtype=np.float32)
    cur_anchor = sample.anchor_obs[-1]
    cur_vel = sample.anchor_obs_vel[-1]
    other = []
    for j in range(tracks.shape[0]):
        if j == obj:
            continue
        mask = vis[j, :, :obs_len]
        if not np.any(mask):
            continue
        pts = tracks[j, :, :obs_len]
        denom = np.maximum(mask.sum(axis=0, keepdims=False), 1.0)
        anchor = (pts * mask[..., None]).sum(axis=0) / denom[:, None]
        vel = np.zeros_like(anchor)
        if anchor.shape[0] > 1:
            vel[1:] = anchor[1:] - anchor[:-1]
        other.append((anchor[-1], vel[-1]))
    if not other:
        return np.zeros((10,), dtype=np.float32)
    pos = np.stack([x[0] for x in other], axis=0)
    vel = np.stack([x[1] for x in other], axis=0)
    rel = pos - cur_anchor[None]
    dist = _safe_norm(rel, axis=-1)
    near_idx = int(dist.argmin())
    near_rel = rel[near_idx]
    near_vel = vel[near_idx] - cur_vel
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
    return feats


def global_motion_features(cache: np.lib.npyio.NpzFile) -> np.ndarray:
    tracks = np.asarray(cache["tracks_xy"], dtype=np.float32)
    vis = np.asarray(cache["visibility"]).astype(bool)
    obs_len = int(cache["obs_len"].item())
    anchors = []
    vels = []
    for j in range(tracks.shape[0]):
        mask = vis[j, :, :obs_len]
        if not np.any(mask):
            continue
        pts = tracks[j, :, :obs_len]
        denom = np.maximum(mask.sum(axis=0, keepdims=False), 1.0)
        anchor = (pts * mask[..., None]).sum(axis=0) / denom[:, None]
        vel = np.zeros_like(anchor)
        if anchor.shape[0] > 1:
            vel[1:] = anchor[1:] - anchor[:-1]
        anchors.append(anchor[-1])
        vels.append(vel[-1])
    if not anchors:
        return np.zeros((8,), dtype=np.float32)
    anchors = np.stack(anchors, axis=0)
    vels = np.stack(vels, axis=0)
    feats = np.asarray(
        [
            np.median(vels[:, 0]),
            np.median(vels[:, 1]),
            vels[:, 0].std(),
            vels[:, 1].std(),
            _safe_norm(vels, axis=-1).mean(),
            _safe_norm(vels, axis=-1).std(),
            anchors[:, 0].std(),
            anchors[:, 1].std(),
        ],
        dtype=np.float32,
    )
    return feats


def sample_difficulty(sample: OSTFObjectSample, cache: np.lib.npyio.NpzFile) -> dict[str, float]:
    cv_pred, _, _ = analytic_constant_velocity_predict([sample], semantic_mode="observed_memory")
    cv_pred = cv_pred[0]
    gt = sample.fut_points
    valid = sample.fut_vis
    err = np.abs(cv_pred - gt).sum(axis=-1)
    point_l1 = float(err[valid].mean()) if np.any(valid) else 0.0
    endpoint = float(err[:, -1][valid[:, -1]].mean()) if np.any(valid[:, -1]) else point_l1
    teacher_anchor = sample.anchor_fut
    full_anchor = np.concatenate([sample.anchor_obs, sample.anchor_fut], axis=0)
    future_accel = np.zeros((max(len(teacher_anchor), 1), 2), dtype=np.float32)
    if full_anchor.shape[0] >= 3:
        accel = full_anchor[2:] - 2.0 * full_anchor[1:-1] + full_anchor[:-2]
        future_accel[: accel[-sample.h :].shape[0]] = accel[-sample.h :]
    curvature = float(_safe_norm(future_accel, axis=-1).mean())
    occlusion_ratio = float(1.0 - sample.fut_vis.mean())
    reappear = 0.0
    vis_any = sample.fut_vis.any(axis=0)
    if vis_any.size >= 3:
        dropped = False
        for flag in vis_any:
            if not flag:
                dropped = True
            elif dropped and flag:
                reappear = 1.0
                break
    neigh = neighbor_features(sample, cache)
    interaction = float(max(0.0, 0.10 - float(neigh[0]))) + float(np.linalg.norm(neigh[6:8]))
    global_feat = global_motion_features(cache)
    global_motion = float(np.linalg.norm(global_feat[:2]))
    return {
        "cv_point_l1_proxy": point_l1,
        "cv_endpoint_proxy": endpoint,
        "curvature_proxy": curvature,
        "occlusion_ratio": occlusion_ratio,
        "reappearance_flag": reappear,
        "interaction_proxy": interaction,
        "global_motion_proxy": global_motion,
    }


def context_features_for_sample(sample: OSTFObjectSample) -> dict[str, np.ndarray | float]:
    cache = load_cache_npz(sample)
    pre = np.load(str(cache["predecode_path"].item()), allow_pickle=True)
    crop_feat = np.asarray(sample.semantic_feat, dtype=np.float32)
    box_feat = observed_box_features(sample, pre)
    neigh_feat = neighbor_features(sample, cache)
    global_feat = global_motion_features(cache)
    diff = sample_difficulty(sample, cache)
    context_vec = np.concatenate(
        [
            crop_feat,
            box_feat,
            neigh_feat,
            global_feat,
            np.asarray(
                [
                    diff["cv_point_l1_proxy"],
                    diff["cv_endpoint_proxy"],
                    diff["curvature_proxy"],
                    diff["occlusion_ratio"],
                    diff["reappearance_flag"],
                    diff["interaction_proxy"],
                    diff["global_motion_proxy"],
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    ).astype(np.float32)
    return {
        "crop_feat": crop_feat,
        "box_feat": box_feat,
        "neighbor_feat": neigh_feat,
        "global_feat": global_feat,
        "context_vec": context_vec,
        **diff,
    }


def build_context_cache_for_combo(combo: str, seed: int = 42) -> dict[str, Any]:
    rows, proto_centers = build_v18_rows(combo, seed=seed)
    records = []
    for split, samples in rows.items():
        for s in samples:
            feats = context_features_for_sample(s)
            records.append(
                {
                    "key": sample_key(s),
                    "item_key": s.item_key,
                    "object_id": int(s.object_id),
                    "dataset": s.dataset,
                    "split": split,
                    "source_combo": combo,
                    "proto_target": int(s.proto_target),
                    **feats,
                }
            )
    # train stats for normalization and hard-score thresholds
    train = [r for r in records if r["split"] == "train"]
    ctx = np.stack([r["context_vec"] for r in train], axis=0)
    mean = ctx.mean(axis=0).astype(np.float32)
    std = ctx.std(axis=0).astype(np.float32) + 1e-6
    for r in records:
        r["context_vec_norm"] = ((r["context_vec"] - mean) / std).astype(np.float32)
    train_cv = np.asarray([r["cv_point_l1_proxy"] for r in train], dtype=np.float32)
    train_curv = np.asarray([r["curvature_proxy"] for r in train], dtype=np.float32)
    train_occ = np.asarray([r["occlusion_ratio"] for r in train], dtype=np.float32)
    train_int = np.asarray([r["interaction_proxy"] for r in train], dtype=np.float32)
    cv_mu, cv_std = float(train_cv.mean()), float(train_cv.std() + 1e-6)
    curv_mu, curv_std = float(train_curv.mean()), float(train_curv.std() + 1e-6)
    int_mu, int_std = float(train_int.mean()), float(train_int.std() + 1e-6)
    for r in records:
        hard = (
            (float(r["cv_point_l1_proxy"]) - cv_mu) / cv_std
            + 0.75 * (float(r["curvature_proxy"]) - curv_mu) / curv_std
            + 0.50 * float(r["occlusion_ratio"])
            + 0.35 * (float(r["interaction_proxy"]) - int_mu) / int_std
            + 0.20 * float(r["reappearance_flag"])
        )
        r["hardness_score"] = float(hard)
    return {
        "records": records,
        "context_mean": mean,
        "context_std": std,
        "proto_centers": proto_centers,
    }


def save_context_cache(bundle: dict[str, Any], out_path: Path) -> None:
    records = bundle["records"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        item_key=np.asarray([r["item_key"] for r in records], dtype=object),
        object_id=np.asarray([r["object_id"] for r in records], dtype=np.int64),
        dataset=np.asarray([r["dataset"] for r in records], dtype=object),
        split=np.asarray([r["split"] for r in records], dtype=object),
        source_combo=np.asarray([r["source_combo"] for r in records], dtype=object),
        proto_target=np.asarray([r["proto_target"] for r in records], dtype=np.int64),
        crop_feat=np.stack([r["crop_feat"] for r in records], axis=0),
        box_feat=np.stack([r["box_feat"] for r in records], axis=0),
        neighbor_feat=np.stack([r["neighbor_feat"] for r in records], axis=0),
        global_feat=np.stack([r["global_feat"] for r in records], axis=0),
        context_vec=np.stack([r["context_vec"] for r in records], axis=0),
        context_vec_norm=np.stack([r["context_vec_norm"] for r in records], axis=0),
        cv_point_l1_proxy=np.asarray([r["cv_point_l1_proxy"] for r in records], dtype=np.float32),
        cv_endpoint_proxy=np.asarray([r["cv_endpoint_proxy"] for r in records], dtype=np.float32),
        curvature_proxy=np.asarray([r["curvature_proxy"] for r in records], dtype=np.float32),
        occlusion_ratio=np.asarray([r["occlusion_ratio"] for r in records], dtype=np.float32),
        reappearance_flag=np.asarray([r["reappearance_flag"] for r in records], dtype=np.float32),
        interaction_proxy=np.asarray([r["interaction_proxy"] for r in records], dtype=np.float32),
        global_motion_proxy=np.asarray([r["global_motion_proxy"] for r in records], dtype=np.float32),
        hardness_score=np.asarray([r["hardness_score"] for r in records], dtype=np.float32),
        context_mean=bundle["context_mean"],
        context_std=bundle["context_std"],
        proto_centers=bundle["proto_centers"],
        future_leakage_audit=np.asarray(False),
    )


def load_context_cache(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    z = np.load(path, allow_pickle=True)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for i in range(len(z["item_key"])):
        key = key_tuple(z["item_key"][i].item() if hasattr(z["item_key"][i], "item") else z["item_key"][i], int(z["object_id"][i]))
        out[key] = {
            "dataset": str(z["dataset"][i]),
            "split": str(z["split"][i]),
            "source_combo": str(z["source_combo"][i]),
            "proto_target": int(z["proto_target"][i]),
            "crop_feat": np.asarray(z["crop_feat"][i], dtype=np.float32),
            "box_feat": np.asarray(z["box_feat"][i], dtype=np.float32),
            "neighbor_feat": np.asarray(z["neighbor_feat"][i], dtype=np.float32),
            "global_feat": np.asarray(z["global_feat"][i], dtype=np.float32),
            "context_vec": np.asarray(z["context_vec"][i], dtype=np.float32),
            "context_vec_norm": np.asarray(z["context_vec_norm"][i], dtype=np.float32),
            "cv_point_l1_proxy": float(z["cv_point_l1_proxy"][i]),
            "cv_endpoint_proxy": float(z["cv_endpoint_proxy"][i]),
            "curvature_proxy": float(z["curvature_proxy"][i]),
            "occlusion_ratio": float(z["occlusion_ratio"][i]),
            "reappearance_flag": float(z["reappearance_flag"][i]),
            "interaction_proxy": float(z["interaction_proxy"][i]),
            "global_motion_proxy": float(z["global_motion_proxy"][i]),
            "hardness_score": float(z["hardness_score"][i]),
        }
    return out


def evaluate_subset_metrics(
    samples: list[OSTFObjectSample],
    pred_points: np.ndarray,
    pred_vis_logits: np.ndarray | None,
    pred_proto_logits: np.ndarray | None,
    subset_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if subset_mask is None:
        subset_mask = np.ones((len(samples),), dtype=bool)
    idx = np.where(np.asarray(subset_mask, dtype=bool))[0]
    if idx.size == 0:
        return {
            "item_count": 0,
            "point_L1_px": None,
            "endpoint_error_px": None,
            "PCK_4px": None,
            "PCK_8px": None,
            "PCK_16px": None,
            "PCK_32px": None,
            "object_extent_iou": None,
            "visibility_F1": None,
            "semantic_top1": None,
            "semantic_top5": None,
        }
    return {
        "item_count": int(idx.size),
        **eval_metrics_extended(
            pred_points=pred_points[idx],
            pred_vis_logits=pred_vis_logits[idx] if pred_vis_logits is not None else None,
            pred_proto_logits=pred_proto_logits[idx] if pred_proto_logits is not None else None,
            gt_points=np.stack([samples[i].fut_points for i in idx]),
            gt_vis=np.stack([samples[i].fut_vis for i in idx]),
            gt_anchor=np.stack([samples[i].anchor_fut for i in idx]),
            proto_target=np.asarray([samples[i].proto_target for i in idx], dtype=np.int64),
        ),
    }


def hard_subset_flags(context_records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    cv = np.asarray([r["cv_point_l1_proxy"] for r in context_records], dtype=np.float32)
    curv = np.asarray([r["curvature_proxy"] for r in context_records], dtype=np.float32)
    occ = np.asarray([r["occlusion_ratio"] for r in context_records], dtype=np.float32)
    inter = np.asarray([r["interaction_proxy"] for r in context_records], dtype=np.float32)
    return {
        "top10_cv_hard": _rank_flag(cv, 0.10),
        "top20_cv_hard": _rank_flag(cv, 0.20),
        "top30_cv_hard": _rank_flag(cv, 0.30),
        "nonlinear_hard": _rank_flag(curv, 0.20),
        "occlusion_hard": np.logical_or(_rank_flag(occ, 0.20), occ > 0.15),
        "interaction_hard": _rank_flag(inter, 0.20),
    }


def item_map(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for row in report.get("item_scores", []):
        out[key_tuple(row["item_key"], row["object_id"])] = row
    return out


def bootstrap_delta(values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    assert values_a.shape == values_b.shape
    n = values_a.shape[0]
    if n == 0:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    delta = values_a - values_b
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(delta[idx].mean()))
    means = np.asarray(means, dtype=np.float64)
    lo, hi = np.percentile(means, [2.5, 97.5]).tolist()
    return {
        "item_count": int(n),
        "mean_delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool((lo > 0.0) or (hi < 0.0)),
    }
