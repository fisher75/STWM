#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tools.ostf_v17_common_20260502 import (
    ROOT,
    OSTFObjectSample,
    assign_semantic_prototypes,
    batch_from_samples,
    bootstrap_delta,
    dump_json,
    jsonable,
    kmeans_semantic_prototypes,
    load_json,
    load_v16_samples,
    set_seed,
    write_doc,
)


def _onehot_proto_logits(proto_target: np.ndarray, k: int = 32, logit_value: float = 8.0) -> np.ndarray:
    out = np.zeros((proto_target.shape[0], k), dtype=np.float32)
    out[np.arange(proto_target.shape[0]), np.clip(proto_target, 0, k - 1)] = logit_value
    return out


def semantic_logits_from_observed_memory(
    samples: list[OSTFObjectSample],
    *,
    proto_centers: np.ndarray | None,
    proto_count: int = 32,
    horizon: int | None = None,
    temperature: float = 8.0,
) -> np.ndarray:
    if horizon is None:
        horizon = int(samples[0].h) if samples else 1
    if proto_centers is None:
        majority = np.zeros((len(samples), proto_count), dtype=np.float32)
        majority[:, 0] = temperature
        base = majority
    else:
        feats = np.stack([s.semantic_feat for s in samples], axis=0).astype(np.float32)
        centers = np.asarray(proto_centers, dtype=np.float32)
        dist = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        base = (-temperature * dist).astype(np.float32)
    return np.repeat(base[:, None, :], horizon, axis=1)


def analytic_constant_velocity_predict(
    samples: list[OSTFObjectSample],
    *,
    proto_count: int = 32,
    proto_centers: np.ndarray | None = None,
    semantic_mode: str = "oracle",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_preds = []
    vis_logits = []
    sem_logits = semantic_logits_from_observed_memory(
        samples,
        proto_centers=proto_centers,
        proto_count=proto_count,
    ) if semantic_mode == "observed_memory" else None
    for s in samples:
        last = s.obs_points[:, -1]
        vel = s.obs_points[:, -1] - s.obs_points[:, -2]
        times = np.arange(1, s.h + 1, dtype=np.float32)[None, :, None]
        pred = last[:, None, :] + vel[:, None, :] * times
        point_preds.append(pred.astype(np.float32))
        vis_logits.append(np.where(np.repeat(s.obs_vis[:, -1:], s.h, axis=1), 8.0, -8.0).astype(np.float32))
        if sem_logits is None:
            if "oracle" == semantic_mode:
                pass
    if sem_logits is None:
        sem_rows = [
            np.repeat(_onehot_proto_logits(np.asarray([s.proto_target]), proto_count)[:, None, :], s.h, axis=1)[0]
            for s in samples
        ]
        sem_logits = np.stack(sem_rows)
    return np.stack(point_preds), np.stack(vis_logits), sem_logits


def _fit_affine_last_step(s: OSTFObjectSample) -> np.ndarray:
    x = s.obs_points[:, -2] - s.anchor_obs[-2][None]
    y = s.obs_points[:, -1] - s.anchor_obs[-1][None]
    valid = np.logical_and(s.obs_vis[:, -2], s.obs_vis[:, -1])
    if int(valid.sum()) < 3:
        return np.eye(2, dtype=np.float32)
    x = x[valid]
    y = y[valid]
    xtx = x.T @ x + 1e-4 * np.eye(2, dtype=np.float32)
    xty = x.T @ y
    return (np.linalg.solve(xtx, xty)).astype(np.float32)


def analytic_affine_motion_predict(
    samples: list[OSTFObjectSample],
    *,
    proto_count: int = 32,
    proto_centers: np.ndarray | None = None,
    semantic_mode: str = "oracle",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_preds = []
    vis_logits = []
    sem_logits = semantic_logits_from_observed_memory(
        samples,
        proto_centers=proto_centers,
        proto_count=proto_count,
    ) if semantic_mode == "observed_memory" else None
    for s in samples:
        a = _fit_affine_last_step(s)
        anchor_last = s.anchor_obs[-1]
        anchor_vel = s.anchor_obs[-1] - s.anchor_obs[-2]
        rel = s.obs_points[:, -1] - anchor_last[None]
        preds = []
        rel_cur = rel.copy()
        for step in range(s.h):
            rel_cur = rel_cur @ a.T
            anchor_t = anchor_last + anchor_vel * float(step + 1)
            preds.append(rel_cur + anchor_t[None])
        point_preds.append(np.stack(preds, axis=1).astype(np.float32))
        vis_logits.append(np.where(np.repeat(s.obs_vis[:, -1:], s.h, axis=1), 8.0, -8.0).astype(np.float32))
    if sem_logits is None:
        sem_rows = [
            np.repeat(_onehot_proto_logits(np.asarray([s.proto_target]), proto_count)[:, None, :], s.h, axis=1)[0]
            for s in samples
        ]
        sem_logits = np.stack(sem_rows)
    return np.stack(point_preds), np.stack(vis_logits), sem_logits


def extent_center_from_points(points: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    big = torch.full_like(points[..., 0], 1e6)
    x = points[..., 0]
    y = points[..., 1]
    x_min = torch.where(valid, x, big).amin(dim=1)
    y_min = torch.where(valid, y, big).amin(dim=1)
    x_max = torch.where(valid, x, -big).amax(dim=1)
    y_max = torch.where(valid, y, -big).amax(dim=1)
    extent = torch.stack([x_max - x_min, y_max - y_min], dim=-1)
    center = torch.stack([(x_max + x_min) * 0.5, (y_max + y_min) * 0.5], dim=-1)
    extent = torch.nan_to_num(extent, nan=0.0, posinf=0.0, neginf=0.0)
    center = torch.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
    return extent, center


def loss_point_valid(pred_points: torch.Tensor, gt_points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not torch.any(valid):
        return pred_points.new_tensor(0.0)
    return torch.nn.functional.smooth_l1_loss(pred_points[valid], gt_points[valid])


def loss_endpoint(pred_points: torch.Tensor, gt_points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    last_valid = valid[:, :, -1]
    if not torch.any(last_valid):
        return pred_points.new_tensor(0.0)
    return torch.nn.functional.smooth_l1_loss(pred_points[:, :, -1][last_valid], gt_points[:, :, -1][last_valid])


def loss_extent(pred_points: torch.Tensor, gt_points: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    pred_extent, _ = extent_center_from_points(pred_points, valid)
    gt_extent, _ = extent_center_from_points(gt_points, valid)
    frame_mask = valid.any(dim=1)
    if not torch.any(frame_mask):
        return pred_points.new_tensor(0.0)
    return torch.nn.functional.smooth_l1_loss(pred_extent[frame_mask], gt_extent[frame_mask])


def loss_temporal_smoothness(pred_points: torch.Tensor) -> torch.Tensor:
    if pred_points.shape[2] < 3:
        return pred_points.new_tensor(0.0)
    accel = pred_points[:, :, 2:] - 2.0 * pred_points[:, :, 1:-1] + pred_points[:, :, :-2]
    return accel.abs().mean()


def eval_metrics_extended(
    *,
    pred_points: np.ndarray,
    pred_vis_logits: np.ndarray | None,
    pred_proto_logits: np.ndarray | None,
    gt_points: np.ndarray,
    gt_vis: np.ndarray,
    gt_anchor: np.ndarray,
    proto_target: np.ndarray,
) -> dict[str, Any]:
    err_px = np.abs(pred_points - gt_points).sum(axis=-1) * 1000.0
    valid = gt_vis.astype(bool)
    point_l1 = float(err_px[valid].mean()) if np.any(valid) else 0.0
    endpoint = float(err_px[:, :, -1][gt_vis[:, :, -1]].mean()) if np.any(gt_vis[:, :, -1]) else point_l1
    metrics: dict[str, Any] = {
        "point_L1_px": point_l1,
        "endpoint_error_px": endpoint,
        "PCK_4px": float((err_px[valid] < 4.0).mean()) if np.any(valid) else 0.0,
        "PCK_8px": float((err_px[valid] < 8.0).mean()) if np.any(valid) else 0.0,
        "PCK_16px": float((err_px[valid] < 16.0).mean()) if np.any(valid) else 0.0,
        "PCK_32px": float((err_px[valid] < 32.0).mean()) if np.any(valid) else 0.0,
    }
    pred_anchor = pred_points.mean(axis=1)
    metrics["anchor_centroid_L1_px"] = float((np.abs(pred_anchor - gt_anchor).sum(axis=-1) * 1000.0).mean())

    vals = []
    for b in range(pred_points.shape[0]):
        for t in range(pred_points.shape[2]):
            mask = valid[b, :, t]
            if not np.any(mask):
                continue
            pred = pred_points[b, mask, t]
            gt = gt_points[b, mask, t]
            px0, py0 = pred.min(axis=0)
            px1, py1 = pred.max(axis=0)
            gx0, gy0 = gt.min(axis=0)
            gx1, gy1 = gt.max(axis=0)
            ix0, iy0 = max(px0, gx0), max(py0, gy0)
            ix1, iy1 = min(px1, gx1), min(py1, gy1)
            inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
            pa = max(px1 - px0, 0.0) * max(py1 - py0, 0.0)
            ga = max(gx1 - gx0, 0.0) * max(gy1 - gy0, 0.0)
            union = pa + ga - inter
            vals.append(float(inter / union) if union > 0 else 0.0)
    metrics["object_extent_iou"] = float(np.mean(vals)) if vals else 0.0

    if pred_vis_logits is not None:
        pred_vis = pred_vis_logits > 0.0
        tp = int(np.logical_and(pred_vis, gt_vis).sum())
        fp = int(np.logical_and(pred_vis, np.logical_not(gt_vis)).sum())
        fn = int(np.logical_and(np.logical_not(pred_vis), gt_vis).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        metrics["visibility_F1"] = float(2 * prec * rec / max(prec + rec, 1e-8))
    else:
        metrics["visibility_F1"] = None

    semantic = {
        "semantic_changed_top5": None,
        "stable_semantic_preservation": None,
        "semantic_metric_note": "future semantic target is constant per object prototype in current OSTF teacher cache; changed/stable semantic split remains weak.",
    }
    if pred_proto_logits is not None:
        top5 = np.argsort(pred_proto_logits, axis=-1)[..., -5:]
        top1 = pred_proto_logits.argmax(axis=-1)
        tgt = proto_target[:, None]
        semantic.update(
            {
                "semantic_top1": float((top1 == tgt).mean()),
                "semantic_top5": float((top5 == tgt[..., None]).any(axis=-1).mean()),
            }
        )
    return {**metrics, **semantic}


def item_scores_from_predictions(
    samples: list[OSTFObjectSample],
    pred_points: np.ndarray,
    pred_vis_logits: np.ndarray | None,
    pred_proto_logits: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pred_vis = pred_vis_logits > 0.0 if pred_vis_logits is not None else None
    for i, s in enumerate(samples):
        err = np.abs(pred_points[i] - s.fut_points).sum(axis=-1) * 1000.0
        valid = s.fut_vis
        px = float(err[valid].mean()) if np.any(valid) else 0.0
        endpoint = float(err[:, -1][s.fut_vis[:, -1]].mean()) if np.any(s.fut_vis[:, -1]) else px
        pred_anchor = pred_points[i].mean(axis=0)
        anchor_l1 = float((np.abs(pred_anchor - s.anchor_fut).sum(axis=-1) * 1000.0).mean())
        vis_f1 = None
        if pred_vis is not None:
            pv = pred_vis[i]
            tp = int(np.logical_and(pv, s.fut_vis).sum())
            fp = int(np.logical_and(pv, np.logical_not(s.fut_vis)).sum())
            fn = int(np.logical_and(np.logical_not(pv), s.fut_vis).sum())
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            vis_f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
        semantic_top5 = None
        if pred_proto_logits is not None:
            top5 = np.argsort(pred_proto_logits[i], axis=-1)[..., -5:]
            semantic_top5 = float((top5 == s.proto_target).any(axis=-1).mean())
        vals = []
        for t in range(s.h):
            mask = valid[:, t]
            if not np.any(mask):
                continue
            pred = pred_points[i, mask, t]
            gt = s.fut_points[mask, t]
            px0, py0 = pred.min(axis=0)
            px1, py1 = pred.max(axis=0)
            gx0, gy0 = gt.min(axis=0)
            gx1, gy1 = gt.max(axis=0)
            ix0, iy0 = max(px0, gx0), max(py0, gy0)
            ix1, iy1 = min(px1, gx1), min(py1, gy1)
            inter = max(ix1 - ix0, 0.0) * max(iy1 - iy0, 0.0)
            pa = max(px1 - px0, 0.0) * max(py1 - py0, 0.0)
            ga = max(gx1 - gx0, 0.0) * max(gy1 - gy0, 0.0)
            union = pa + ga - inter
            vals.append(float(inter / union) if union > 0 else 0.0)
        rows.append(
            {
                "item_key": s.item_key,
                "dataset": s.dataset,
                "object_index": s.object_index,
                "object_id": s.object_id,
                "point_l1_px": px,
                "endpoint_error_px": endpoint,
                "anchor_l1_px": anchor_l1,
                "extent_iou": float(np.mean(vals)) if vals else 0.0,
                "visibility_f1": vis_f1,
                "semantic_top5": semantic_top5,
            }
        )
    return rows


def metrics_by_dataset(samples: list[OSTFObjectSample], pred_points: np.ndarray, pred_vis_logits: np.ndarray | None, pred_proto_logits: np.ndarray | None) -> dict[str, Any]:
    datasets = sorted({s.dataset for s in samples})
    out: dict[str, Any] = {}
    for ds in datasets:
        idx = [i for i, s in enumerate(samples) if s.dataset == ds]
        out[ds] = eval_metrics_extended(
            pred_points=pred_points[idx],
            pred_vis_logits=pred_vis_logits[idx] if pred_vis_logits is not None else None,
            pred_proto_logits=pred_proto_logits[idx] if pred_proto_logits is not None else None,
            gt_points=np.stack([samples[i].fut_points for i in idx]),
            gt_vis=np.stack([samples[i].fut_vis for i in idx]),
            gt_anchor=np.stack([samples[i].anchor_fut for i in idx]),
            proto_target=np.asarray([samples[i].proto_target for i in idx], dtype=np.int64),
        )
    return out


def build_v18_rows(combo: str, seed: int = 42) -> tuple[dict[str, list[OSTFObjectSample]], np.ndarray]:
    rows = load_v16_samples(combo)
    centers = kmeans_semantic_prototypes(rows["train"], k=32, iters=25, seed=seed)
    assign_semantic_prototypes(rows, centers)
    return rows, centers
