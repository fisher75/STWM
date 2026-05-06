#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import numpy as np


PX_SCALE = 1000.0


def _extent_iou_single(pred_points: np.ndarray, gt_points: np.ndarray, valid: np.ndarray) -> float:
    vals = []
    for t in range(pred_points.shape[1]):
        mask = valid[:, t]
        if not np.any(mask):
            continue
        pred = pred_points[mask, t]
        gt = gt_points[mask, t]
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
    return float(np.mean(vals)) if vals else 0.0


def _det_item_metrics(sample: Any, pred_points: np.ndarray, pred_vis_logits: np.ndarray | None) -> dict[str, Any]:
    pred_points = np.nan_to_num(pred_points, nan=0.0, posinf=2.0, neginf=-1.0).astype(np.float32, copy=False)
    err_px = np.abs(pred_points - sample.fut_points).sum(axis=-1) * PX_SCALE
    valid = sample.fut_vis
    point_l1 = float(err_px[valid].mean()) if np.any(valid) else 0.0
    endpoint = float(err_px[:, -1][valid[:, -1]].mean()) if np.any(valid[:, -1]) else point_l1
    pck8 = float((err_px[valid] < 8.0).mean()) if np.any(valid) else 0.0
    pck16 = float((err_px[valid] < 16.0).mean()) if np.any(valid) else 0.0
    pck32 = float((err_px[valid] < 32.0).mean()) if np.any(valid) else 0.0
    pred_anchor = pred_points.mean(axis=0)
    anchor_l1 = float((np.abs(pred_anchor - sample.anchor_fut).sum(axis=-1) * PX_SCALE).mean())
    extent_iou = _extent_iou_single(pred_points, sample.fut_points, valid)
    miss16 = float(endpoint > 16.0)
    miss32 = float(endpoint > 32.0)
    miss64 = float(endpoint > 64.0)
    if pred_vis_logits is not None:
        pred_vis = pred_vis_logits > 0.0
        tp = int(np.logical_and(pred_vis, sample.fut_vis).sum())
        fp = int(np.logical_and(pred_vis, np.logical_not(sample.fut_vis)).sum())
        fn = int(np.logical_and(np.logical_not(pred_vis), sample.fut_vis).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        vis_f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
    else:
        vis_f1 = None
    return {
        "point_l1_px": point_l1,
        "endpoint_error_px": endpoint,
        "PCK_8px": pck8,
        "PCK_16px": pck16,
        "PCK_32px": pck32,
        "anchor_centroid_L1_px": anchor_l1,
        "object_extent_iou": extent_iou,
        "MissRate_16px": miss16,
        "MissRate_32px": miss32,
        "MissRate_64px": miss64,
        "visibility_F1": vis_f1,
    }


def multimodal_item_scores_v26(
    samples: list[Any],
    *,
    point_modes: np.ndarray,
    mode_logits: np.ndarray,
    top1_point_pred: np.ndarray,
    weighted_point_pred: np.ndarray,
    pred_vis_logits: np.ndarray | None,
    pred_proto_logits: np.ndarray | None,
    pred_logvar: np.ndarray | None,
    cv_mode_index: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    point_modes = np.nan_to_num(point_modes, nan=0.0, posinf=2.0, neginf=-1.0).astype(np.float32, copy=False)
    top1_point_pred = np.nan_to_num(top1_point_pred, nan=0.0, posinf=2.0, neginf=-1.0).astype(np.float32, copy=False)
    weighted_point_pred = np.nan_to_num(weighted_point_pred, nan=0.0, posinf=2.0, neginf=-1.0).astype(np.float32, copy=False)
    mode_logits = np.nan_to_num(mode_logits, nan=0.0, posinf=25.0, neginf=-25.0).astype(np.float32, copy=False)
    if pred_logvar is not None:
        pred_logvar = np.nan_to_num(pred_logvar, nan=0.0, posinf=2.0, neginf=-6.0).astype(np.float32, copy=False)
    probs = np.exp(mode_logits - mode_logits.max(axis=-1, keepdims=True))
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-8)
    for i, s in enumerate(samples):
        top1 = _det_item_metrics(s, top1_point_pred[i], None if pred_vis_logits is None else pred_vis_logits[i])
        weighted = _det_item_metrics(s, weighted_point_pred[i], None if pred_vis_logits is None else pred_vis_logits[i])
        valid = s.fut_vis
        err_modes = np.abs(point_modes[i] - s.fut_points[:, :, None, :]).sum(axis=-1) * PX_SCALE
        if np.any(valid):
            denom = np.maximum(valid.sum(), 1)
            ade_k = (err_modes * valid[..., None]).sum(axis=(0, 1)) / denom
            pck8_k = ((err_modes[valid] < 8.0).reshape(-1, err_modes.shape[-1])).mean(axis=0)
            pck16_k = ((err_modes[valid] < 16.0).reshape(-1, err_modes.shape[-1])).mean(axis=0)
            pck32_k = ((err_modes[valid] < 32.0).reshape(-1, err_modes.shape[-1])).mean(axis=0)
        else:
            ade_k = np.zeros((point_modes.shape[3],), dtype=np.float32)
            pck8_k = np.zeros_like(ade_k)
            pck16_k = np.zeros_like(ade_k)
            pck32_k = np.zeros_like(ade_k)
        if np.any(valid[:, -1]):
            endpoint_k = err_modes[:, -1, :][valid[:, -1]].mean(axis=0)
        else:
            endpoint_k = ade_k.copy()
        best_ade_idx = int(np.argmin(ade_k))
        best_fde_idx = int(np.argmin(endpoint_k))
        minade = float(ade_k[best_ade_idx])
        minfde = float(endpoint_k[best_fde_idx])
        best_pck8 = float(pck8_k.max())
        best_pck16 = float(pck16_k.max())
        best_pck32 = float(pck32_k.max())
        miss16 = float(minfde > 16.0)
        miss32 = float(minfde > 32.0)
        miss64 = float(minfde > 64.0)
        endpoint_points = point_modes[i, :, -1] * PX_SCALE
        pair_d = []
        for a in range(endpoint_points.shape[1]):
            for b in range(a + 1, endpoint_points.shape[1]):
                pair_d.append(float(np.linalg.norm(endpoint_points[:, a] - endpoint_points[:, b], axis=-1).mean()))
        pair_mean = float(np.mean(pair_d)) if pair_d else 0.0
        collapse8 = float(pair_mean < 8.0)
        top1_mode = int(np.argmax(probs[i]))
        top1_matches_oracle = bool(top1_mode == best_fde_idx)
        entropy = float(-(probs[i] * np.log(np.maximum(probs[i], 1e-8))).sum())
        nll = None
        if pred_logvar is not None:
            diff_sq = np.clip(((point_modes[i] - s.fut_points[:, :, None, :]) ** 2).sum(axis=-1), 0.0, 64.0)
            logvar = pred_logvar[i]
            if logvar.ndim == 2:
                logvar = logvar[None].repeat(s.m, axis=0)
            logvar = np.broadcast_to(logvar, diff_sq.shape)
            valid_f = valid[..., None].astype(np.float32)
            mode_nll = 0.5 * (np.exp(-logvar) * diff_sq + logvar) * valid_f
            denom = max(float(valid.sum()), 1.0)
            mode_nll = mode_nll.sum(axis=(0, 1)) / denom
            logprob = np.log(np.maximum(probs[i], 1e-8))
            nll = float(-np.log(np.exp(logprob - mode_nll).sum()))
        row = {
            "item_key": s.item_key,
            "dataset": s.dataset,
            "object_id": int(s.object_id),
            "top1_point_L1_px": top1["point_l1_px"],
            "top1_endpoint_error_px": top1["endpoint_error_px"],
            "top1_PCK_8px": top1["PCK_8px"],
            "top1_PCK_16px": top1["PCK_16px"],
            "top1_PCK_32px": top1["PCK_32px"],
            "top1_anchor_centroid_L1_px": top1["anchor_centroid_L1_px"],
            "top1_object_extent_iou": top1["object_extent_iou"],
            "top1_visibility_F1": top1["visibility_F1"],
            "weighted_point_L1_px": weighted["point_l1_px"],
            "weighted_endpoint_error_px": weighted["endpoint_error_px"],
            "weighted_object_extent_iou": weighted["object_extent_iou"],
            "minADE_K_px": minade,
            "minFDE_K_px": minfde,
            "BestOfK_PCK_8px": best_pck8,
            "BestOfK_PCK_16px": best_pck16,
            "BestOfK_PCK_32px": best_pck32,
            "MissRate_16px": miss16,
            "MissRate_32px": miss32,
            "MissRate_64px": miss64,
            "best_mode_idx_ADE": best_ade_idx,
            "best_mode_idx_FDE": best_fde_idx,
            "best_mode_is_cv_ADE": bool(best_ade_idx == cv_mode_index),
            "best_mode_is_cv_FDE": bool(best_fde_idx == cv_mode_index),
            "top1_mode_idx": top1_mode,
            "top1_mode_is_cv": bool(top1_mode == cv_mode_index),
            "top1_mode_matches_oracle_fde": top1_matches_oracle,
            "mode_entropy": entropy,
            "pairwise_endpoint_diversity_px": pair_mean,
            "collapse_rate_8px": collapse8,
            "nll": nll,
            "occlusion_hard": bool(s.subset_flags.get("occlusion_hard", False)),
            "nonlinear_hard": bool(s.subset_flags.get("nonlinear_hard", False)),
            "interaction_hard": bool(s.subset_flags.get("interaction_hard", False)),
            "top20_cv_hard": bool(s.subset_flags.get("top20_cv_hard", False)),
            "top30_cv_hard": bool(s.subset_flags.get("top30_cv_hard", False)),
        }
        if pred_proto_logits is not None:
            top5 = np.argsort(pred_proto_logits[i], axis=-1)[..., -5:]
            top1_sem = pred_proto_logits[i].argmax(axis=-1)
            row["semantic_top1"] = float((top1_sem == s.proto_target).mean())
            row["semantic_top5"] = float((top5 == s.proto_target).any(axis=-1).mean())
        else:
            row["semantic_top1"] = None
            row["semantic_top5"] = None
        rows.append(row)
    return rows


def aggregate_item_rows_v26(rows: list[dict[str, Any]], *, subset_key: str | None = None, dataset: str | None = None) -> dict[str, Any]:
    filt = []
    for r in rows:
        if dataset is not None and r["dataset"] != dataset:
            continue
        if subset_key is not None and not r.get(subset_key, False):
            continue
        filt.append(r)
    if not filt:
        return {"item_count": 0}
    numeric_keys = [
        "top1_point_L1_px",
        "top1_endpoint_error_px",
        "top1_PCK_8px",
        "top1_PCK_16px",
        "top1_PCK_32px",
        "top1_anchor_centroid_L1_px",
        "top1_object_extent_iou",
        "top1_visibility_F1",
        "weighted_point_L1_px",
        "weighted_endpoint_error_px",
        "weighted_object_extent_iou",
        "minADE_K_px",
        "minFDE_K_px",
        "BestOfK_PCK_8px",
        "BestOfK_PCK_16px",
        "BestOfK_PCK_32px",
        "MissRate_16px",
        "MissRate_32px",
        "MissRate_64px",
        "pairwise_endpoint_diversity_px",
        "collapse_rate_8px",
        "mode_entropy",
        "nll",
        "semantic_top1",
        "semantic_top5",
    ]
    out = {"item_count": len(filt)}
    for key in numeric_keys:
        vals = [float(r[key]) for r in filt if r.get(key) is not None and np.isfinite(float(r[key]))]
        out[key] = float(np.mean(vals)) if vals else None
    out["best_mode_non_cv_rate"] = float(np.mean([0.0 if r["best_mode_is_cv_FDE"] else 1.0 for r in filt]))
    out["top1_mode_non_cv_rate"] = float(np.mean([0.0 if r["top1_mode_is_cv"] else 1.0 for r in filt]))
    out["top1_matches_oracle_fde_rate"] = float(np.mean([1.0 if r["top1_mode_matches_oracle_fde"] else 0.0 for r in filt]))
    return out


def paired_bootstrap_from_rows_v26(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    metric: str,
    higher_better: bool,
    subset_key: str | None = None,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    amap = {(r["item_key"], int(r["object_id"])): r for r in rows_a}
    bmap = {(r["item_key"], int(r["object_id"])): r for r in rows_b}
    keys = sorted(set(amap) & set(bmap))
    av = []
    bv = []
    for k in keys:
        ra = amap[k]
        rb = bmap[k]
        if subset_key is not None and (not ra.get(subset_key, False) or not rb.get(subset_key, False)):
            continue
        if ra.get(metric) is None or rb.get(metric) is None:
            continue
        av.append(float(ra[metric]))
        bv.append(float(rb[metric]))
    if not av:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    av_np = np.asarray(av, dtype=np.float64)
    bv_np = np.asarray(bv, dtype=np.float64)
    if not higher_better:
        av_np = -av_np
        bv_np = -bv_np
    delta = av_np - bv_np
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(delta), size=len(delta))
        means.append(float(delta[idx].mean()))
    means_np = np.asarray(means, dtype=np.float64)
    lo, hi = np.percentile(means_np, [2.5, 97.5]).tolist()
    return {
        "item_count": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool((lo > 0.0) or (hi < 0.0)),
    }


def hypothesis_diversity_valid_v26(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    div = np.mean([float(r["pairwise_endpoint_diversity_px"]) for r in rows])
    collapse = np.mean([float(r["collapse_rate_8px"]) for r in rows])
    non_cv = np.mean([0.0 if r["best_mode_is_cv_FDE"] else 1.0 for r in rows])
    return bool(div > 8.0 and collapse < 0.75 and non_cv > 0.10)
