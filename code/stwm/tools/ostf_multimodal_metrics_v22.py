#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

import numpy as np

from stwm.tools.ostf_multimodal_metrics_v21 import PX_SCALE, _det_item_metrics, _extent_iou_single, paired_bootstrap_from_rows
from stwm.tools.ostf_v17_common_20260502 import OSTFObjectSample


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)


def _gaussian_mixture_nll(
    gt_points: np.ndarray,
    valid: np.ndarray,
    point_modes: np.ndarray,
    probs: np.ndarray,
    logvar_modes: np.ndarray | None,
) -> float | None:
    if logvar_modes is None:
        return None
    # gt_points [M,H,2], point_modes [M,H,K,2], logvar_modes [H,K] or [M,H,K] or [M,H,K,1]
    err2 = ((point_modes - gt_points[:, :, None, :]) ** 2).sum(axis=-1) * (PX_SCALE**2)  # [M,H,K]
    if logvar_modes.ndim == 2:
        logvar = logvar_modes[None, :, :]
    elif logvar_modes.ndim == 4:
        logvar = logvar_modes[..., 0]
    else:
        logvar = logvar_modes
    logvar = np.broadcast_to(logvar, err2.shape)
    mask = valid[:, :, None].astype(np.float64)
    denom = max(float(mask.sum()), 1.0)
    log_comp = -0.5 * ((err2 / np.exp(logvar)) + logvar)
    log_comp = (log_comp * mask).sum(axis=(0, 1)) / denom
    log_probs = np.log(np.maximum(probs, 1e-8)) + log_comp
    mx = float(log_probs.max())
    ll = mx + np.log(np.exp(log_probs - mx).sum())
    return float(-ll)


def multimodal_item_scores_v22(
    samples: list[OSTFObjectSample],
    *,
    point_modes: np.ndarray,  # [B,M,H,K,2]
    mode_logits: np.ndarray,  # [B,K]
    point_pred: np.ndarray,  # [B,M,H,2]
    top1_pred: np.ndarray,  # [B,M,H,2]
    pred_vis_logits: np.ndarray | None,
    pred_proto_logits: np.ndarray | None,
    logvar_modes: np.ndarray | None = None,  # [B,H,K] or [B,M,H,K]
    subset_flags: dict[str, np.ndarray] | None = None,
    cv_mode_index: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    probs = _softmax(mode_logits)
    for i, s in enumerate(samples):
        valid = s.fut_vis
        det = _det_item_metrics(s, point_pred[i], None if pred_vis_logits is None else pred_vis_logits[i])
        top1_det = _det_item_metrics(s, top1_pred[i], None if pred_vis_logits is None else pred_vis_logits[i])
        err_modes = np.abs(point_modes[i] - s.fut_points[:, :, None, :]).sum(axis=-1) * PX_SCALE  # [M,H,K]
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
            fde_k = err_modes[:, -1, :][valid[:, -1]].mean(axis=0)
        else:
            fde_k = ade_k.copy()
        best_ade_idx = int(np.argmin(ade_k))
        best_fde_idx = int(np.argmin(fde_k))
        top1_idx = int(np.argmax(probs[i]))
        top1_prob = float(probs[i][top1_idx])
        oracle_best_prob = float(probs[i][best_fde_idx])
        expected_fde = float((probs[i] * fde_k).sum())
        expected_ade = float((probs[i] * ade_k).sum())
        endpoint_points = point_modes[i, :, -1] * PX_SCALE
        pair_d = []
        for a in range(endpoint_points.shape[1]):
            for b in range(a + 1, endpoint_points.shape[1]):
                pair_d.append(float(np.linalg.norm(endpoint_points[:, a] - endpoint_points[:, b], axis=-1).mean()))
        pair_mean = float(np.mean(pair_d)) if pair_d else 0.0
        miss16 = float(np.min(fde_k) > 16.0)
        miss32 = float(np.min(fde_k) > 32.0)
        top1_miss16 = float(fde_k[top1_idx] > 16.0)
        top1_miss32 = float(fde_k[top1_idx] > 32.0)
        row = {
            "item_key": s.item_key,
            "dataset": s.dataset,
            "object_id": int(s.object_id),
            "weighted_point_l1_px": det["point_l1_px"],
            "weighted_endpoint_error_px": det["endpoint_error_px"],
            "weighted_PCK_8px": det["PCK_8px"],
            "weighted_PCK_16px": det["PCK_16px"],
            "weighted_PCK_32px": det["PCK_32px"],
            "weighted_anchor_centroid_L1_px": det["anchor_centroid_L1_px"],
            "weighted_object_extent_iou": det["object_extent_iou"],
            "weighted_visibility_F1": det["visibility_F1"],
            "top1_point_l1_px": top1_det["point_l1_px"],
            "top1_endpoint_error_px": top1_det["endpoint_error_px"],
            "top1_PCK_8px": top1_det["PCK_8px"],
            "top1_PCK_16px": top1_det["PCK_16px"],
            "top1_PCK_32px": top1_det["PCK_32px"],
            "top1_anchor_centroid_L1_px": top1_det["anchor_centroid_L1_px"],
            "top1_object_extent_iou": top1_det["object_extent_iou"],
            "top1_visibility_F1": top1_det["visibility_F1"],
            "top1_MissRate_16px": top1_miss16,
            "top1_MissRate_32px": top1_miss32,
            "minADE_K_px": float(ade_k[best_ade_idx]),
            "minFDE_K_px": float(fde_k[best_fde_idx]),
            "expected_ADE_px": expected_ade,
            "expected_FDE_px": expected_fde,
            "BestOfK_PCK_8px": float(pck8_k.max()),
            "BestOfK_PCK_16px": float(pck16_k.max()),
            "BestOfK_PCK_32px": float(pck32_k.max()),
            "MissRate_16px": miss16,
            "MissRate_32px": miss32,
            "best_mode_idx_ADE": best_ade_idx,
            "best_mode_idx_FDE": best_fde_idx,
            "best_mode_is_cv_FDE": bool(best_fde_idx == cv_mode_index),
            "top1_mode_idx": top1_idx,
            "top1_mode_is_cv": bool(top1_idx == cv_mode_index),
            "top1_matches_oracle_best": bool(top1_idx == best_fde_idx),
            "top1_mode_prob": top1_prob,
            "oracle_best_mode_prob": oracle_best_prob,
            "mode_entropy": float(-(probs[i] * np.log(np.maximum(probs[i], 1e-8))).sum()),
            "pairwise_endpoint_diversity_px": pair_mean,
            "collapse_rate_8px": float(pair_mean < 8.0),
            "collapse_rate_16px": float(pair_mean < 16.0),
            "cv_mode_probability": float(probs[i][cv_mode_index]) if cv_mode_index < probs.shape[1] else None,
            "mode_nll": _gaussian_mixture_nll(s.fut_points, valid, point_modes[i], probs[i], None if logvar_modes is None else logvar_modes[i]),
        }
        if pred_proto_logits is not None:
            top5 = np.argsort(pred_proto_logits[i], axis=-1)[..., -5:]
            top1 = pred_proto_logits[i].argmax(axis=-1)
            row["semantic_top1"] = float((top1 == s.proto_target).mean())
            row["semantic_top5"] = float((top5 == s.proto_target).any(axis=-1).mean())
        else:
            row["semantic_top1"] = None
            row["semantic_top5"] = None
        if subset_flags is not None:
            for name, arr in subset_flags.items():
                row[name] = bool(arr[i])
        rows.append(row)
    return rows


def aggregate_rows_v22(rows: list[dict[str, Any]], *, subset_key: str | None = None, dataset: str | None = None) -> dict[str, Any]:
    filt = []
    for r in rows:
        if dataset is not None and r["dataset"] != dataset:
            continue
        if subset_key is not None and not r.get(subset_key, False):
            continue
        filt.append(r)
    if not filt:
        return {"item_count": 0}
    keys = [
        "weighted_point_l1_px",
        "weighted_endpoint_error_px",
        "weighted_PCK_8px",
        "weighted_PCK_16px",
        "weighted_PCK_32px",
        "weighted_object_extent_iou",
        "top1_point_l1_px",
        "top1_endpoint_error_px",
        "top1_PCK_8px",
        "top1_PCK_16px",
        "top1_PCK_32px",
        "top1_object_extent_iou",
        "top1_MissRate_16px",
        "top1_MissRate_32px",
        "minADE_K_px",
        "minFDE_K_px",
        "expected_ADE_px",
        "expected_FDE_px",
        "BestOfK_PCK_8px",
        "BestOfK_PCK_16px",
        "BestOfK_PCK_32px",
        "MissRate_16px",
        "MissRate_32px",
        "mode_entropy",
        "pairwise_endpoint_diversity_px",
        "collapse_rate_8px",
        "collapse_rate_16px",
        "top1_mode_prob",
        "oracle_best_mode_prob",
        "cv_mode_probability",
        "mode_nll",
        "semantic_top1",
        "semantic_top5",
    ]
    out: dict[str, Any] = {"item_count": len(filt)}
    for k in keys:
        vals = [float(r[k]) for r in filt if r.get(k) is not None]
        out[k] = float(np.mean(vals)) if vals else None
    out["top1_mode_accuracy"] = float(np.mean([1.0 if r["top1_matches_oracle_best"] else 0.0 for r in filt]))
    out["best_mode_non_cv_rate"] = float(np.mean([0.0 if r["best_mode_is_cv_FDE"] else 1.0 for r in filt]))
    out["top1_non_cv_rate"] = float(np.mean([0.0 if r["top1_mode_is_cv"] else 1.0 for r in filt]))
    return out


def calibration_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "item_count": 0,
            "top1_mode_accuracy": None,
            "ece_top1_mode": None,
            "oracle_best_mode_prob_mean": None,
            "mode_nll_mean": None,
            "hypothesis_diversity_valid": False,
        }
    conf = np.asarray([float(r["top1_mode_prob"]) for r in rows], dtype=np.float64)
    corr = np.asarray([1.0 if r["top1_matches_oracle_best"] else 0.0 for r in rows], dtype=np.float64)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not np.any(mask):
            continue
        ece += float(np.abs(conf[mask].mean() - corr[mask].mean()) * mask.mean())
    nll_vals = [float(r["mode_nll"]) for r in rows if r.get("mode_nll") is not None]
    div = np.mean([float(r["pairwise_endpoint_diversity_px"]) for r in rows])
    collapse = np.mean([float(r["collapse_rate_16px"]) for r in rows])
    return {
        "item_count": int(len(rows)),
        "top1_mode_accuracy": float(corr.mean()),
        "ece_top1_mode": float(ece),
        "oracle_best_mode_prob_mean": float(np.mean([float(r["oracle_best_mode_prob"]) for r in rows])),
        "top1_mode_prob_mean": float(conf.mean()),
        "mode_nll_mean": float(np.mean(nll_vals)) if nll_vals else None,
        "hypothesis_diversity_valid": bool(div > 12.0 and collapse < 0.60),
        "pairwise_endpoint_diversity_px": float(div),
        "collapse_rate_16px": float(collapse),
    }


def expected_vs_oracle_bootstrap(
    rows: list[dict[str, Any]],
    *,
    subset_key: str | None = None,
) -> dict[str, Any]:
    oracle = []
    expected = []
    for r in rows:
        if subset_key is not None and not r.get(subset_key, False):
            continue
        oracle.append(float(r["minFDE_K_px"]))
        expected.append(float(r["expected_FDE_px"]))
    if not oracle:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    a = [{"item_key": str(i), "object_id": 0, "m": o} for i, o in enumerate(oracle)]
    b = [{"item_key": str(i), "object_id": 0, "m": e} for i, e in enumerate(expected)]
    rows_a = [{"item_key": str(i), "object_id": 0, "metric": float(oracle[i])} for i in range(len(oracle))]
    rows_b = [{"item_key": str(i), "object_id": 0, "metric": float(expected[i])} for i in range(len(expected))]
    return paired_bootstrap_from_rows(rows_a, rows_b, metric="metric", higher_better=False)
