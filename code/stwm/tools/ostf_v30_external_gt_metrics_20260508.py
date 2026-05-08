from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import torch


PX_THRESHOLDS = (8.0, 16.0, 32.0, 64.0, 128.0)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def visibility_f1(gt: np.ndarray, logits_or_bool: np.ndarray) -> float:
    pred = logits_or_bool > 0 if logits_or_bool.dtype.kind in "fc" else logits_or_bool.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return float(2.0 * prec * rec / max(prec + rec, 1e-8))


def point_error_metrics(fut_points: np.ndarray, fut_vis: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    valid = fut_vis.astype(bool)
    err = np.abs(pred - fut_points).sum(axis=-1)
    all_err = err[valid] if np.any(valid) else np.asarray([0.0], dtype=np.float32)
    endpoint = err[:, -1][valid[:, -1]] if np.any(valid[:, -1]) else all_err
    out = {
        "minADE": float(all_err.mean()),
        "minFDE": float(endpoint.mean()),
        "endpoint_L1": float(endpoint.mean()),
    }
    for thr in PX_THRESHOLDS:
        out[f"MissRate@{int(thr)}"] = float(out["minFDE"] > thr)
        out[f"PCK@{int(thr)}"] = float((all_err < thr).mean())
    out["threshold_auc_endpoint_16_32_64_128"] = float(
        np.mean([1.0 - out[f"MissRate@{thr}"] for thr in (16, 32, 64, 128)])
    )
    return out


def best_of_k_metrics(fut_points: np.ndarray, fut_vis: np.ndarray, modes: np.ndarray) -> dict[str, float]:
    # modes: [M,H,K,2]
    rows = []
    for k in range(modes.shape[2]):
        rows.append(point_error_metrics(fut_points, fut_vis, modes[:, :, k]))
    best_ade = min(r["minADE"] for r in rows)
    best_fde = min(r["minFDE"] for r in rows)
    out = {"minADE_K": float(best_ade), "minFDE_K": float(best_fde)}
    for thr in PX_THRESHOLDS:
        out[f"BestOfK_PCK@{int(thr)}"] = max(r[f"PCK@{int(thr)}"] for r in rows)
        out[f"BestOfK_MissRate@{int(thr)}"] = float(best_fde > thr)
    out["threshold_auc_endpoint_16_32_64_128"] = float(
        np.mean([1.0 - out[f"BestOfK_MissRate@{thr}"] for thr in (16, 32, 64, 128)])
    )
    return out


def relative_layout_error(fut_points: np.ndarray, fut_vis: np.ndarray, pred: np.ndarray) -> float:
    vals = []
    for t in range(fut_points.shape[1]):
        valid = fut_vis[:, t].astype(bool)
        if int(valid.sum()) < 2:
            continue
        p = pred[valid, t]
        g = fut_points[valid, t]
        vals.append(float(np.abs((p - p.mean(axis=0)) - (g - g.mean(axis=0))).sum(axis=-1).mean()))
    return float(np.mean(vals)) if vals else 0.0


def item_row(
    *,
    uid: str,
    dataset: str,
    horizon: int,
    m_points: int,
    cache_path: str | None = None,
    fut_points: np.ndarray,
    fut_vis: np.ndarray,
    pred: np.ndarray,
    modes: np.ndarray | None = None,
    visibility_logits: np.ndarray | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    item_key = f"{uid}|H{int(horizon)}|M{int(m_points)}|{cache_path or ''}"
    row: dict[str, Any] = {
        "uid": uid,
        "item_key": item_key,
        "cache_path": cache_path,
        "dataset": dataset,
        "H": int(horizon),
        "M": int(m_points),
    }
    row.update(point_error_metrics(fut_points, fut_vis, pred))
    row["relative_deformation_layout_error"] = relative_layout_error(fut_points, fut_vis, pred)
    if modes is not None:
        row.update(best_of_k_metrics(fut_points, fut_vis, modes))
    else:
        row["minADE_K"] = row["minADE"]
        row["minFDE_K"] = row["minFDE"]
    row["visibility_F1"] = visibility_f1(fut_vis, visibility_logits) if visibility_logits is not None else None
    for tag in tags or []:
        row[f"v30_{tag}"] = True
    return row


def aggregate_rows(rows: list[dict[str, Any]], *, subset_key: str | None = None, dataset: str | None = None, horizon: int | None = None, m_points: int | None = None) -> dict[str, Any]:
    filt = []
    for row in rows:
        if subset_key and not row.get(subset_key, False):
            continue
        if dataset and row.get("dataset") != dataset:
            continue
        if horizon and int(row.get("H", 0)) != int(horizon):
            continue
        if m_points and int(row.get("M", 0)) != int(m_points):
            continue
        filt.append(row)
    out: dict[str, Any] = {"item_count": len(filt)}
    keys = [
        "minADE",
        "minFDE",
        "minADE_K",
        "minFDE_K",
        "endpoint_L1",
        "MissRate@16",
        "MissRate@32",
        "MissRate@64",
        "MissRate@128",
        "threshold_auc_endpoint_16_32_64_128",
        "PCK@8",
        "PCK@16",
        "PCK@32",
        "PCK@64",
        "BestOfK_PCK@8",
        "BestOfK_PCK@16",
        "BestOfK_PCK@32",
        "BestOfK_PCK@64",
        "visibility_F1",
        "relative_deformation_layout_error",
    ]
    for key in keys:
        vals = [float(r[key]) for r in filt if r.get(key) is not None and math.isfinite(float(r[key]))]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def aggregate_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = sorted({r["dataset"] for r in rows})
    horizons = sorted({int(r["H"]) for r in rows})
    m_values = sorted({int(r["M"]) for r in rows})
    return {
        "all": aggregate_rows(rows),
        "by_dataset": {ds: aggregate_rows(rows, dataset=ds) for ds in datasets},
        "by_horizon": {f"H{h}": aggregate_rows(rows, horizon=h) for h in horizons},
        "by_M": {f"M{m}": aggregate_rows(rows, m_points=m) for m in m_values},
        "subsets": {
            "motion": aggregate_rows(rows, subset_key="v30_motion"),
            "occlusion_reappearance": aggregate_rows(rows, subset_key="v30_occlusion_reappearance"),
            "nonlinear_large_disp": aggregate_rows(rows, subset_key="v30_nonlinear_large_disp"),
            "long_gap": aggregate_rows(rows, subset_key="v30_long_gap"),
        },
    }


def paired_bootstrap(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric: str,
    *,
    higher_better: bool,
    subset_key: str | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    def keys_for(r: dict[str, Any]) -> list[str]:
        legacy = f"{r.get('uid')}|H{r.get('H')}|M{r.get('M')}"
        full = str(r.get("item_key") or f"{legacy}|{r.get('cache_path','')}")
        return [full, legacy] if full != legacy else [legacy]

    def build_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            for key in keys_for(row):
                out.setdefault(key, row)
        return out

    amap = build_map(rows_a)
    bmap = build_map(rows_b)
    vals = []
    for key in sorted(set(amap) & set(bmap)):
        if subset_key and (not amap[key].get(subset_key, False) or not bmap[key].get(subset_key, False)):
            continue
        if amap[key].get(metric) is None or bmap[key].get(metric) is None:
            continue
        delta = float(amap[key][metric]) - float(bmap[key][metric])
        vals.append(delta if higher_better else -delta)
    if not vals:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    arr = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(123)
    means = [float(arr[rng.integers(0, arr.size, size=arr.size)].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(np.asarray(means), [2.5, 97.5]).tolist()
    return {"item_count": int(arr.size), "mean_delta": float(arr.mean()), "ci95": [float(lo), float(hi)], "zero_excluded": bool(lo > 0 or hi < 0)}


def grouped_count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for row in rows:
        out[str(row.get(key, "unknown"))] += 1
    return dict(sorted(out.items()))
