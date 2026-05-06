#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from stwm.tools.ostf_traceanything_common_v26_20260502 import (
    ROOT,
    TraceAnythingOSTFSample,
    analytic_last_observed_copy_predict,
    build_v26_rows,
)
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import (
    aggregate_item_rows_v26,
    multimodal_item_scores_v26,
    paired_bootstrap_from_rows_v26,
)
from stwm.tools.ostf_v18_common_20260502 import semantic_logits_from_observed_memory


PX_SCALE = 1000.0
COMBOS = ("M128_H32", "M512_H32", "M128_H64")
GAMMA_GRID = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)


def scalar(x: Any) -> Any:
    a = np.asarray(x)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def load_combo(combo: str) -> tuple[dict[str, list[TraceAnythingOSTFSample]], np.ndarray]:
    return build_v26_rows(combo, seed=42)


def observed_memory_logits(
    samples: list[TraceAnythingOSTFSample],
    proto_centers: np.ndarray,
    proto_count: int = 32,
) -> np.ndarray:
    return semantic_logits_from_observed_memory(samples, proto_centers=proto_centers, proto_count=proto_count)


def visibility_logits_last(samples: list[TraceAnythingOSTFSample]) -> np.ndarray:
    return np.stack(
        [
            np.where(np.repeat(s.obs_vis[:, -1:], s.h, axis=1), 8.0, -8.0).astype(np.float32)
            for s in samples
        ],
        axis=0,
    )


def predict_last(samples: list[TraceAnythingOSTFSample]) -> np.ndarray:
    return np.stack([np.repeat(s.obs_points[:, -1:, :], s.h, axis=1).astype(np.float32) for s in samples], axis=0)


def predict_damped_velocity(samples: list[TraceAnythingOSTFSample], gamma: float) -> np.ndarray:
    preds = []
    for s in samples:
        last = s.obs_points[:, -1]
        vel = s.obs_points[:, -1] - s.obs_points[:, -2]
        times = np.arange(1, s.h + 1, dtype=np.float32)[None, :, None]
        preds.append((last[:, None, :] + float(gamma) * vel[:, None, :] * times).astype(np.float32))
    return np.stack(preds, axis=0)


def predict_lowpass_spline(samples: list[TraceAnythingOSTFSample]) -> np.ndarray:
    preds = []
    for s in samples:
        obs = s.obs_points.astype(np.float32)
        if obs.shape[1] >= 4:
            v = (obs[:, -1] - obs[:, -4]) / 3.0
        else:
            v = obs[:, -1] - obs[:, -2]
        # A conservative decaying velocity baseline. It is deliberately non-oracle.
        steps = np.arange(1, s.h + 1, dtype=np.float32)
        damp_cum = np.cumsum(np.power(0.75, steps - 1.0))[None, :, None]
        preds.append((obs[:, -1:, :] + v[:, None, :] * damp_cum).astype(np.float32))
    return np.stack(preds, axis=0)


def _fit_affine_stable(s: TraceAnythingOSTFSample) -> np.ndarray:
    x = s.obs_points[:, -2] - s.anchor_obs[-2][None]
    y = s.obs_points[:, -1] - s.anchor_obs[-1][None]
    valid = np.logical_and(s.obs_vis[:, -2], s.obs_vis[:, -1])
    if int(valid.sum()) < 6:
        return np.eye(2, dtype=np.float32)
    x = x[valid].astype(np.float64)
    y = y[valid].astype(np.float64)
    xtx = x.T @ x + 1e-3 * np.eye(2, dtype=np.float64)
    xty = x.T @ y
    try:
        mat = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        mat = np.eye(2, dtype=np.float64)
    if not np.isfinite(mat).all():
        mat = np.eye(2, dtype=np.float64)
    return np.clip(mat, -1.25, 1.25).astype(np.float32)


def predict_stable_affine(samples: list[TraceAnythingOSTFSample], anchor_gamma: float = 0.25) -> np.ndarray:
    preds = []
    for s in samples:
        mat = _fit_affine_stable(s)
        anchor_last = s.anchor_obs[-1]
        anchor_vel = s.anchor_obs[-1] - s.anchor_obs[-2]
        rel = s.obs_points[:, -1] - anchor_last[None]
        rel_cur = rel.copy()
        steps = []
        for step in range(s.h):
            rel_cur = rel_cur @ mat.T
            anchor_t = anchor_last + float(anchor_gamma) * anchor_vel * float(step + 1)
            steps.append(rel_cur + anchor_t[None])
        preds.append(np.stack(steps, axis=1).astype(np.float32))
    return np.stack(preds, axis=0)


def predict_oracle_gamma(samples: list[TraceAnythingOSTFSample], gammas: tuple[float, ...] = GAMMA_GRID) -> np.ndarray:
    all_preds = [predict_damped_velocity(samples, g) for g in gammas]
    out = []
    for i, s in enumerate(samples):
        valid = s.fut_vis
        best = None
        best_score = float("inf")
        for pred in all_preds:
            err = np.abs(pred[i] - s.fut_points).sum(axis=-1) * PX_SCALE
            score = float(err[:, -1][valid[:, -1]].mean()) if np.any(valid[:, -1]) else float(err[valid].mean())
            if score < best_score:
                best_score = score
                best = pred[i]
        out.append(best.astype(np.float32))
    return np.stack(out, axis=0)


def item_rows_for_prediction(
    samples: list[TraceAnythingOSTFSample],
    pred_points: np.ndarray,
    proto_centers: np.ndarray,
) -> list[dict[str, Any]]:
    vis_logits = visibility_logits_last(samples)
    sem_logits = observed_memory_logits(samples, proto_centers)
    return multimodal_item_scores_v26(
        samples,
        point_modes=pred_points[:, :, :, None, :],
        mode_logits=np.zeros((len(samples), 1), dtype=np.float32),
        top1_point_pred=pred_points,
        weighted_point_pred=pred_points,
        pred_vis_logits=vis_logits,
        pred_proto_logits=sem_logits,
        pred_logvar=None,
        cv_mode_index=0,
    )


def evaluate_prior(
    samples: list[TraceAnythingOSTFSample],
    proto_centers: np.ndarray,
    pred_points: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = item_rows_for_prediction(samples, pred_points, proto_centers)
    subsets = {
        "top20_cv_hard": aggregate_item_rows_v26(rows, subset_key="top20_cv_hard"),
        "top30_cv_hard": aggregate_item_rows_v26(rows, subset_key="top30_cv_hard"),
        "occlusion": aggregate_item_rows_v26(rows, subset_key="occlusion_hard"),
        "nonlinear": aggregate_item_rows_v26(rows, subset_key="nonlinear_hard"),
        "interaction": aggregate_item_rows_v26(rows, subset_key="interaction_hard"),
    }
    by_ds = {ds: aggregate_item_rows_v26(rows, dataset=ds) for ds in sorted({s.dataset for s in samples})}
    return rows, aggregate_item_rows_v26(rows), subsets, by_ds


def choose_gamma_on_val(
    samples: list[TraceAnythingOSTFSample],
    proto_centers: np.ndarray,
    gammas: tuple[float, ...] = GAMMA_GRID,
) -> tuple[float, dict[str, Any]]:
    scores = {}
    best_gamma = gammas[0]
    best_score = float("inf")
    for gamma in gammas:
        _, agg, _, _ = evaluate_prior(samples, proto_centers, predict_damped_velocity(samples, gamma))
        score = float(agg.get("minFDE_K_px") or float("inf"))
        scores[str(gamma)] = agg
        if score < best_score:
            best_score = score
            best_gamma = gamma
    return float(best_gamma), scores


def displacement_stats(samples: list[TraceAnythingOSTFSample], pred_points: np.ndarray | None = None) -> dict[str, Any]:
    vals_l1 = []
    vals_l2 = []
    endpoint_l1 = []
    endpoint_l2 = []
    for i, s in enumerate(samples):
        ref = pred_points[i] if pred_points is not None else np.repeat(s.obs_points[:, -1:, :], s.h, axis=1)
        diff = s.fut_points - ref
        valid = s.fut_vis
        if np.any(valid):
            l1 = np.abs(diff).sum(axis=-1)[valid] * PX_SCALE
            l2 = np.linalg.norm(diff, axis=-1)[valid] * PX_SCALE
            vals_l1.extend(l1.astype(float).tolist())
            vals_l2.extend(l2.astype(float).tolist())
        if np.any(valid[:, -1]):
            ed = diff[:, -1][valid[:, -1]]
            endpoint_l1.extend((np.abs(ed).sum(axis=-1) * PX_SCALE).astype(float).tolist())
            endpoint_l2.extend((np.linalg.norm(ed, axis=-1) * PX_SCALE).astype(float).tolist())

    def q(xs: list[float]) -> dict[str, Any]:
        if not xs:
            return {"count": 0}
        arr = np.asarray(xs, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p90": float(np.percentile(arr, 90)),
            "p95": float(np.percentile(arr, 95)),
        }

    return {
        "all_step_l1_px": q(vals_l1),
        "all_step_l2_px": q(vals_l2),
        "endpoint_l1_px": q(endpoint_l1),
        "endpoint_l2_px": q(endpoint_l2),
    }


def finite_affine_bug_check(samples: list[TraceAnythingOSTFSample]) -> dict[str, Any]:
    bad = 0
    max_abs = []
    for s in samples:
        mat = _fit_affine_stable(s)
        if not np.isfinite(mat).all():
            bad += 1
        max_abs.append(float(np.abs(mat).max()))
    return {
        "sample_count": len(samples),
        "nonfinite_stable_affine_count": int(bad),
        "stable_affine_max_abs_p95": float(np.percentile(max_abs, 95)) if max_abs else None,
        "stable_affine_numeric_bug_detected": bool(bad > 0),
    }


def bootstrap_metric(a: list[dict[str, Any]], b: list[dict[str, Any]], metric: str, higher_better: bool, subset_key: str | None = None) -> dict[str, Any]:
    return paired_bootstrap_from_rows_v26(a, b, metric=metric, higher_better=higher_better, subset_key=subset_key)


def dataset_counts(samples: list[TraceAnythingOSTFSample], flags: dict[str, list[bool]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mask in flags.items():
        by_ds: dict[str, int] = defaultdict(int)
        for s, flag in zip(samples, mask):
            if flag:
                by_ds[s.dataset] += 1
        out[name] = {"count": int(sum(by_ds.values())), "by_dataset": dict(sorted(by_ds.items()))}
    return out
