#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_lastobs_v28_common_20260502 import (
    ROOT,
    add_v28_flags_to_item_rows,
    build_v28_rows,
    choose_visibility_aware_gamma_on_val,
    predict_last,
    predict_last_visible_copy,
    predict_median_object_anchor_copy,
    predict_visibility_aware_cv,
    predict_visibility_aware_damped_velocity,
    visibility_logits_last_visible,
)
from stwm.tools.ostf_traceanything_common_v26_20260502 import analytic_constant_velocity_predict
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import aggregate_item_rows_v26, multimodal_item_scores_v26
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v27_prior_utils_20260502 import observed_memory_logits, predict_stable_affine


PX_SCALE = 1000.0
COMBOS = ("M128_H32", "M512_H32", "M128_H64", "M512_H64")
PRIMARY_MANIFEST_COMBOS = ("M128_H32", "M128_H64")
PRIOR_NAMES = (
    "last_observed_copy",
    "last_visible_copy",
    "visibility_aware_damped",
    "visibility_aware_cv",
    "constant_velocity",
    "fixed_affine",
    "median_object_anchor_copy",
)
THRESHOLDS = (16.0, 32.0, 64.0, 128.0)
V29_MANIFEST_DIR = ROOT / "manifests/ostf_v29_antiprior"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def quantiles(values: list[float] | np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p80": float(np.percentile(arr, 80)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def combo_horizon(combo: str) -> int:
    return int(combo.split("_H", 1)[1])


def combo_m(combo: str) -> int:
    return int(combo.split("_H", 1)[0].replace("M", ""))


def sample_uid(sample: Any, combo: str) -> str:
    return f"{combo}::{sample.split}::{sample.item_key}::obj{int(sample.object_id)}"


def logical_uid_from_row(row: dict[str, Any]) -> str:
    return f"{row.get('item_key')}::obj{int(row.get('object_id', -1))}"


def sample_logical_uid(sample: Any) -> str:
    return f"{sample.item_key}::obj{int(sample.object_id)}"


def dataset_counts(samples: list[Any]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for sample in samples:
        out[str(sample.dataset)] += 1
    return dict(sorted(out.items()))


def _last_visible_xy(obs_points: np.ndarray, obs_vis: np.ndarray) -> np.ndarray:
    out = obs_points[:, -1].astype(np.float32, copy=True)
    for i in range(obs_points.shape[0]):
        idx = np.flatnonzero(obs_vis[i])
        if idx.size:
            out[i] = obs_points[i, int(idx[-1])]
    return out


def _last_two_visible_velocity(obs_points: np.ndarray, obs_vis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    last = obs_points[:, -1].astype(np.float32, copy=True)
    vel = np.zeros_like(last, dtype=np.float32)
    for i in range(obs_points.shape[0]):
        idx = np.flatnonzero(obs_vis[i])
        if idx.size:
            last_idx = int(idx[-1])
            last[i] = obs_points[i, last_idx]
            if idx.size >= 2:
                prev_idx = int(idx[-2])
                vel[i] = (obs_points[i, last_idx] - obs_points[i, prev_idx]) / max(float(last_idx - prev_idx), 1.0)
    return last, vel


def _safe_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else 0.0


def endpoint_error_px(sample: Any, pred_points: np.ndarray) -> float:
    valid = sample.fut_vis[:, -1]
    if not np.any(valid):
        valid_all = sample.fut_vis
        if not np.any(valid_all):
            return 0.0
        return float((np.abs(pred_points - sample.fut_points).sum(axis=-1)[valid_all] * PX_SCALE).mean())
    return float((np.abs(pred_points[:, -1] - sample.fut_points[:, -1]).sum(axis=-1)[valid] * PX_SCALE).mean())


def ade_error_px(sample: Any, pred_points: np.ndarray) -> float:
    valid = sample.fut_vis
    if not np.any(valid):
        return 0.0
    return float((np.abs(pred_points - sample.fut_points).sum(axis=-1)[valid] * PX_SCALE).mean())


def pck_px(sample: Any, pred_points: np.ndarray, threshold: float) -> float:
    valid = sample.fut_vis
    if not np.any(valid):
        return 0.0
    err = np.abs(pred_points - sample.fut_points).sum(axis=-1) * PX_SCALE
    return float((err[valid] < float(threshold)).mean())


def relative_deformation_error_px(sample: Any, pred_points: np.ndarray) -> float:
    vals: list[float] = []
    for t in range(sample.h):
        valid = sample.fut_vis[:, t]
        if int(valid.sum()) < 2:
            continue
        pred = pred_points[valid, t]
        gt = sample.fut_points[valid, t]
        pred_rel = pred - pred.mean(axis=0, keepdims=True)
        gt_rel = gt - gt.mean(axis=0, keepdims=True)
        vals.append(float(np.abs(pred_rel - gt_rel).sum(axis=-1).mean() * PX_SCALE))
    return float(np.mean(vals)) if vals else 0.0


def future_displacement_features(sample: Any) -> dict[str, float]:
    last_frame = sample.obs_points[:, -1]
    last_visible, vel_visible = _last_two_visible_velocity(sample.obs_points, sample.obs_vis)
    endpoint = sample.fut_points[:, -1]
    endpoint_valid = sample.fut_vis[:, -1]
    all_valid = sample.fut_vis
    if np.any(endpoint_valid):
        endpoint_disp_last_visible = np.linalg.norm(endpoint[endpoint_valid] - last_visible[endpoint_valid], axis=-1) * PX_SCALE
        endpoint_disp_last_frame = np.linalg.norm(endpoint[endpoint_valid] - last_frame[endpoint_valid], axis=-1) * PX_SCALE
    else:
        endpoint_disp_last_visible = np.asarray([0.0])
        endpoint_disp_last_frame = np.asarray([0.0])
    if np.any(all_valid):
        all_disp_last_visible = np.linalg.norm(sample.fut_points - last_visible[:, None], axis=-1)[all_valid] * PX_SCALE
    else:
        all_disp_last_visible = np.asarray([0.0])
    if sample.fut_points.shape[1] >= 3:
        acc = sample.fut_points[:, 2:] - 2.0 * sample.fut_points[:, 1:-1] + sample.fut_points[:, :-2]
        acc_valid = sample.fut_vis[:, 2:] & sample.fut_vis[:, 1:-1] & sample.fut_vis[:, :-2]
        future_curvature = float(np.linalg.norm(acc, axis=-1)[acc_valid].mean() * PX_SCALE) if np.any(acc_valid) else 0.0
    else:
        future_curvature = 0.0
    return {
        "future_endpoint_displacement_from_last_visible_px": float(np.mean(endpoint_disp_last_visible)),
        "future_endpoint_displacement_from_last_frame_px": float(np.mean(endpoint_disp_last_frame)),
        "future_displacement_from_last_visible_px": float(np.mean(all_disp_last_visible)),
        "observed_velocity_magnitude_px": float(np.linalg.norm(vel_visible, axis=-1).mean() * PX_SCALE),
        "future_curvature_acceleration_px": future_curvature,
        "occlusion_ratio": float(sample.occlusion_ratio),
        "reappearance_flag": float(sample.reappearance_flag),
        "valid_future_point_ratio": float(sample.fut_vis.mean()) if sample.fut_vis.size else 0.0,
        "target_extraction_uncertainty": float((1.0 - sample.fut_vis.mean()) + (1.0 - _safe_mean(sample.fut_conf[sample.fut_vis])) if np.any(sample.fut_vis) else 2.0),
    }


def _metric_update_for_prediction(row: dict[str, Any], sample: Any, pred_points: np.ndarray) -> dict[str, Any]:
    fde = endpoint_error_px(sample, pred_points)
    row["MissRate_128px"] = float(fde > 128.0)
    row["BestOfK_PCK_64px"] = pck_px(sample, pred_points, 64.0)
    row["threshold_auc_endpoint_16_32_64_128"] = float(np.mean([1.0 - float(fde > thr) for thr in THRESHOLDS]))
    row["relative_deformation_layout_error_px"] = relative_deformation_error_px(sample, pred_points)
    return row


def aggregate_extended_rows(rows: list[dict[str, Any]], *, subset_key: str | None = None, dataset: str | None = None) -> dict[str, Any]:
    base = aggregate_item_rows_v26(rows, subset_key=subset_key, dataset=dataset)
    filt = []
    for row in rows:
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if subset_key is not None and not row.get(subset_key, False):
            continue
        filt.append(row)
    for key in ["MissRate_128px", "BestOfK_PCK_64px", "threshold_auc_endpoint_16_32_64_128", "relative_deformation_layout_error_px"]:
        vals = [float(r[key]) for r in filt if r.get(key) is not None and np.isfinite(float(r[key]))]
        base[key] = float(np.mean(vals)) if vals else None
    return base


def prior_predictions(samples: list[Any], proto_centers: np.ndarray, *, visibility_gamma: float) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    pred_last = predict_last(samples)
    pred_last_visible = predict_last_visible_copy(samples)
    pred_vis_damped = predict_visibility_aware_damped_velocity(samples, visibility_gamma)
    pred_vis_cv = predict_visibility_aware_cv(samples)
    pred_cv, pred_cv_vis, pred_cv_sem = analytic_constant_velocity_predict(
        samples,
        proto_count=32,
        proto_centers=proto_centers,
        semantic_mode="observed_memory",
    )
    pred_affine = predict_stable_affine(samples, anchor_gamma=0.25)
    pred_median = predict_median_object_anchor_copy(samples)
    vis_logits = visibility_logits_last_visible(samples)
    sem_logits = observed_memory_logits(samples, proto_centers, proto_count=32)
    return {
        "last_observed_copy": (pred_last, vis_logits, sem_logits),
        "last_visible_copy": (pred_last_visible, vis_logits, sem_logits),
        "visibility_aware_damped": (pred_vis_damped, vis_logits, sem_logits),
        "visibility_aware_cv": (pred_vis_cv, vis_logits, sem_logits),
        "constant_velocity": (pred_cv, pred_cv_vis, pred_cv_sem),
        "fixed_affine": (pred_affine, vis_logits, sem_logits),
        "median_object_anchor_copy": (pred_median, vis_logits, sem_logits),
    }


def rows_for_prior_prediction(samples: list[Any], pred_points: np.ndarray, pred_vis: np.ndarray, pred_sem: np.ndarray) -> list[dict[str, Any]]:
    rows = multimodal_item_scores_v26(
        samples,
        point_modes=pred_points[:, :, :, None, :],
        mode_logits=np.zeros((len(samples), 1), dtype=np.float32),
        top1_point_pred=pred_points,
        weighted_point_pred=pred_points,
        pred_vis_logits=pred_vis,
        pred_proto_logits=pred_sem,
        pred_logvar=None,
        cv_mode_index=0,
    )
    rows = add_v28_flags_to_item_rows(rows, samples)
    for row, sample, pred in zip(rows, samples, pred_points):
        _metric_update_for_prediction(row, sample, pred)
    return rows


def evaluate_prior_suite(samples: list[Any], proto_centers: np.ndarray, *, visibility_gamma: float) -> dict[str, Any]:
    preds = prior_predictions(samples, proto_centers, visibility_gamma=visibility_gamma)
    out: dict[str, Any] = {}
    for name, (pred, pred_vis, pred_sem) in preds.items():
        rows = rows_for_prior_prediction(samples, pred, pred_vis, pred_sem)
        datasets = sorted({s.dataset for s in samples})
        out[name] = {
            "item_rows": rows,
            "metrics": aggregate_extended_rows(rows),
            "metrics_by_dataset": {ds: aggregate_extended_rows(rows, dataset=ds) for ds in datasets},
        }
    return out


def metric_distributions_from_rows(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    return quantiles([float(r[metric]) for r in rows if r.get(metric) is not None])


def item_feature_payload(sample: Any, combo: str, split: str, prior_rows_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    idx = None
    uid = sample_logical_uid(sample)
    for i, row in enumerate(next(iter(prior_rows_by_name.values()))):
        if logical_uid_from_row(row) == uid:
            idx = i
            break
    if idx is None:
        raise RuntimeError(f"Could not match sample {uid} in prior rows")
    prior_metrics = {
        name: {
            "minADE_K_px": rows[idx].get("minADE_K_px"),
            "minFDE_K_px": rows[idx].get("minFDE_K_px"),
            "MissRate_16px": rows[idx].get("MissRate_16px"),
            "MissRate_32px": rows[idx].get("MissRate_32px"),
            "MissRate_64px": rows[idx].get("MissRate_64px"),
            "MissRate_128px": rows[idx].get("MissRate_128px"),
            "threshold_auc_endpoint_16_32_64_128": rows[idx].get("threshold_auc_endpoint_16_32_64_128"),
            "BestOfK_PCK_8px": rows[idx].get("BestOfK_PCK_8px"),
            "BestOfK_PCK_16px": rows[idx].get("BestOfK_PCK_16px"),
            "BestOfK_PCK_32px": rows[idx].get("BestOfK_PCK_32px"),
            "BestOfK_PCK_64px": rows[idx].get("BestOfK_PCK_64px"),
            "relative_deformation_layout_error_px": rows[idx].get("relative_deformation_layout_error_px"),
            "visibility_F1": rows[idx].get("top1_visibility_F1"),
            "object_extent_iou": rows[idx].get("top1_object_extent_iou"),
        }
        for name, rows in prior_rows_by_name.items()
    }
    features = future_displacement_features(sample)
    return {
        "uid": sample_uid(sample, combo),
        "logical_uid": uid,
        "combo": combo,
        "M": combo_m(combo),
        "H": combo_horizon(combo),
        "split": split,
        "item_key": sample.item_key,
        "object_id": int(sample.object_id),
        "semantic_id": int(sample.semantic_id),
        "dataset": sample.dataset,
        "source_cache_path": sample.source_cache_path,
        "frame_count": len(sample.frame_paths),
        "horizon_feasible": len(sample.frame_paths) >= 8 + int(sample.h),
        "reason_tags": list(sample.reason_tags),
        "semantic_identity_confuser": bool(sample.subset_flags.get("semantic_identity_confuser", False)),
        "occlusion_reappearance": bool(sample.subset_flags.get("occlusion_reappearance", False) or sample.reappearance_flag > 0.5),
        "subset_flags": dict(sample.subset_flags),
        **features,
        "prior_metrics": prior_metrics,
    }


def summarize_manifest_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, int] = defaultdict(int)
    by_combo: dict[str, int] = defaultdict(int)
    by_subset: dict[str, int] = defaultdict(int)
    for entry in entries:
        by_dataset[str(entry.get("dataset"))] += 1
        by_combo[str(entry.get("combo"))] += 1
        for tag in entry.get("v29_subset_tags", []):
            by_subset[str(tag)] += 1
    return {
        "item_count": len(entries),
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_combo": dict(sorted(by_combo.items())),
        "by_subset": dict(sorted(by_subset.items())),
    }


def dump_manifest(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest_name": path.stem,
        "generated_at_utc": utc_now(),
        "item_count": len(entries),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def available_external_dataset_preflight() -> dict[str, Any]:
    candidates = {
        "PointOdyssey": [ROOT / "data/PointOdyssey", ROOT / "data/pointodyssey", Path("/raid/chen034/data/PointOdyssey")],
        "TAP-Vid": [ROOT / "data/tapvid", ROOT / "data/TAP-Vid", Path("/raid/chen034/data/tapvid")],
        "TAPVid-3D": [ROOT / "data/tapvid3d", ROOT / "data/TAPVid-3D", Path("/raid/chen034/data/tapvid3d")],
        "Kubric": [ROOT / "data/kubric", ROOT / "data/Kubric", Path("/raid/chen034/data/kubric")],
        "MatrixCity": [ROOT / "data/MatrixCity", ROOT / "data/matrixcity", Path("/raid/chen034/data/MatrixCity")],
        "Spring": [ROOT / "data/Spring", ROOT / "data/spring", Path("/raid/chen034/data/Spring")],
    }
    out = {}
    for name, paths in candidates.items():
        existing = [str(p) for p in paths if p.exists()]
        out[name] = {
            "available": bool(existing),
            "existing_paths": existing,
            "exact_access_blocker": None if existing else f"no official local {name} data directory found under checked roots",
        }
    return out


def write_simple_doc(path: Path, title: str, payload: dict[str, Any], keys: list[str]) -> None:
    write_doc(path, title, payload, keys)


def finite_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        return bool(value) and math.isfinite(float(value))
    except Exception:
        return bool(value)

