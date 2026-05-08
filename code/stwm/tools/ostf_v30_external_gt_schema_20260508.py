#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc


DATASET_NAMES = ("pointodyssey", "tapvid", "tapvid3d", "kubric", "spring", "matrixcity")
DATASET_DISPLAY = {
    "pointodyssey": "PointOdyssey v1.2",
    "tapvid": "TAP-Vid",
    "tapvid3d": "TAPVid-3D",
    "kubric": "Kubric / MOVi",
    "spring": "Spring",
    "matrixcity": "MatrixCity",
}
DEFAULT_DATA_ROOTS = (
    ROOT / "data",
    Path("/home/chen034/workspace/stwm/data"),
    Path("/raid/chen034/workspace/data"),
    Path("/home/chen034/workspace/data"),
    Path("/raid/chen034/data"),
    Path("/home/chen034/data"),
)
EXTERNAL_CACHE_ROOT = ROOT / "outputs/cache/stwm_ostf_v30_external_gt"
EXTERNAL_MANIFEST_DIR = ROOT / "manifests/ostf_v30_external_gt"
AGG_CACHE_REPORT = ROOT / "reports/stwm_ostf_v30_external_gt_cache_build_20260508.json"
AGG_CACHE_DOC = ROOT / "docs/STWM_OSTF_V30_EXTERNAL_GT_CACHE_BUILD_20260508.md"
PX_THRESHOLDS = (16.0, 32.0, 64.0, 128.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_roots() -> list[Path]:
    out: list[Path] = []
    seen = set()
    for root in [*DEFAULT_DATA_ROOTS, *[Path(x) for x in os.environ.get("STWM_DATA_ROOTS", "").split(":") if x.strip()]]:
        key = str(root)
        if key not in seen:
            seen.add(key)
            out.append(root)
    return out


def dataset_candidate_names(dataset: str) -> tuple[str, ...]:
    return {
        "pointodyssey": ("pointodyssey", "PointOdyssey", "point_odyssey"),
        "tapvid": ("tapvid", "TAP-Vid", "tap_vid"),
        "tapvid3d": ("tapvid3d", "TAPVid-3D", "tapvid_3d"),
        "kubric": ("kubric", "Kubric"),
        "spring": ("spring", "Spring"),
        "matrixcity": ("matrixcity", "MatrixCity", "matrix_city"),
    }[dataset]


def candidate_paths(dataset: str) -> list[Path]:
    paths = []
    for root in data_roots():
        for name in dataset_candidate_names(dataset):
            paths.append(root / name)
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def manifest_roots() -> list[Path]:
    roots = []
    for root in data_roots():
        roots.append(root / "_manifests")
    roots.append(Path("/home/chen034/workspace/data/_manifests"))
    seen = set()
    out = []
    for path in roots:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def read_related_manifests(dataset: str) -> list[dict[str, Any]]:
    patterns = [
        "*_preflight.json",
        "*_postcheck.json",
        "pointodyssey_hard_complete_after_*.json",
        "tapvid*_postcheck.json",
        "tapvid3d*_postcheck.json",
        "kubric*_postcheck.json",
        "hard_complete_summary_*.json",
    ]
    aliases = {
        "pointodyssey": ("pointodyssey", "hard_complete_summary", "stage1_v2_pointodyssey"),
        "tapvid": ("tapvid",),
        "tapvid3d": ("tapvid3d", "adt", "pstudio", "drivetrack"),
        "kubric": ("kubric", "movi"),
        "spring": ("spring",),
        "matrixcity": ("matrixcity",),
    }[dataset]
    out = []
    seen = set()
    for root in manifest_roots():
        if not root.exists():
            continue
        for pattern in patterns:
            for path in root.glob(pattern):
                lower = path.name.lower()
                if not any(alias in lower for alias in aliases):
                    continue
                if str(path) in seen:
                    continue
                seen.add(str(path))
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    payload = {"parse_error": str(exc)}
                out.append({"path": str(path), "size_bytes": path.stat().st_size, "payload": payload})
    return out


def _fast_dir_stats(path: Path, max_walk_files: int = 20000) -> dict[str, Any]:
    file_count = 0
    total_size = 0
    key_files = []
    split_files = []
    truncated = False
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            file_count += 1
            p = Path(dirpath) / name
            try:
                total_size += p.stat().st_size
            except OSError:
                pass
            lower = name.lower()
            if len(key_files) < 50 and lower.endswith((".npz", ".pkl", ".mp4", ".tfrecord", ".zip", ".json")):
                key_files.append(str(p))
            if len(split_files) < 50 and any(x in lower for x in ("train", "val", "test", "split")):
                split_files.append(str(p))
            if file_count >= max_walk_files:
                truncated = True
                break
        if truncated:
            break
    return {
        "file_count_scanned": file_count,
        "total_size_bytes_scanned": total_size,
        "scan_truncated": truncated,
        "key_files_found": key_files,
        "split_files_found": split_files,
    }


def _manifest_count_size(dataset: str, manifests: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    best_files = None
    best_size = None
    for record in manifests:
        payload = record.get("payload", {})
        for key in ("file_count", "total_file_count", "tfrecord_like_count_total", "trajectory_like_total_files"):
            val = payload.get(key)
            if isinstance(val, int):
                best_files = max(best_files or 0, val)
        for key in ("total_size", "total_size_bytes", "total_size_human"):
            val = payload.get(key)
            if isinstance(val, int):
                best_size = max(best_size or 0, val)
        if dataset in payload and isinstance(payload[dataset], dict):
            sub = payload[dataset]
            if isinstance(sub.get("total_file_count"), int):
                best_files = max(best_files or 0, sub["total_file_count"])
            if isinstance(sub.get("total_size_bytes"), int):
                best_size = max(best_size or 0, sub["total_size_bytes"])
    return best_files, best_size


def audit_dataset(dataset: str) -> dict[str, Any]:
    paths = candidate_paths(dataset)
    existing = [p for p in paths if p.exists()]
    manifests = read_related_manifests(dataset)
    manifest_files, manifest_size = _manifest_count_size(dataset, manifests)
    scan = _fast_dir_stats(existing[0]) if existing and manifest_files is None else {}
    file_count = manifest_files if manifest_files is not None else scan.get("file_count_scanned", 0)
    total_size = manifest_size if manifest_size is not None else scan.get("total_size_bytes_scanned", 0)
    key_files = scan.get("key_files_found", [])
    split_files = scan.get("split_files_found", [])
    if dataset == "pointodyssey" and existing:
        p = existing[0]
        key_files = [str(x) for x in list(p.glob("*/*/anno.npz"))[:20]]
        split_files = [str(p / x) for x in ("train", "val", "test") if (p / x).exists()]
        gt = bool(key_files)
        rgb = bool(list(p.glob("*/*/rgbs")) or list(p.glob("*/*.mp4")))
        vis = gt
        cam = bool(list(p.glob("*/*/info.npz")) or key_files)
        status = "complete" if gt and rgb else "partial"
        blocker = None if status == "complete" else "PointOdyssey path exists but required anno/rgb files are incomplete"
    elif dataset == "tapvid" and existing:
        p = existing[0]
        key_files = [str(x) for x in list(p.glob("**/*.pkl"))[:20]]
        split_files = [str(x) for x in list(p.glob("**/*.txt"))[:20]]
        gt = bool(key_files)
        rgb = gt
        vis = gt
        cam = False
        status = "partial" if gt else "unavailable"
        blocker = "TAP-Vid provides sparse point tracks; not object-dense M128/M512 without additional sampling/teacher" if gt else "no TAP-Vid pkl found"
    elif dataset == "tapvid3d" and existing:
        p = existing[0]
        key_files = [str(x) for x in list(p.glob("**/*.npz"))[:20]]
        split_files = [str(x) for x in list((p / "minival_dataset").glob("*"))[:20]] if (p / "minival_dataset").exists() else []
        gt = bool(key_files)
        rgb = True
        vis = gt
        cam = gt
        status = "partial" if gt else "unavailable"
        blocker = "TAPVid-3D local data is minival/debug style; usable diagnostic, not full train/val/test main benchmark" if gt else "no TAPVid-3D npz found"
    elif dataset == "kubric" and existing:
        p = existing[0]
        key_files = [str(x) for x in list(p.glob("**/*.tfrecord*"))[:20]]
        gt = bool(key_files)
        rgb = gt
        vis = gt
        cam = gt
        status = "partial" if gt else "manual_access_required"
        blocker = "Kubric/MOVi found but no direct OSTF point-trajectory adapter is certified for this preflight" if gt else "Kubric repo/runtime exists but no usable TFDS/tfrecord point cache found"
    else:
        gt = False
        rgb = False
        vis = False
        cam = False
        status = "unavailable" if not existing else "unknown"
        blocker = f"{DATASET_DISPLAY[dataset]} not found under scanned roots" if not existing else "path exists but dataset-specific GT layout is unknown"
    return {
        "dataset": DATASET_DISPLAY[dataset],
        "dataset_key": dataset,
        "available": bool(existing),
        "candidate_paths": [str(p) for p in existing],
        "searched_paths": [str(p) for p in paths],
        "official_source_expected": {
            "pointodyssey": "PointOdyssey official v1.2 release",
            "tapvid": "TAP-Vid official pkl releases",
            "tapvid3d": "TAPVid-3D official benchmark npz releases",
            "kubric": "Kubric/MOVi official generated TFDS/tfrecord data",
            "spring": "Spring official benchmark release",
            "matrixcity": "MatrixCity official release",
        }[dataset],
        "file_count": int(file_count or 0),
        "total_size_bytes": int(total_size or 0),
        "key_files_found": key_files[:50],
        "split_files_found": split_files[:50],
        "gt_point_trajectories_available": bool(gt),
        "visibility_or_occlusion_available": bool(vis),
        "rgb_frames_or_video_available": bool(rgb),
        "camera_intrinsics_extrinsics_available": bool(cam),
        "license_or_manual_access_required": status == "manual_access_required",
        "completeness_status": status,
        "exact_blocker": blocker,
        "related_manifest_paths": [m["path"] for m in manifests],
    }


def load_external_audit() -> dict[str, Any]:
    path = ROOT / "reports/stwm_ostf_v30_external_gt_data_root_audit_20260508.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_cache_report(dataset_key: str, payload: dict[str, Any]) -> None:
    current = json.loads(AGG_CACHE_REPORT.read_text(encoding="utf-8")) if AGG_CACHE_REPORT.exists() else {
        "report_name": "stwm_ostf_v30_external_gt_cache_build",
        "generated_at_utc": utc_now(),
        "datasets": {},
    }
    current["generated_at_utc"] = utc_now()
    current.setdefault("datasets", {})[dataset_key] = payload
    ready = [
        key
        for key, val in current.get("datasets", {}).items()
        if val.get("cache_ready") or val.get("partial_cache_ready")
    ]
    current["external_gt_cache_ready"] = any(current.get("datasets", {}).get(key, {}).get("cache_ready") for key in current.get("datasets", {}))
    current["partial_usable_datasets"] = ready
    dump_json(AGG_CACHE_REPORT, current)
    write_doc(
        AGG_CACHE_DOC,
        "STWM OSTF V30 External GT Cache Build",
        current,
        ["external_gt_cache_ready", "partial_usable_datasets"],
    )


def choose_indices(valid_mask: np.ndarray, m: int) -> np.ndarray:
    idx = np.flatnonzero(valid_mask)
    if idx.size < m:
        return np.asarray([], dtype=np.int64)
    if idx.size == m:
        return idx.astype(np.int64)
    pos = np.linspace(0, idx.size - 1, m).round().astype(np.int64)
    return idx[pos].astype(np.int64)


def safe_norm_points(points: np.ndarray) -> np.ndarray:
    return np.nan_to_num(points.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)


def save_ostf_npz(
    path: Path,
    *,
    video_uid: str,
    dataset: str,
    split: str,
    frame_paths: list[str],
    obs_points: np.ndarray,
    fut_points: np.ndarray,
    obs_vis: np.ndarray,
    fut_vis: np.ndarray,
    obs_conf: np.ndarray | None = None,
    fut_conf: np.ndarray | None = None,
    obs_points_3d: np.ndarray | None = None,
    fut_points_3d: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    extrinsics: np.ndarray | None = None,
    point_sampling_method: str,
    source_path: str,
    coordinate_system: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    obs_conf_arr = obs_conf if obs_conf is not None else obs_vis.astype(np.float32)
    fut_conf_arr = fut_conf if fut_conf is not None else fut_vis.astype(np.float32)
    point_ids = np.arange(obs_points.shape[0], dtype=np.int64)
    kwargs: dict[str, Any] = {
        "video_uid": np.asarray(video_uid),
        "dataset": np.asarray(dataset),
        "split": np.asarray(split),
        "frame_paths": np.asarray(frame_paths, dtype=object),
        "obs_points": safe_norm_points(obs_points),
        "fut_points": safe_norm_points(fut_points),
        "obs_vis": obs_vis.astype(bool),
        "fut_vis": fut_vis.astype(bool),
        "obs_conf": obs_conf_arr.astype(np.float32),
        "fut_conf": fut_conf_arr.astype(np.float32),
        "point_id": point_ids,
        "object_id": np.asarray(-1, dtype=np.int64),
        "semantic_id": np.asarray(-1, dtype=np.int64),
        "instance_id": np.asarray(-1, dtype=np.int64),
        "point_sampling_method": np.asarray(point_sampling_method),
        "no_future_leakage": np.asarray(True),
        "source_gt_not_teacher": np.asarray(True),
        "source_path": np.asarray(source_path),
        "coordinate_system": np.asarray(coordinate_system),
        "M": np.asarray(obs_points.shape[0], dtype=np.int64),
        "obs_len": np.asarray(obs_points.shape[1], dtype=np.int64),
        "horizon": np.asarray(fut_points.shape[1], dtype=np.int64),
        "valid_future_point_ratio": np.asarray(float(fut_vis.mean()) if fut_vis.size else 0.0, dtype=np.float32),
    }
    if obs_points_3d is not None:
        kwargs["obs_points_3d"] = obs_points_3d.astype(np.float32)
    if fut_points_3d is not None:
        kwargs["fut_points_3d"] = fut_points_3d.astype(np.float32)
    if intrinsics is not None:
        kwargs["intrinsics"] = intrinsics.astype(np.float32)
    if extrinsics is not None:
        kwargs["extrinsics"] = extrinsics.astype(np.float32)
    np.savez_compressed(path, **kwargs)


@dataclass
class ExternalSample:
    uid: str
    dataset: str
    split: str
    cache_path: str
    frame_paths: list[str]
    obs_points: np.ndarray
    fut_points: np.ndarray
    obs_vis: np.ndarray
    fut_vis: np.ndarray
    obs_conf: np.ndarray
    fut_conf: np.ndarray
    coordinate_system: str


def load_external_sample(path: Path) -> ExternalSample:
    z = np.load(path, allow_pickle=True)
    return ExternalSample(
        uid=str(np.asarray(z["video_uid"]).item()),
        dataset=str(np.asarray(z["dataset"]).item()),
        split=str(np.asarray(z["split"]).item()),
        cache_path=str(path.relative_to(ROOT)),
        frame_paths=[str(x) for x in np.asarray(z["frame_paths"], dtype=object).tolist()],
        obs_points=np.asarray(z["obs_points"], dtype=np.float32),
        fut_points=np.asarray(z["fut_points"], dtype=np.float32),
        obs_vis=np.asarray(z["obs_vis"]).astype(bool),
        fut_vis=np.asarray(z["fut_vis"]).astype(bool),
        obs_conf=np.asarray(z["obs_conf"], dtype=np.float32),
        fut_conf=np.asarray(z["fut_conf"], dtype=np.float32),
        coordinate_system=str(np.asarray(z["coordinate_system"]).item()),
    )


def discover_external_cache_files(dataset: str | None = None, combo: str | None = None, split: str | None = None) -> list[Path]:
    root = EXTERNAL_CACHE_ROOT
    if dataset:
        root = root / dataset
    if combo:
        root = root / combo
    if split:
        root = root / split
    return sorted(root.rglob("*.npz")) if root.exists() else []


def point_metrics(sample: ExternalSample, pred: np.ndarray) -> dict[str, Any]:
    valid = sample.fut_vis
    err = np.abs(pred - sample.fut_points).sum(axis=-1)
    all_err = err[valid] if np.any(valid) else np.asarray([0.0])
    if np.any(valid[:, -1]):
        endpoint = err[:, -1][valid[:, -1]]
    else:
        endpoint = all_err
    fde = float(endpoint.mean()) if endpoint.size else 0.0
    ade = float(all_err.mean()) if all_err.size else 0.0
    out = {
        "minADE": ade,
        "minFDE": fde,
        "visibility_F1": visibility_f1(sample.fut_vis, np.repeat(sample.obs_vis[:, -1:], sample.fut_points.shape[1], axis=1)),
        "relative_deformation_layout_error": relative_layout_error(sample, pred),
    }
    for thr in PX_THRESHOLDS:
        out[f"MissRate@{int(thr)}"] = float(fde > thr)
        out[f"PCK@{int(thr)}"] = float((all_err < thr).mean()) if all_err.size else 0.0
    out["threshold_auc_endpoint_16_32_64_128"] = float(np.mean([1.0 - out[f"MissRate@{int(thr)}"] for thr in PX_THRESHOLDS]))
    return out


def visibility_f1(gt: np.ndarray, pred: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return float(2.0 * prec * rec / max(prec + rec, 1e-8))


def relative_layout_error(sample: ExternalSample, pred: np.ndarray) -> float:
    vals = []
    for t in range(sample.fut_points.shape[1]):
        valid = sample.fut_vis[:, t]
        if int(valid.sum()) < 2:
            continue
        p = pred[valid, t]
        g = sample.fut_points[valid, t]
        vals.append(float(np.abs((p - p.mean(axis=0)) - (g - g.mean(axis=0))).sum(axis=-1).mean()))
    return float(np.mean(vals)) if vals else 0.0


def prior_predictions(sample: ExternalSample, gamma: float = 0.0) -> dict[str, np.ndarray]:
    last = sample.obs_points[:, -1]
    last_visible = last.copy()
    vel_visible = np.zeros_like(last)
    for i in range(sample.obs_points.shape[0]):
        idx = np.flatnonzero(sample.obs_vis[i])
        if idx.size:
            last_idx = int(idx[-1])
            last_visible[i] = sample.obs_points[i, last_idx]
            if idx.size >= 2:
                prev_idx = int(idx[-2])
                vel_visible[i] = (sample.obs_points[i, last_idx] - sample.obs_points[i, prev_idx]) / max(float(last_idx - prev_idx), 1.0)
    velocity = sample.obs_points[:, -1] - sample.obs_points[:, -2]
    h = sample.fut_points.shape[1]
    t = np.arange(1, h + 1, dtype=np.float32)[None, :, None]
    anchor = np.median(last_visible, axis=0, keepdims=True)
    layout = last_visible - anchor
    return {
        "last_observed_copy": np.repeat(last[:, None, :], h, axis=1),
        "last_visible_copy": np.repeat(last_visible[:, None, :], h, axis=1),
        "visibility_aware_damped": last_visible[:, None, :] + gamma * vel_visible[:, None, :] * t,
        "visibility_aware_cv": last_visible[:, None, :] + vel_visible[:, None, :] * t,
        "constant_velocity": last[:, None, :] + velocity[:, None, :] * t,
        "median_object_anchor_copy": np.repeat((anchor[:, None, :] + layout[:, None, :]), h, axis=1),
        "fixed_affine": last[:, None, :] + 0.25 * velocity[:, None, :] * t,
    }


def aggregate_metric_rows(rows: list[dict[str, Any]], subset_key: str | None = None, dataset: str | None = None, horizon: int | None = None) -> dict[str, Any]:
    filt = []
    for r in rows:
        if subset_key and not r.get(subset_key, False):
            continue
        if dataset and r.get("dataset") != dataset:
            continue
        if horizon and int(r.get("H", 0)) != int(horizon):
            continue
        filt.append(r)
    out = {"item_count": len(filt)}
    for key in [
        "minADE",
        "minFDE",
        "MissRate@16",
        "MissRate@32",
        "MissRate@64",
        "MissRate@128",
        "threshold_auc_endpoint_16_32_64_128",
        "PCK@8",
        "PCK@16",
        "PCK@32",
        "PCK@64",
        "visibility_F1",
        "relative_deformation_layout_error",
    ]:
        vals = [float(r[key]) for r in filt if r.get(key) is not None and np.isfinite(float(r[key]))]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def paired_bootstrap(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str, higher_better: bool, subset_key: str | None = None, n_boot: int = 1000) -> dict[str, Any]:
    amap = {r["uid"]: r for r in rows_a}
    bmap = {r["uid"]: r for r in rows_b}
    vals = []
    for key in sorted(set(amap) & set(bmap)):
        a = amap[key]
        b = bmap[key]
        if subset_key and (not a.get(subset_key, False) or not b.get(subset_key, False)):
            continue
        if a.get(metric) is None or b.get(metric) is None:
            continue
        delta = float(a[metric]) - float(b[metric])
        vals.append(delta if higher_better else -delta)
    if not vals:
        return {"item_count": 0, "mean_delta": None, "ci95": [None, None], "zero_excluded": False}
    arr = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, arr.size, size=arr.size)
        means.append(float(arr[idx].mean()))
    lo, hi = np.percentile(np.asarray(means), [2.5, 97.5]).tolist()
    return {"item_count": int(arr.size), "mean_delta": float(arr.mean()), "ci95": [float(lo), float(hi)], "zero_excluded": bool(lo > 0 or hi < 0)}
