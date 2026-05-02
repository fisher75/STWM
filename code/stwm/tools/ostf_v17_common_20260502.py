#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
V16_AUDIT_PATH = ROOT / "reports/stwm_real_object_dense_teacher_cache_audit_v16_20260502.json"
VIS_V16_PATH = ROOT / "reports/stwm_real_teacher_object_dense_visualization_v16_20260502.json"


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, title: str, payload: dict[str, Any], keys: list[str]) -> None:
    lines = [f"# {title}", ""]
    for key in keys:
        if key in payload:
            lines.append(f"- {key}: `{jsonable(payload[key])}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class OSTFObjectSample:
    item_key: str
    dataset: str
    split: str
    source_cache_path: str
    object_index: int
    object_id: int
    m: int
    h: int
    obs_points: np.ndarray  # [M,Tobs,2] normalized
    fut_points: np.ndarray  # [M,H,2] normalized
    obs_vis: np.ndarray  # [M,Tobs]
    fut_vis: np.ndarray  # [M,H]
    rel_xy: np.ndarray  # [M,2]
    anchor_obs: np.ndarray  # [Tobs,2]
    anchor_fut: np.ndarray  # [H,2]
    anchor_obs_vel: np.ndarray  # [Tobs,2]
    semantic_feat: np.ndarray  # [F]
    semantic_valid: bool
    semantic_id: int
    proto_target: int = -1


def _scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _object_rel(query_points: np.ndarray) -> np.ndarray:
    pts = query_points.astype(np.float32)
    mn = pts.min(axis=0, keepdims=True)
    mx = pts.max(axis=0, keepdims=True)
    return ((pts - mn) / np.maximum(mx - mn, 1.0)).astype(np.float32)


def _anchor_from_points(points: np.ndarray, vis: np.ndarray) -> np.ndarray:
    w = vis.astype(np.float32)
    denom = np.maximum(w.sum(axis=0), 1.0)
    return ((points * w[..., None]).sum(axis=0) / denom[:, None]).astype(np.float32)


def _velocity(xy_t_2: np.ndarray) -> np.ndarray:
    vel = np.zeros_like(xy_t_2, dtype=np.float32)
    if xy_t_2.shape[0] >= 2:
        vel[1:] = xy_t_2[1:] - xy_t_2[:-1]
    return vel


def load_v16_samples(
    combo: str,
    *,
    limit_per_split: dict[str, int] | None = None,
    restrict_item_keys: set[str] | dict[str, set[str]] | None = None,
) -> dict[str, list[OSTFObjectSample]]:
    combo_root = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16" / combo
    out: dict[str, list[OSTFObjectSample]] = {"train": [], "val": [], "test": []}
    if not combo_root.exists():
        return out
    for cache_path in sorted(combo_root.glob("*/*.npz")):
        z = np.load(cache_path, allow_pickle=True)
        split = str(_scalar(z["split"]))
        if split not in out:
            continue
        item_key = str(_scalar(z["item_key"]))
        if restrict_item_keys is not None:
            if isinstance(restrict_item_keys, dict):
                if item_key not in restrict_item_keys.get(split, set()):
                    continue
            elif item_key not in restrict_item_keys:
                continue
        if limit_per_split and len(out[split]) >= limit_per_split.get(split, 10**12):
            continue
        tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
        vis = np.asarray(z["visibility"]).astype(bool)
        query = np.asarray(z["query_points_xy"], dtype=np.float32)
        object_ids = np.asarray(z.get("object_id", np.arange(tracks.shape[0])), dtype=np.int64)
        raw_size = np.asarray(z["raw_size"], dtype=np.float32)
        pre = np.load(str(_scalar(z["predecode_path"])), allow_pickle=True)
        sem_feat_all = np.asarray(pre["semantic_features"], dtype=np.float32)
        sem_ids_all = np.asarray(pre.get("semantic_entity_dominant_instance_id", np.full((sem_feat_all.shape[0],), -1, dtype=np.int64)), dtype=np.int64)
        scale = float(max(raw_size.tolist()))
        obs_len = int(_scalar(z["obs_len"]))
        horizon = int(_scalar(z["horizon"]))
        for obj in range(tracks.shape[0]):
            sem_idx = obj if obj < sem_feat_all.shape[0] else sem_feat_all.shape[0] - 1
            points = tracks[obj] / scale
            visibility = vis[obj]
            anchor_obs = _anchor_from_points(points[:, :obs_len], visibility[:, :obs_len]).astype(np.float32)
            anchor_fut = _anchor_from_points(points[:, obs_len : obs_len + horizon], visibility[:, obs_len : obs_len + horizon]).astype(np.float32)
            out[split].append(
                OSTFObjectSample(
                    item_key=str(_scalar(z["item_key"])),
                    dataset=str(_scalar(z["dataset"])),
                    split=split,
                    source_cache_path=str(cache_path.relative_to(ROOT)),
                    object_index=int(obj),
                    object_id=int(object_ids[obj]) if obj < object_ids.shape[0] else int(obj),
                    m=int(points.shape[0]),
                    h=horizon,
                    obs_points=points[:, :obs_len].astype(np.float32),
                    fut_points=points[:, obs_len : obs_len + horizon].astype(np.float32),
                    obs_vis=visibility[:, :obs_len].astype(bool),
                    fut_vis=visibility[:, obs_len : obs_len + horizon].astype(bool),
                    rel_xy=_object_rel(query[obj]).astype(np.float32),
                    anchor_obs=anchor_obs,
                    anchor_fut=anchor_fut,
                    anchor_obs_vel=_velocity(anchor_obs),
                    semantic_feat=sem_feat_all[sem_idx].astype(np.float32),
                    semantic_valid=bool(np.linalg.norm(sem_feat_all[sem_idx]) > 1e-6),
                    semantic_id=int(sem_ids_all[sem_idx]) if sem_idx < len(sem_ids_all) else -1,
                )
            )
    return out


def collapse_to_m1(rows: dict[str, list[OSTFObjectSample]]) -> dict[str, list[OSTFObjectSample]]:
    out: dict[str, list[OSTFObjectSample]] = {"train": [], "val": [], "test": []}
    for split, samples in rows.items():
        for s in samples:
            new = OSTFObjectSample(
                item_key=s.item_key,
                dataset=s.dataset,
                split=s.split,
                source_cache_path=s.source_cache_path,
                object_index=s.object_index,
                object_id=s.object_id,
                m=1,
                h=s.h,
                obs_points=s.anchor_obs[None].copy(),
                fut_points=s.anchor_fut[None].copy(),
                obs_vis=np.ones((1, s.anchor_obs.shape[0]), dtype=bool),
                fut_vis=np.ones((1, s.anchor_fut.shape[0]), dtype=bool),
                rel_xy=np.asarray([[0.5, 0.5]], dtype=np.float32),
                anchor_obs=s.anchor_obs.copy(),
                anchor_fut=s.anchor_fut.copy(),
                anchor_obs_vel=s.anchor_obs_vel.copy(),
                semantic_feat=s.semantic_feat.copy(),
                semantic_valid=s.semantic_valid,
                semantic_id=s.semantic_id,
                proto_target=s.proto_target,
            )
            out[split].append(new)
    return out


def common_item_keys_for_horizon(horizon: int) -> dict[str, set[str]]:
    base = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16"
    item_sets: list[dict[str, set[str]]] = []
    for combo in [f"M128_H{horizon}", f"M512_H{horizon}"]:
        combo_root = base / combo
        keys = {"train": set(), "val": set(), "test": set()}
        for path in sorted(combo_root.glob("*/*.npz")):
            z = np.load(path, allow_pickle=True)
            split = str(_scalar(z["split"]))
            if split in keys:
                keys[split].add(str(_scalar(z["item_key"])))
        item_sets.append(keys)
    if not item_sets:
        return {"train": set(), "val": set(), "test": set()}
    common = {split: set(keys) for split, keys in item_sets[0].items()}
    for s in item_sets[1:]:
        for split in common:
            common[split] &= s.get(split, set())
    return common


def kmeans_semantic_prototypes(train_samples: list[OSTFObjectSample], k: int = 32, iters: int = 20, seed: int = 42) -> np.ndarray:
    feats = np.stack([s.semantic_feat for s in train_samples if s.semantic_valid], axis=0)
    if feats.shape[0] == 0:
        return np.zeros((k, 10), dtype=np.float32)
    rng = np.random.default_rng(seed)
    if feats.shape[0] < k:
        idx = rng.choice(feats.shape[0], size=k, replace=True)
    else:
        idx = rng.choice(feats.shape[0], size=k, replace=False)
    centers = feats[idx].copy()
    for _ in range(iters):
        dist = ((feats[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        assign = dist.argmin(axis=1)
        for j in range(k):
            mask = assign == j
            if mask.any():
                centers[j] = feats[mask].mean(axis=0)
    return centers.astype(np.float32)


def assign_semantic_prototypes(rows: dict[str, list[OSTFObjectSample]], centers: np.ndarray) -> None:
    for samples in rows.values():
        for s in samples:
            if s.semantic_valid:
                dist = ((centers - s.semantic_feat[None]) ** 2).sum(axis=-1)
                s.proto_target = int(dist.argmin())
            else:
                s.proto_target = 0


def batch_from_samples(samples: list[OSTFObjectSample], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "obs_points": torch.tensor(np.stack([s.obs_points for s in samples]), device=device, dtype=torch.float32),
        "obs_vis": torch.tensor(np.stack([s.obs_vis for s in samples]), device=device, dtype=torch.bool),
        "fut_points": torch.tensor(np.stack([s.fut_points for s in samples]), device=device, dtype=torch.float32),
        "fut_vis": torch.tensor(np.stack([s.fut_vis for s in samples]), device=device, dtype=torch.bool),
        "rel_xy": torch.tensor(np.stack([s.rel_xy for s in samples]), device=device, dtype=torch.float32),
        "anchor_obs": torch.tensor(np.stack([s.anchor_obs for s in samples]), device=device, dtype=torch.float32),
        "anchor_fut": torch.tensor(np.stack([s.anchor_fut for s in samples]), device=device, dtype=torch.float32),
        "anchor_obs_vel": torch.tensor(np.stack([s.anchor_obs_vel for s in samples]), device=device, dtype=torch.float32),
        "semantic_feat": torch.tensor(np.stack([s.semantic_feat for s in samples]), device=device, dtype=torch.float32),
        "proto_target": torch.tensor([s.proto_target for s in samples], device=device, dtype=torch.long),
    }


def iter_batches(samples: list[OSTFObjectSample], batch_size: int, *, shuffle: bool, seed: int) -> list[list[OSTFObjectSample]]:
    idx = list(range(len(samples)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(idx)
    return [[samples[i] for i in idx[start : start + batch_size]] for start in range(0, len(idx), batch_size)]


def anchor_extent_iou(pred_points: np.ndarray, gt_points: np.ndarray, valid_mask: np.ndarray) -> float:
    vals = []
    for b in range(pred_points.shape[0]):
        for t in range(pred_points.shape[2]):
            valid = valid_mask[b, :, t]
            if not np.any(valid):
                continue
            pred = pred_points[b, valid, t]
            gt = gt_points[b, valid, t]
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
            vals.append(float(inter / union)) if union > 0 else vals.append(0.0)
    return float(np.mean(vals)) if vals else 0.0


def eval_metrics_from_predictions(
    *,
    pred_points: np.ndarray,  # [B,M,H,2]
    pred_vis_logits: np.ndarray | None,  # [B,M,H]
    pred_proto_logits: np.ndarray | None,  # [B,H,K]
    gt_points: np.ndarray,
    gt_vis: np.ndarray,
    gt_anchor: np.ndarray,
    proto_target: np.ndarray,
) -> dict[str, Any]:
    err_px = np.abs(pred_points - gt_points).sum(axis=-1) * 1000.0
    valid = gt_vis.astype(bool)
    point_l1 = float(err_px[valid].mean()) if np.any(valid) else 0.0
    endpoint = float(err_px[:, :, -1][gt_vis[:, :, -1]].mean()) if np.any(gt_vis[:, :, -1]) else point_l1
    pck4 = float((err_px[valid] < 4.0).mean()) if np.any(valid) else 0.0
    pck8 = float((err_px[valid] < 8.0).mean()) if np.any(valid) else 0.0
    pck16 = float((err_px[valid] < 16.0).mean()) if np.any(valid) else 0.0
    pred_anchor = pred_points.mean(axis=1)
    anchor_l1 = float((np.abs(pred_anchor - gt_anchor).sum(axis=-1) * 1000.0).mean())
    extent_iou = anchor_extent_iou(pred_points, gt_points, valid)
    if pred_vis_logits is not None:
        pred_vis = pred_vis_logits > 0.0
        tp = int(np.logical_and(pred_vis, gt_vis).sum())
        fp = int(np.logical_and(pred_vis, np.logical_not(gt_vis)).sum())
        fn = int(np.logical_and(np.logical_not(pred_vis), gt_vis).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        vis_f1 = float(2 * prec * rec / max(prec + rec, 1e-8))
    else:
        vis_f1 = None
    semantic = {
        "semantic_changed_top5": None,
        "stable_semantic_preservation": None,
        "semantic_metric_note": "future semantic target is constant per object prototype in current V17 cache; changed subset is not a strong signal here",
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
    return {
        "point_L1_px": point_l1,
        "endpoint_error_px": endpoint,
        "PCK_4px": pck4,
        "PCK_8px": pck8,
        "PCK_16px": pck16,
        "visibility_F1": vis_f1,
        "anchor_centroid_L1_px": anchor_l1,
        "object_extent_iou": extent_iou,
        **semantic,
    }


def verify_v16_cache() -> dict[str, Any]:
    audit = load_json(V16_AUDIT_PATH)
    vis = load_json(VIS_V16_PATH)
    combos = audit.get("per_combo", {})
    required = ["M128_H8", "M128_H16", "M512_H8", "M512_H16"]
    combo_exists = {k: bool(combos.get(k, {}).get("processed_clip_count", 0) > 0) for k in required}
    result = {
        "cache_exists": all(combo_exists.values()),
        "combo_exists": combo_exists,
        "processed_clip_count": int(audit.get("processed_clip_count", 0)),
        "valid_point_ratio": float(audit.get("valid_point_ratio", 0.0)),
        "teacher_source": "cotracker_official" if audit.get("real_teacher_tracks_exist") else "missing",
        "persistent_point_identity_valid": bool(audit.get("persistent_point_identity_valid", False)),
        "fake_dense_or_anchor_copied": bool(audit.get("fake_dense_or_anchor_copied", True)),
        "raw_visualization_exists": bool(vis.get("visualization_ready", False)),
        "future_leakage_audit_passed": bool(audit.get("future_leakage_audit_passed", False)),
    }
    result["cache_verification_passed"] = (
        result["cache_exists"]
        and result["processed_clip_count"] >= 2000
        and result["valid_point_ratio"] > 0.5
        and result["teacher_source"] == "cotracker_official"
        and result["persistent_point_identity_valid"]
        and not result["fake_dense_or_anchor_copied"]
        and result["raw_visualization_exists"]
        and result["future_leakage_audit_passed"]
    )
    return result


def bootstrap_delta(a: np.ndarray, b: np.ndarray, *, n_boot: int = 2000, seed: int = 42) -> dict[str, Any]:
    assert a.shape == b.shape
    rng = np.random.default_rng(seed)
    delta = a - b
    n = delta.shape[0]
    means = np.empty((n_boot,), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(delta[idx].mean())
    lo, hi = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "mean_delta": float(delta.mean()),
        "ci95": [float(lo), float(hi)],
        "zero_excluded": bool(lo > 0 or hi < 0),
        "item_count": int(n),
    }
