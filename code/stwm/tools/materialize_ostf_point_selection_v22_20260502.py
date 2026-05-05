#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import OSTFObjectSample, ROOT, dump_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import build_v18_rows


def _seed_from_sample(sample: OSTFObjectSample, seed: int) -> int:
    base = abs(hash((sample.item_key, int(sample.object_id), int(sample.object_index), int(seed)))) % (2**32 - 1)
    return int(base)


def _point_motion_score(sample: OSTFObjectSample) -> np.ndarray:
    vel = np.diff(sample.obs_points, axis=1, prepend=sample.obs_points[:, :1])
    speed = np.linalg.norm(vel, axis=-1)
    return speed.mean(axis=1).astype(np.float32)


def _point_visibility_score(sample: OSTFObjectSample) -> np.ndarray:
    return sample.obs_vis.astype(np.float32).mean(axis=1)


def _boundary_distance(rel_xy: np.ndarray) -> np.ndarray:
    rel = np.asarray(rel_xy, dtype=np.float32)
    return np.minimum.reduce([rel[:, 0], 1.0 - rel[:, 0], rel[:, 1], 1.0 - rel[:, 1]]).astype(np.float32)


def _fps_indices(points: np.ndarray, target_m: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] <= target_m:
        return np.arange(pts.shape[0], dtype=np.int64)
    picked = [int(np.argmax(np.linalg.norm(pts - pts.mean(axis=0, keepdims=True), axis=-1)))]
    min_dist = np.full((pts.shape[0],), np.inf, dtype=np.float32)
    while len(picked) < target_m:
        last = pts[picked[-1]]
        d = np.linalg.norm(pts - last[None], axis=-1)
        min_dist = np.minimum(min_dist, d)
        next_idx = int(np.argmax(min_dist))
        if next_idx in picked:
            break
        picked.append(next_idx)
    if len(picked) < target_m:
        remain = [i for i in range(pts.shape[0]) if i not in picked]
        picked.extend(remain[: target_m - len(picked)])
    return np.asarray(sorted(picked[:target_m]), dtype=np.int64)


def point_metadata(sample: OSTFObjectSample) -> np.ndarray:
    rel = np.asarray(sample.rel_xy, dtype=np.float32)
    motion = _point_motion_score(sample)
    vis = _point_visibility_score(sample)
    bdist = _boundary_distance(rel)
    interior = bdist
    boundary = 1.0 - np.clip(bdist / max(float(bdist.max()), 1e-6), 0.0, 1.0)
    radius = np.linalg.norm(rel - 0.5, axis=-1).astype(np.float32)
    accel = np.diff(sample.obs_points, n=2, axis=1)
    accel_mag = (
        np.linalg.norm(accel, axis=-1).mean(axis=1).astype(np.float32)
        if accel.shape[1] > 0
        else np.zeros((sample.obs_points.shape[0],), dtype=np.float32)
    )
    return np.stack([motion, vis, boundary, interior, radius, accel_mag], axis=-1).astype(np.float32)


def select_point_indices(sample: OSTFObjectSample, target_m: int, strategy: str, seed: int = 42) -> np.ndarray:
    m = int(sample.obs_points.shape[0])
    if target_m >= m:
        return np.arange(m, dtype=np.int64)
    rel = np.asarray(sample.rel_xy, dtype=np.float32)
    motion = _point_motion_score(sample)
    vis = _point_visibility_score(sample)
    bdist = _boundary_distance(rel)
    rng = np.random.default_rng(_seed_from_sample(sample, seed))

    if strategy == "uniform":
        idx = np.linspace(0, m - 1, num=target_m, dtype=np.int64)
    elif strategy == "random_uniform":
        idx = np.sort(rng.choice(m, size=target_m, replace=False)).astype(np.int64)
    elif strategy == "boundary_only":
        idx = np.argsort(bdist)[:target_m].astype(np.int64)
    elif strategy == "interior_only":
        idx = np.argsort(-bdist)[:target_m].astype(np.int64)
    elif strategy == "visibility_stable":
        idx = np.argsort(-vis)[:target_m].astype(np.int64)
    elif strategy == "high_motion":
        idx = np.argsort(-motion)[:target_m].astype(np.int64)
    elif strategy == "farthest":
        idx = _fps_indices(rel, target_m)
    elif strategy == "boundary_interior":
        half = target_m // 2
        boundary_idx = np.argsort(bdist)[:half]
        interior_idx = np.argsort(-bdist)[: target_m - half]
        idx = np.unique(np.concatenate([boundary_idx, interior_idx], axis=0))
        if idx.shape[0] < target_m:
            remain = [i for i in range(m) if i not in set(idx.tolist())]
            idx = np.concatenate([idx, np.asarray(remain[: target_m - idx.shape[0]], dtype=np.int64)], axis=0)
        idx = np.sort(idx[:target_m]).astype(np.int64)
    elif strategy == "high_motion_visibility":
        score = motion + 0.35 * vis
        idx = np.argsort(-score)[:target_m].astype(np.int64)
    else:
        raise ValueError(f"Unknown point selection strategy: {strategy}")
    return np.asarray(np.sort(idx), dtype=np.int64)


def subset_sample_points(
    sample: OSTFObjectSample,
    target_m: int,
    strategy: str,
    *,
    seed: int = 42,
) -> OSTFObjectSample:
    idx = select_point_indices(sample, target_m, strategy, seed=seed)
    return replace(
        sample,
        m=int(idx.shape[0]),
        obs_points=sample.obs_points[idx].copy(),
        fut_points=sample.fut_points[idx].copy(),
        obs_vis=sample.obs_vis[idx].copy(),
        fut_vis=sample.fut_vis[idx].copy(),
        rel_xy=sample.rel_xy[idx].copy(),
    )


def shuffle_point_identities(sample: OSTFObjectSample, seed: int = 42) -> OSTFObjectSample:
    rng = np.random.default_rng(_seed_from_sample(sample, seed))
    idx = np.arange(sample.m, dtype=np.int64)
    rng.shuffle(idx)
    return replace(
        sample,
        obs_points=sample.obs_points[idx].copy(),
        obs_vis=sample.obs_vis[idx].copy(),
        rel_xy=sample.rel_xy[idx].copy(),
    )


def derive_rows_with_point_selection(
    rows: dict[str, list[OSTFObjectSample]],
    *,
    target_m: int,
    strategy: str,
    seed: int = 42,
) -> dict[str, list[OSTFObjectSample]]:
    out: dict[str, list[OSTFObjectSample]] = {"train": [], "val": [], "test": []}
    for split, samples in rows.items():
        out[split] = [subset_sample_points(s, target_m, strategy, seed=seed) for s in samples]
    return out


def audit_point_selection(
    *,
    combo: str = "M512_H8",
    target_ms: tuple[int, ...] = (1, 32, 128, 256, 512),
    strategies: tuple[str, ...] = ("uniform", "boundary_interior", "high_motion", "visibility_stable", "farthest"),
    seed: int = 42,
) -> dict[str, Any]:
    rows, _ = build_v18_rows(combo, seed=seed)
    train = rows["train"]
    out: dict[str, Any] = {"audit_name": "stwm_ostf_point_selection_v22", "combo": combo, "strategies": {}}
    for strategy in strategies:
        strat_payload: dict[str, Any] = {}
        for target_m in target_ms:
            derived = [subset_sample_points(s, target_m, strategy, seed=seed) for s in train[: min(len(train), 256)]]
            meta = np.concatenate([point_metadata(s) for s in derived], axis=0) if derived else np.zeros((0, 6), dtype=np.float32)
            strat_payload[f"M{target_m}"] = {
                "sample_count": int(len(derived)),
                "mean_points_per_object": float(np.mean([s.m for s in derived])) if derived else 0.0,
                "motion_score_mean": float(meta[:, 0].mean()) if meta.size else 0.0,
                "visibility_score_mean": float(meta[:, 1].mean()) if meta.size else 0.0,
                "boundary_score_mean": float(meta[:, 2].mean()) if meta.size else 0.0,
                "interior_score_mean": float(meta[:, 3].mean()) if meta.size else 0.0,
            }
        out["strategies"][strategy] = strat_payload
    return out


def main() -> int:
    payload = audit_point_selection()
    report = ROOT / "reports/stwm_ostf_point_selection_v22_20260502.json"
    doc = ROOT / "docs/STWM_OSTF_POINT_SELECTION_V22_20260502.md"
    dump_json(report, payload)
    write_doc(
        doc,
        "STWM OSTF V22 Point Selection Audit",
        payload,
        ["audit_name", "combo"],
    )
    print(report.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
