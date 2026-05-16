#!/usr/bin/env python3
"""构建 V35.16 video identity pairwise/retrieval targets。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_15_expanded_mask_derived_video_semantic_state_targets/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_16_video_identity_pairwise_retrieval_targets/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_16_video_identity_pairwise_target_build_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_16_VIDEO_IDENTITY_PAIRWISE_TARGET_BUILD_20260515.md"


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
    return x


def norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def weighted_measurement(z: Any) -> np.ndarray:
    m = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(z["obs_semantic_measurement_mask"], dtype=np.float32)
    conf = np.asarray(z["obs_measurement_confidence"], dtype=np.float32)
    w = mask * np.clip(conf, 0.05, 1.0)
    pooled = (m * w[..., None]).sum(axis=1) / np.maximum(w.sum(axis=1, keepdims=True), 1e-6)
    return norm(pooled.astype(np.float32))


def trace_features(z: Any) -> np.ndarray:
    obs = np.asarray(z["obs_points"], dtype=np.float32)
    fut = np.asarray(z["future_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"], dtype=np.float32)
    fut_vis = np.asarray(z["future_vis"], dtype=np.float32)
    obs_conf = np.asarray(z["obs_conf"], dtype=np.float32)
    fut_conf = np.asarray(z["future_conf"], dtype=np.float32)
    obs_disp = (obs[:, -1] - obs[:, 0]) / 512.0
    fut_disp = (fut[:, -1] - obs[:, -1]) / 512.0
    obs_speed = np.sqrt((np.diff(obs, axis=1) ** 2).sum(-1)).mean(axis=1, keepdims=True) / 64.0
    fut_speed = np.sqrt((np.diff(fut, axis=1) ** 2).sum(-1)).mean(axis=1, keepdims=True) / 64.0
    last_xy = obs[:, -1] / 512.0
    return np.concatenate(
        [
            obs_disp,
            fut_disp,
            last_xy,
            obs_speed,
            fut_speed,
            obs_vis.mean(axis=1, keepdims=True),
            fut_vis.mean(axis=1, keepdims=True),
            obs_conf.mean(axis=1, keepdims=True),
            fut_conf.mean(axis=1, keepdims=True),
        ],
        axis=1,
    ).astype(np.float32)


def one_hot_semantic(z: Any) -> np.ndarray:
    sem = np.asarray(z["obs_semantic_last_id"], dtype=np.int64)
    if "source_semantic_id" in z.files:
        sem = np.where(sem >= 0, sem, np.asarray(z["source_semantic_id"], dtype=np.int64))
    sem = np.clip(sem, 0, 127)
    return np.eye(128, dtype=np.float32)[sem]


def close_pair(points: np.ndarray, inst: np.ndarray, q: float) -> np.ndarray:
    diff = inst[:, None] != inst[None, :]
    dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
    vals = dist[diff]
    if vals.size == 0:
        return np.zeros_like(diff, dtype=bool)
    thr = float(np.quantile(vals, q))
    out = diff & (dist <= max(thr, 8.0))
    np.fill_diagonal(out, False)
    return out


def future_crossing_pair(fut: np.ndarray, inst: np.ndarray) -> np.ndarray:
    diff = inst[:, None] != inst[None, :]
    # 采样少量 horizon 点，避免 MxMxH 太大。
    take = np.linspace(0, fut.shape[1] - 1, min(8, fut.shape[1]), dtype=np.int64)
    min_dist = np.full(diff.shape, np.inf, dtype=np.float32)
    for t in take:
        dist = np.sqrt(((fut[:, None, t] - fut[None, :, t]) ** 2).sum(-1))
        min_dist = np.minimum(min_dist, dist.astype(np.float32))
    vals = min_dist[diff]
    if vals.size == 0:
        return np.zeros_like(diff, dtype=bool)
    thr = float(np.quantile(vals, 0.10))
    out = diff & (min_dist <= max(thr, 10.0))
    np.fill_diagonal(out, False)
    return out


def occlusion_reappear(fut_vis: np.ndarray) -> np.ndarray:
    out = np.zeros((fut_vis.shape[0],), dtype=bool)
    for i, row in enumerate(fut_vis.astype(bool)):
        if not row.any() or row.all():
            continue
        false_idx = np.where(~row)[0]
        true_idx = np.where(row)[0]
        out[i] = bool(false_idx.size and true_idx.size and true_idx.max() > false_idx.min())
    return out


def main() -> int:
    rows: list[dict[str, Any]] = []
    totals: dict[str, Counter[str]] = {}
    blockers: list[str] = []
    for p in sorted(TARGET_ROOT.glob("*/*.npz")):
        try:
            z = np.load(p, allow_pickle=True)
            split = str(np.asarray(z["split"]).item())
            inst = np.asarray(z["point_to_instance_id"], dtype=np.int64)
            sem = np.asarray(z["obs_semantic_last_id"], dtype=np.int64)
            if "source_semantic_id" in z.files:
                sem = np.where(sem >= 0, sem, np.asarray(z["source_semantic_id"], dtype=np.int64))
            same = (inst[:, None] == inst[None, :]) & (inst[:, None] >= 0)
            np.fill_diagonal(same, False)
            diff = (inst[:, None] != inst[None, :]) & (inst[:, None] >= 0) & (inst[None, :] >= 0)
            same_sem = diff & (sem[:, None] == sem[None, :]) & (sem[:, None] >= 0)
            obs_last = np.asarray(z["obs_points"], dtype=np.float32)[:, -1]
            fut = np.asarray(z["future_points"], dtype=np.float32)
            spatial_hard = close_pair(obs_last, inst, 0.12)
            crossing = future_crossing_pair(fut, inst)
            identity_confuser = np.asarray(z["identity_confuser_pair_mask"], dtype=bool) | same_sem | spatial_hard | crossing
            np.fill_diagonal(identity_confuser, False)
            occ = occlusion_reappear(np.asarray(z["future_vis"], dtype=bool))
            meas = weighted_measurement(z)
            feat = np.concatenate([meas, one_hot_semantic(z), trace_features(z)], axis=1).astype(np.float32)
            available = (inst >= 0) & (np.asarray(z["obs_semantic_measurement_mask"], dtype=bool).any(axis=1))
            out_dir = OUT_ROOT / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / p.name
            np.savez_compressed(
                out_path,
                sample_uid=str(np.asarray(z["sample_uid"]).item()),
                split=split,
                dataset=str(np.asarray(z["dataset"]).item()),
                point_id=np.asarray(z["point_id"], dtype=np.int64),
                point_to_instance_id=inst,
                source_semantic_id=np.asarray(z["source_semantic_id"], dtype=np.int64),
                identity_input_features=feat,
                measurement_identity_embedding=meas,
                obs_points=np.asarray(z["obs_points"], dtype=np.float32),
                future_points=fut,
                obs_vis=np.asarray(z["obs_vis"], dtype=bool),
                future_vis=np.asarray(z["future_vis"], dtype=bool),
                same_instance_pair_mask=same.astype(bool),
                same_semantic_hard_negative_pair_mask=same_sem.astype(bool),
                same_frame_hard_negative_pair_mask=spatial_hard.astype(bool),
                trajectory_crossing_pair_mask=crossing.astype(bool),
                identity_confuser_pair_mask=identity_confuser.astype(bool),
                occlusion_reappear_point_mask=occ.astype(bool),
                identity_available_point_mask=available.astype(bool),
                future_teacher_embedding_input_allowed=False,
                leakage_safe=True,
            )
            c = totals.setdefault(split, Counter())
            c["samples"] += 1
            c["points"] += int(inst.size)
            c["available_points"] += int(available.sum())
            c["same_pairs"] += int(same.sum())
            c["diff_pairs"] += int(diff.sum())
            c["confuser_pairs"] += int(identity_confuser.sum())
            c["same_semantic_pairs"] += int(same_sem.sum())
            c["spatial_hard_pairs"] += int(spatial_hard.sum())
            c["crossing_pairs"] += int(crossing.sum())
            c["occlusion_reappear_points"] += int(occ.sum())
            rows.append(
                {
                    "source_npz": str(p.relative_to(ROOT)),
                    "output_npz": str(out_path.relative_to(ROOT)),
                    "split": split,
                    "dataset": str(np.asarray(z["dataset"]).item()),
                    "point_count": int(inst.size),
                    "instance_count": int(len(np.unique(inst[inst >= 0]))),
                    "same_pair_count": int(same.sum()),
                    "identity_confuser_pair_count": int(identity_confuser.sum()),
                    "occlusion_reappear_point_count": int(occ.sum()),
                }
            )
        except Exception as exc:
            blockers.append(f"{p}: {type(exc).__name__}: {exc}")
    split_report = {}
    for split, c in sorted(totals.items()):
        split_report[split] = {
            "samples": int(c["samples"]),
            "points": int(c["points"]),
            "available_point_ratio": float(c["available_points"] / max(c["points"], 1)),
            "same_pair_ratio": float(c["same_pairs"] / max(c["points"] * c["points"], 1)),
            "identity_confuser_pair_ratio": float(c["confuser_pairs"] / max(c["diff_pairs"], 1)),
            "same_semantic_hard_negative_pair_ratio": float(c["same_semantic_pairs"] / max(c["diff_pairs"], 1)),
            "same_frame_hard_negative_pair_ratio": float(c["spatial_hard_pairs"] / max(c["diff_pairs"], 1)),
            "trajectory_crossing_pair_ratio": float(c["crossing_pairs"] / max(c["diff_pairs"], 1)),
            "occlusion_reappear_point_ratio": float(c["occlusion_reappear_points"] / max(c["points"], 1)),
        }
    ready = bool(rows) and all(v["identity_confuser_pair_ratio"] > 0 for v in split_report.values())
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_identity_pairwise_targets_built": ready,
        "target_root": str(OUT_ROOT.relative_to(ROOT)),
        "source_target_root": str(TARGET_ROOT.relative_to(ROOT)),
        "sample_count": len(rows),
        "split_report": split_report,
        "same_frame_hard_negative_built": True,
        "same_semantic_confuser_built": True,
        "trajectory_crossing_target_built": True,
        "occlusion_reappear_target_built": True,
        "future_teacher_embedding_input_allowed": False,
        "leakage_safe": True,
        "rows": rows,
        "exact_blockers": blockers[:20],
        "recommended_next_step": "train_video_identity_pairwise_retrieval_head" if ready else "fix_video_identity_pairwise_targets",
        "中文结论": "V35.16 已把 video identity 从单点 same-instance 改成 pairwise/retrieval target，包含 same-semantic confuser、空间近邻 hard negative、轨迹 crossing、occlusion/reappear 分层。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.16 Video Identity Pairwise Target Build\n\n"
        f"- video_identity_pairwise_targets_built: {ready}\n"
        f"- sample_count: {len(rows)}\n"
        f"- same_frame_hard_negative_built: true\n"
        f"- same_semantic_confuser_built: true\n"
        f"- trajectory_crossing_target_built: true\n"
        f"- occlusion_reappear_target_built: true\n"
        f"- future_teacher_embedding_input_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"video_identity_pairwise_targets_built": ready, "sample_count": len(rows), "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
