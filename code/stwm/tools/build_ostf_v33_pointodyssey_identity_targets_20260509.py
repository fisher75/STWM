#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import (
    V33_IDENTITY_ROOT,
    assign_mask_ids,
    cache_paths,
    mask_path_for_frame,
    scalar,
)

REPORT = ROOT / "reports/stwm_ostf_v33_pointodyssey_identity_target_build_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_POINTODYSSEY_IDENTITY_TARGET_BUILD_20260509.md"


def _assign_query_instance(obs_instance: np.ndarray, obs_vis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Assign each point to an observed instance without using future labels."""
    m, t_obs = obs_instance.shape
    point_to_instance = np.full((m,), -1, dtype=np.int64)
    confidence = np.zeros((m,), dtype=np.float32)
    assignment_frame = np.full((m,), -1, dtype=np.int64)
    method = np.full((m,), "unavailable", dtype=object)
    for i in range(m):
        visible_idx = np.flatnonzero(obs_vis[i].astype(bool))
        if visible_idx.size == 0:
            continue
        valid_visible = [int(t) for t in visible_idx if int(obs_instance[i, int(t)]) >= 0]
        visible_count = max(int(visible_idx.size), 1)
        last_vis = int(visible_idx[-1])
        last_id = int(obs_instance[i, last_vis])
        if last_id >= 0:
            point_to_instance[i] = last_id
            confidence[i] = float(len(valid_visible) / visible_count)
            assignment_frame[i] = last_vis
            method[i] = "last_visible"
            continue
        if valid_visible:
            ids = obs_instance[i, valid_visible].astype(np.int64)
            uniq, counts = np.unique(ids, return_counts=True)
            best = int(uniq[np.argmax(counts)])
            best_frames = [int(t) for t in valid_visible if int(obs_instance[i, int(t)]) == best]
            point_to_instance[i] = best
            confidence[i] = float(counts.max() / visible_count)
            assignment_frame[i] = int(best_frames[-1])
            method[i] = "consensus"
    return point_to_instance, confidence, assignment_frame, method


def process_one(path: Path, *, overwrite: bool = False) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    uid = str(scalar(z, "video_uid", path.stem))
    split = str(scalar(z, "split", path.parent.name))
    dataset = str(scalar(z, "dataset", "pointodyssey"))
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    fut_points = np.asarray(z["fut_points"], dtype=np.float32)
    obs_vis = np.asarray(z["obs_vis"]).astype(bool)
    fut_vis = np.asarray(z["fut_vis"]).astype(bool)
    frame_paths = np.asarray(z["frame_paths"], dtype=object)
    m, t_obs = obs_points.shape[:2]
    h = fut_points.shape[1]
    point_id = np.asarray(z["point_id"], dtype=np.int64) if "point_id" in z.files else np.arange(m, dtype=np.int64)
    out = V33_IDENTITY_ROOT / split / f"{uid}.npz"
    obs_instance = np.full((m, t_obs), -1, dtype=np.int64)
    fut_instance = np.full((m, h), -1, dtype=np.int64)
    masks_available = True
    reused_existing_lookup = False
    if out.exists():
        try:
            old = np.load(out, allow_pickle=True)
            old_obs = np.asarray(old["obs_instance_id"], dtype=np.int64)
            old_fut = np.asarray(old["fut_instance_id"], dtype=np.int64)
            if old_obs.shape == obs_instance.shape and old_fut.shape == fut_instance.shape:
                obs_instance = old_obs
                fut_instance = old_fut
                reused_existing_lookup = True
                masks_available = bool((obs_instance >= 0).any() or (fut_instance >= 0).any())
        except Exception:
            reused_existing_lookup = False
    if not reused_existing_lookup:
        for t in range(t_obs):
            mp = mask_path_for_frame(str(frame_paths[t]))
            if not mp.exists():
                masks_available = False
            obs_instance[:, t] = assign_mask_ids(obs_points[:, t], obs_vis[:, t], mp)
        for t in range(h):
            frame_idx = t_obs + t
            mp = mask_path_for_frame(str(frame_paths[frame_idx]))
            if not mp.exists():
                masks_available = False
            fut_instance[:, t] = assign_mask_ids(fut_points[:, t], fut_vis[:, t], mp)
    obs_instance_available = (obs_instance >= 0) & obs_vis
    fut_instance_available = (fut_instance >= 0) & fut_vis
    point_to_instance, point_conf, assignment_frame, assignment_method = _assign_query_instance(obs_instance, obs_vis)
    fut_same_instance = (
        (fut_instance == point_to_instance[:, None])
        & (point_to_instance[:, None] >= 0)
        & fut_instance_available
    )
    class_semantic_id = np.full((m, h), -1, dtype=np.int64)
    class_available = np.zeros((m, h), dtype=bool)
    out.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not out.exists():
        np.savez_compressed(
            out,
            sample_uid=uid,
            dataset=dataset,
            split=split,
            source_npz=str(path.relative_to(ROOT)),
            frame_paths=frame_paths,
            point_id=point_id,
            fut_same_point_valid=fut_vis,
            fut_point_visible_target=fut_vis,
            fut_point_visible_mask=np.ones_like(fut_vis, dtype=bool),
            point_to_instance_id=point_to_instance,
            obs_instance_id=obs_instance,
            fut_instance_id=fut_instance,
            obs_instance_available_mask=obs_instance_available,
            fut_instance_available_mask=fut_instance_available,
            fut_same_instance_as_obs=fut_same_instance,
            semantic_class_id=class_semantic_id,
            class_available_mask=class_available,
            teacher_source="none",
            point_assignment_confidence=point_conf,
            point_to_instance_assignment_frame=assignment_frame,
            point_to_instance_assignment_method=assignment_method,
            leakage_safe=True,
            input_uses_observed_only=True,
            future_targets_supervision_only=True,
            M=np.asarray(m, dtype=np.int64),
            horizon=np.asarray(h, dtype=np.int64),
            instance_identity_unavailable=not masks_available,
        )
    return {
        "source": str(path.relative_to(ROOT)),
        "sidecar": str(out.relative_to(ROOT)),
        "M": int(m),
        "H": int(h),
        "split": split,
        "point_identity_available": bool(point_id.shape[0] == m),
        "instance_identity_available": bool((point_to_instance >= 0).any()),
        "point_identity_count": int(m),
        "instance_identity_count": int((point_to_instance >= 0).sum()),
        "assignment_confidence_mean": float(point_conf.mean()),
        "assignment_confidence_p10": float(np.quantile(point_conf, 0.10)),
        "last_visible_assignment_count": int((assignment_method == "last_visible").sum()),
        "consensus_assignment_count": int((assignment_method == "consensus").sum()),
        "unavailable_assignment_count": int((assignment_method == "unavailable").sum()),
        "reused_existing_mask_lookup": reused_existing_lookup,
        "leakage_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    paths = cache_paths()
    if args.limit:
        paths = paths[: args.limit]
    rows = [process_one(p, overwrite=args.overwrite) for p in paths]
    total = len(rows)
    point_ok = sum(int(r["point_identity_available"]) for r in rows)
    inst_ok = sum(int(r["instance_identity_available"]) for r in rows)
    conf = [float(r["assignment_confidence_mean"]) for r in rows]
    by_combo: dict[str, dict[str, Any]] = {}
    last_visible_count = 0
    consensus_count = 0
    unavailable_count = 0
    total_points = 0
    for r in rows:
        key = f"M{r['M']}_H{r['H']}"
        by_combo.setdefault(key, {"samples": 0, "instance_samples": 0, "point_count": 0, "assigned_points": 0})
        by_combo[key]["samples"] += 1
        by_combo[key]["instance_samples"] += int(r["instance_identity_available"])
        by_combo[key]["point_count"] += int(r["point_identity_count"])
        by_combo[key]["assigned_points"] += int(r["instance_identity_count"])
        last_visible_count += int(r["last_visible_assignment_count"])
        consensus_count += int(r["consensus_assignment_count"])
        unavailable_count += int(r["unavailable_assignment_count"])
        total_points += int(r["point_identity_count"])
    for combo in by_combo.values():
        combo["instance_point_coverage_ratio"] = combo["assigned_points"] / max(combo["point_count"], 1)
    payload = {
        "generated_at_utc": utc_now(),
        "target_root": str(V33_IDENTITY_ROOT.relative_to(ROOT)),
        "total_samples_processed": total,
        "samples_with_point_identity": point_ok,
        "samples_with_instance_identity": inst_ok,
        "point_identity_coverage_ratio": point_ok / max(total, 1),
        "instance_identity_coverage_ratio": inst_ok / max(total, 1),
        "assignment_confidence_mean": float(np.mean(conf)) if conf else 0.0,
        "assignment_confidence_p10": float(np.quantile(conf, 0.10)) if conf else 0.0,
        "last_visible_assignment_ratio": last_visible_count / max(total_points, 1),
        "consensus_assignment_ratio": consensus_count / max(total_points, 1),
        "unavailable_assignment_ratio": unavailable_count / max(total_points, 1),
        "leakage_safe": True,
        "input_uses_observed_only": True,
        "future_targets_supervision_only": True,
        "by_M_H_coverage": by_combo,
        "by_combo": by_combo,
        "examples": rows[:20],
        "exact_blockers": [] if rows else ["no V30 PointOdyssey cache files found"],
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 PointOdyssey Identity Target Build", payload, [
        "total_samples_processed",
        "samples_with_point_identity",
        "samples_with_instance_identity",
        "point_identity_coverage_ratio",
        "instance_identity_coverage_ratio",
        "assignment_confidence_mean",
        "assignment_confidence_p10",
        "last_visible_assignment_ratio",
        "consensus_assignment_ratio",
        "unavailable_assignment_ratio",
        "leakage_safe",
        "input_uses_observed_only",
        "future_targets_supervision_only",
        "target_root",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
