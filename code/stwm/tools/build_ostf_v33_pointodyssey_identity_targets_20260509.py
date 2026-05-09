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
    obs_instance = np.full((m, t_obs), -1, dtype=np.int64)
    fut_instance = np.full((m, h), -1, dtype=np.int64)
    masks_available = True
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
    point_to_instance = obs_instance[:, -1].copy()
    point_conf = (point_to_instance >= 0).astype(np.float32)
    fut_same_instance = (fut_instance == point_to_instance[:, None]) & (point_to_instance[:, None] >= 0) & fut_vis
    class_semantic_id = np.full((m, h), -1, dtype=np.int64)
    class_available = np.zeros((m, h), dtype=bool)
    out = V33_IDENTITY_ROOT / split / f"{uid}.npz"
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
            point_to_instance_id=point_to_instance,
            obs_instance_id=obs_instance,
            fut_instance_id=fut_instance,
            fut_same_instance_as_obs=fut_same_instance,
            semantic_class_id=class_semantic_id,
            class_available_mask=class_available,
            teacher_source="none",
            point_assignment_confidence=point_conf,
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
    by_combo: dict[str, dict[str, int]] = {}
    for r in rows:
        key = f"M{r['M']}_H{r['H']}"
        by_combo.setdefault(key, {"samples": 0, "instance_samples": 0})
        by_combo[key]["samples"] += 1
        by_combo[key]["instance_samples"] += int(r["instance_identity_available"])
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
        "leakage_safe": True,
        "input_uses_observed_only": True,
        "future_targets_supervision_only": True,
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
        "leakage_safe",
        "target_root",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
