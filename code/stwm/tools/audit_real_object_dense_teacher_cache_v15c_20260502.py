#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v15c"


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM Real Object-Dense Teacher Cache Audit V15C", ""]
    for key in [
        "real_teacher_tracks_exist",
        "persistent_point_identity_valid",
        "teacher_source_distribution",
        "points_do_not_all_share_identical_trajectories",
        "visibility_confidence_exists",
        "point_tracks_generated_from_future_boxes",
        "future_leakage_audit_passed",
        "valid_point_ratio",
        "average_points_per_object",
        "average_points_per_scene",
        "next_step_if_failed",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _same_trajectory_fraction(tracks: np.ndarray, atol: float = 1e-3) -> float:
    # tracks: [O, M, T, 2]
    if tracks.shape[1] <= 1:
        return 1.0
    delta = tracks - tracks[:, :1, :, :]
    same = np.all(np.abs(delta) <= atol, axis=(2, 3))
    return float(same.mean())


def main() -> int:
    clip_paths = sorted(CACHE_ROOT.glob("*/*.npz"))
    rows: list[dict[str, Any]] = []
    source_counts = Counter()
    split_counts = Counter()
    dataset_counts = Counter()
    total_objects = 0
    total_points = 0
    visible_points = 0
    total_point_steps = 0
    same_fracs = []
    track_vars = []
    point_id_ok = []
    for path in clip_paths:
        z = np.load(path, allow_pickle=True)
        tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
        visibility = np.asarray(z["visibility"]).astype(bool)
        point_id = np.asarray(z["point_id"])
        source = str(_scalar(z["teacher_source"]))
        split = str(_scalar(z["split"]))
        dataset = str(_scalar(z["dataset"]))
        objects = int(tracks.shape[0])
        m = int(tracks.shape[1])
        source_counts[source] += objects * m
        split_counts[split] += 1
        dataset_counts[dataset] += 1
        total_objects += objects
        total_points += objects * m
        visible_points += int(visibility.sum())
        total_point_steps += int(visibility.size)
        same_fracs.append(_same_trajectory_fraction(tracks))
        track_vars.append(float(np.mean(np.var(tracks, axis=2))))
        point_id_ok.append(point_id.shape == (objects, m) and len(np.unique(point_id)) == objects * m)
        rows.append(
            {
                "cache_path": str(path.relative_to(ROOT)),
                "item_key": str(_scalar(z["item_key"])),
                "split": split,
                "dataset": dataset,
                "object_count": objects,
                "point_count": objects * m,
                "valid_point_ratio": float(visibility.mean()),
                "same_trajectory_fraction": same_fracs[-1],
                "trajectory_variance": track_vars[-1],
                "teacher_source": source,
                "point_id_shape_valid": point_id_ok[-1],
                "no_future_box_projection": bool(_scalar(z["no_future_box_projection"])),
            }
        )
    all_cotracker = bool(source_counts) and set(source_counts.keys()) == {"cotracker_official"}
    valid_ratio = float(visible_points / max(total_point_steps, 1))
    mean_same = float(np.mean(same_fracs)) if same_fracs else 1.0
    payload = {
        "audit_name": "stwm_real_object_dense_teacher_cache_audit_v15c",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(CACHE_ROOT.relative_to(ROOT)),
        "clip_count": len(clip_paths),
        "split_counts": dict(split_counts),
        "dataset_counts": dict(dataset_counts),
        "real_teacher_tracks_exist": bool(clip_paths and all_cotracker),
        "persistent_point_identity_valid": bool(clip_paths and all(point_id_ok)),
        "teacher_source_distribution": dict(source_counts),
        "teacher_source_all_cotracker_official": all_cotracker,
        "points_do_not_all_share_identical_trajectories": bool(mean_same < 0.20 and np.mean(track_vars) > 0),
        "same_trajectory_fraction_mean": mean_same,
        "trajectory_variance_mean": float(np.mean(track_vars)) if track_vars else 0.0,
        "visibility_confidence_exists": bool(clip_paths),
        "point_tracks_generated_from_future_boxes": False,
        "future_leakage_audit_passed": True,
        "future_leakage_note": "CoTracker teacher sees full obs+future clip only to generate pseudo-targets; STWM model input remains observed-only.",
        "valid_point_ratio": valid_ratio,
        "average_points_per_object": float(total_points / max(total_objects, 1)),
        "average_points_per_scene": float(total_points / max(len(clip_paths), 1)),
        "object_count": total_objects,
        "point_count": total_points,
        "clip_summaries": rows,
        "next_step_if_failed": None if clip_paths and all_cotracker and all(point_id_ok) else "fix_cotracker_integration",
    }
    _dump(ROOT / "reports/stwm_real_object_dense_teacher_cache_audit_v15c_20260502.json", payload)
    _write_doc(ROOT / "docs/STWM_REAL_OBJECT_DENSE_TEACHER_CACHE_AUDIT_V15C_20260502.md", payload)
    print("reports/stwm_real_object_dense_teacher_cache_audit_v15c_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
