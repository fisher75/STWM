#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CACHE_BASE = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16"


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
    lines = ["# STWM Real Object-Dense Teacher Cache Audit V16", ""]
    for key in [
        "real_teacher_tracks_exist",
        "persistent_point_identity_valid",
        "fake_dense_or_anchor_copied",
        "processed_clip_count",
        "point_count",
        "valid_point_ratio",
        "success_gate_passed",
        "exact_blocker_if_less_than_500",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    for combo, row in payload.get("per_combo", {}).items():
        lines.extend(["", f"## {combo}"])
        for key in ["processed_clip_count", "point_count", "valid_point_ratio", "same_trajectory_fraction", "persistent_point_identity_valid", "success_gate_passed"]:
            lines.append(f"- {key}: `{row.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _same_trajectory_fraction(tracks: np.ndarray, atol: float = 1e-3) -> float:
    if tracks.shape[1] <= 1:
        return 1.0
    delta = tracks - tracks[:, :1, :, :]
    same = np.all(np.abs(delta) <= atol, axis=(2, 3))
    return float(same.mean())


def _audit_combo(combo_dir: Path) -> dict[str, Any]:
    combo = combo_dir.name
    parts = combo.replace("M", "").split("_H")
    m = int(parts[0]) if len(parts) == 2 else None
    h = int(parts[1]) if len(parts) == 2 else None
    paths = sorted(combo_dir.glob("*/*.npz"))
    reports = [p for p in sorted((ROOT / "reports").glob(f"stwm_cotracker_object_dense_teacher_v16_{combo}*_20260502.json")) if "_teacher_v16_20260502" not in p.name]
    report_payloads = [_load_json(p) for p in reports]
    merged_failures: Counter[str] = Counter()
    failed_clip_count = 0
    for report in report_payloads:
        failed_clip_count += int(report.get("failed_clip_count", 0) or 0)
        for reason, count in (report.get("failure_reason_top", {}) or {}).items():
            merged_failures[str(reason)] += int(count)
    source_counts = Counter()
    split_clip_counts = Counter()
    dataset_clip_counts = Counter()
    split_point_counts = Counter()
    split_object_counts = Counter()
    total_objects = total_points = visible_steps = total_steps = 0
    same_fracs, variances = [], []
    point_id_ok = []
    no_future_box = []
    for path in paths:
        z = np.load(path, allow_pickle=True)
        tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
        visibility = np.asarray(z["visibility"]).astype(bool)
        point_id = np.asarray(z["point_id"])
        split = str(_scalar(z["split"]))
        dataset = str(_scalar(z["dataset"]))
        source = str(_scalar(z["teacher_source"]))
        objects = int(tracks.shape[0])
        points = int(tracks.shape[0] * tracks.shape[1])
        source_counts[source] += points
        split_clip_counts[split] += 1
        dataset_clip_counts[dataset] += 1
        split_point_counts[split] += points
        split_object_counts[split] += objects
        total_objects += objects
        total_points += points
        visible_steps += int(visibility.sum())
        total_steps += int(visibility.size)
        same_fracs.append(_same_trajectory_fraction(tracks))
        variances.append(float(np.mean(np.var(tracks, axis=2))))
        point_id_ok.append(point_id.shape == tracks.shape[:2] and len(np.unique(point_id)) == points)
        no_future_box.append(bool(_scalar(z["no_future_box_projection"])))
    valid_ratio = float(visible_steps / max(total_steps, 1))
    mean_same = float(np.mean(same_fracs)) if same_fracs else 1.0
    all_cotracker = bool(source_counts) and set(source_counts.keys()) == {"cotracker_official"}
    persistent = bool(paths and all(point_id_ok))
    fake_dense = bool(mean_same > 0.20 or not all_cotracker)
    gate = bool(paths and all_cotracker and persistent and valid_ratio > 0.5 and not fake_dense)
    if len(paths) < 500:
        blocker = dict(merged_failures.most_common(10)) or "processed fewer than 500 clips; inspect combo runner failures or available valid windows"
    else:
        blocker = None
    return {
        "combo": combo,
        "M": m,
        "H": h,
        "processed_clip_count": len(paths),
        "failed_clip_count": failed_clip_count,
        "failure_reason_top": dict(merged_failures.most_common(20)),
        "object_count": total_objects,
        "point_count": total_points,
        "average_points_per_object": float(total_points / max(total_objects, 1)),
        "average_points_per_scene": float(total_points / max(len(paths), 1)),
        "valid_point_ratio": valid_ratio,
        "same_trajectory_fraction": mean_same,
        "trajectory_variance": float(np.mean(variances)) if variances else 0.0,
        "visibility_coverage": valid_ratio,
        "per_split_clip_counts": dict(split_clip_counts),
        "per_split_object_counts": dict(split_object_counts),
        "per_split_point_counts": dict(split_point_counts),
        "per_dataset_counts": dict(dataset_clip_counts),
        "teacher_source_distribution": dict(source_counts),
        "real_teacher_tracks_exist": all_cotracker,
        "persistent_point_identity_valid": persistent,
        "fake_dense_or_anchor_copied": fake_dense,
        "no_future_box_projection_all": bool(no_future_box and all(no_future_box)),
        "future_leakage_audit_passed": bool(no_future_box and all(no_future_box)),
        "success_gate_passed": gate and len(paths) >= 500,
        "exact_blocker_if_less_than_500": blocker,
        "runner_report_paths": [str(p.relative_to(ROOT)) for p in reports],
    }


def main() -> int:
    combos = sorted([p for p in CACHE_BASE.glob("M*_H*") if p.is_dir()])
    per_combo = {p.name: _audit_combo(p) for p in combos}
    total_clips = sum(row["processed_clip_count"] for row in per_combo.values())
    total_points = sum(row["point_count"] for row in per_combo.values())
    total_steps = sum(row["point_count"] * (8 + int(row.get("H") or 0)) for row in per_combo.values())
    valid_num = sum(row["valid_point_ratio"] * row["point_count"] * (8 + int(row.get("H") or 0)) for row in per_combo.values())
    payload = {
        "audit_name": "stwm_real_object_dense_teacher_cache_audit_v16",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_base": str(CACHE_BASE.relative_to(ROOT)),
        "per_combo": per_combo,
        "processed_clip_count": total_clips,
        "point_count": total_points,
        "valid_point_ratio": float(valid_num / max(total_steps, 1)),
        "real_teacher_tracks_exist": bool(per_combo and all(row["real_teacher_tracks_exist"] for row in per_combo.values())),
        "persistent_point_identity_valid": bool(per_combo and all(row["persistent_point_identity_valid"] for row in per_combo.values())),
        "fake_dense_or_anchor_copied": bool(any(row["fake_dense_or_anchor_copied"] for row in per_combo.values())),
        "future_leakage_audit_passed": bool(per_combo and all(row["future_leakage_audit_passed"] for row in per_combo.values())),
        "success_gate_passed": bool(per_combo and all(row["success_gate_passed"] for row in per_combo.values())),
        "exact_blocker_if_less_than_500": {k: v["exact_blocker_if_less_than_500"] for k, v in per_combo.items() if v["processed_clip_count"] < 500},
    }
    _dump(ROOT / "reports/stwm_real_object_dense_teacher_cache_audit_v16_20260502.json", payload)
    _write_doc(ROOT / "docs/STWM_REAL_OBJECT_DENSE_TEACHER_CACHE_AUDIT_V16_20260502.md", payload)
    print("reports/stwm_real_object_dense_teacher_cache_audit_v16_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
