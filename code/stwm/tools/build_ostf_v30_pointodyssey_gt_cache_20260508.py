#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json
from stwm.tools.ostf_v30_external_gt_schema_20260508 import (
    EXTERNAL_CACHE_ROOT,
    ROOT,
    audit_dataset,
    choose_indices,
    save_cache_report,
    save_ostf_npz,
    utc_now,
)


DATASET = "pointodyssey"
REPORT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_cache_build_pointodyssey_20260508.json"
OBS_LEN = 8
HORIZONS = (32, 64, 96)
MS = (128, 512, 1024)
WINDOWS_PER_SEQUENCE = {"train": 8, "val": 8, "test": 20}
MAX_SEQUENCES = {"train": 24, "val": 15, "test": 13}


def _scalar(x: np.ndarray) -> Any:
    a = np.asarray(x)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _sequence_dirs(root: Path, split: str) -> list[Path]:
    split_dir = root / split
    if not split_dir.exists():
        return []
    out = []
    for seq in sorted(split_dir.iterdir(), key=lambda p: p.name):
        if seq.is_dir() and (seq / "anno.npz").exists() and (seq / "rgbs").exists():
            out.append(seq)
    return out[: MAX_SEQUENCES[split]]


def _starts(total_frames: int, clip_len: int, count: int) -> list[int]:
    if total_frames < clip_len:
        return []
    max_start = total_frames - clip_len
    if count <= 1:
        return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, max_start, count)))


def _frame_paths(seq: Path, start: int, clip_len: int) -> list[str]:
    files = sorted([*seq.joinpath("rgbs").glob("*.jpg"), *seq.joinpath("rgbs").glob("*.jpeg"), *seq.joinpath("rgbs").glob("*.png")])
    return [str(p) for p in files[start : start + clip_len]]


def _process_sequence(seq: Path, split: str) -> dict[str, Any]:
    z = np.load(seq / "anno.npz", allow_pickle=False)
    trajs = np.asarray(z["trajs_2d"], dtype=np.float32)
    valids = np.asarray(z["valids"]).astype(bool)
    visibs = np.asarray(z["visibs"]).astype(bool)
    intr = np.asarray(z["intrinsics"], dtype=np.float32) if "intrinsics" in z.files else None
    extr = np.asarray(z["extrinsics"], dtype=np.float32) if "extrinsics" in z.files else None
    finite = np.isfinite(trajs).all(axis=-1)
    usable = valids & visibs & finite
    t_total, n_points = usable.shape
    written = 0
    skipped: dict[str, int] = {}
    examples = []
    for horizon in HORIZONS:
        clip_len = OBS_LEN + horizon
        for start in _starts(t_total, clip_len, WINDOWS_PER_SEQUENCE[split]):
            q = start + OBS_LEN - 1
            base_valid = usable[q]
            for m in MS:
                idx = choose_indices(base_valid, m)
                if idx.size < m:
                    skipped[f"M{m}_H{horizon}_insufficient_query_visible_points"] = skipped.get(
                        f"M{m}_H{horizon}_insufficient_query_visible_points", 0
                    ) + 1
                    continue
                obs = trajs[start : start + OBS_LEN, idx].transpose(1, 0, 2)
                fut = trajs[start + OBS_LEN : start + clip_len, idx].transpose(1, 0, 2)
                obs_vis = usable[start : start + OBS_LEN, idx].transpose(1, 0)
                fut_vis = usable[start + OBS_LEN : start + clip_len, idx].transpose(1, 0)
                if float(fut_vis.mean()) < 0.4:
                    skipped[f"M{m}_H{horizon}_valid_future_ratio_lt_0.4"] = skipped.get(
                        f"M{m}_H{horizon}_valid_future_ratio_lt_0.4", 0
                    ) + 1
                    continue
                combo = f"M{m}_H{horizon}"
                uid = f"pointodyssey_{split}_{seq.name}_{start:06d}_{combo}"
                rel = Path("outputs/cache/stwm_ostf_v30_external_gt") / DATASET / combo / split / f"{uid}.npz"
                path = ROOT / rel
                if path.exists() and path.stat().st_size > 0:
                    written += 1
                    continue
                intr_win = intr[start:clip_len] if intr is not None else None
                extr_win = extr[start:clip_len] if extr is not None else None
                save_ostf_npz(
                    path,
                    video_uid=uid,
                    dataset=DATASET,
                    split=split,
                    frame_paths=_frame_paths(seq, start, clip_len),
                    obs_points=obs,
                    fut_points=fut,
                    obs_vis=obs_vis,
                    fut_vis=fut_vis,
                    point_sampling_method="query_visible_even_indices",
                    source_path=str(seq / "anno.npz"),
                    coordinate_system="pixel_xy",
                    intrinsics=intr_win,
                    extrinsics=extr_win,
                )
                written += 1
                if len(examples) < 10:
                    examples.append(str(rel))
    return {
        "sequence": seq.name,
        "split": split,
        "frame_count": int(t_total),
        "source_point_count": int(n_points),
        "written_or_existing_cache_count": int(written),
        "skipped": skipped,
        "examples": examples,
    }


def main() -> int:
    audit = audit_dataset(DATASET)
    root = Path(audit["candidate_paths"][0]) if audit["candidate_paths"] else Path("/nonexistent")
    if audit["completeness_status"] not in {"complete", "partial"} or not root.exists():
        payload = {
            "dataset": DATASET,
            "generated_at_utc": utc_now(),
            "cache_ready": False,
            "exact_blocker": audit.get("exact_blocker") or "PointOdyssey unavailable",
            "audit": audit,
        }
        dump_json(REPORT_PATH, payload)
        save_cache_report(DATASET, payload)
        print(REPORT_PATH.relative_to(ROOT))
        return 0
    per_sequence = []
    for split in ("train", "val", "test"):
        for seq in _sequence_dirs(root, split):
            per_sequence.append(_process_sequence(seq, split))
    combo_counts = {}
    for m in MS:
        for horizon in HORIZONS:
            combo = f"M{m}_H{horizon}"
            combo_counts[combo] = {
                split: len(list((EXTERNAL_CACHE_ROOT / DATASET / combo / split).glob("*.npz")))
                for split in ("train", "val", "test")
            }
    ready_combos = {
        combo: all(counts.get(split, 0) > 0 for split in ("train", "val", "test"))
        for combo, counts in combo_counts.items()
    }
    payload = {
        "dataset": DATASET,
        "generated_at_utc": utc_now(),
        "source_root": str(root),
        "obs_len": OBS_LEN,
        "horizons": list(HORIZONS),
        "M_values": list(MS),
        "cache_root": str((EXTERNAL_CACHE_ROOT / DATASET).relative_to(ROOT)),
        "per_sequence": per_sequence,
        "combo_counts": combo_counts,
        "ready_combos": ready_combos,
        "cache_ready": bool(any(ready_combos.values())),
        "partial_cache_ready": bool(any(ready_combos.values())),
        "source_gt_not_teacher": True,
        "no_future_leakage": True,
    }
    dump_json(REPORT_PATH, payload)
    save_cache_report(DATASET, payload)
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
