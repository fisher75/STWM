#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json
from stwm.tools.ostf_v30_external_gt_schema_20260508 import EXTERNAL_CACHE_ROOT, ROOT, audit_dataset, choose_indices, save_cache_report, save_ostf_npz, utc_now


DATASET = "tapvid3d"
REPORT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_cache_build_tapvid3d_20260508.json"
OBS_LEN = 8
HORIZONS = (32, 64, 96)
MS = (128, 512, 1024)


def _split_for(path: Path) -> str:
    h = int(hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8], 16) % 10
    if h < 6:
        return "train"
    if h < 8:
        return "val"
    return "test"


def _npz_files(root: Path) -> list[Path]:
    files = sorted((root / "minival_dataset").glob("**/*.npz")) if (root / "minival_dataset").exists() else []
    files += sorted((root / "debug_dataset").glob("**/*.npz")) if (root / "debug_dataset").exists() else []
    return files[:400]


def main() -> int:
    audit = audit_dataset(DATASET)
    root = Path(audit["candidate_paths"][0]) if audit["candidate_paths"] else Path("/nonexistent")
    written = 0
    skipped = {}
    examples = []
    if root.exists():
        for npz_path in _npz_files(root):
            try:
                z = np.load(npz_path, allow_pickle=False)
                tracks_key = "tracks_XYZ" if "tracks_XYZ" in z.files else "tracks_xyz"
                if tracks_key not in z.files:
                    skipped["missing_tracks_XYZ"] = skipped.get("missing_tracks_XYZ", 0) + 1
                    continue
                tracks = np.asarray(z[tracks_key], dtype=np.float32)  # [T,N,3]
                vis = np.asarray(z["visibility"] if "visibility" in z.files else np.ones(tracks.shape[:2], dtype=bool)).astype(bool)
                intr = np.asarray(z["fx_fy_cx_cy"], dtype=np.float32) if "fx_fy_cx_cy" in z.files else None
            except Exception as exc:
                skipped["parse_error"] = skipped.get("parse_error", 0) + 1
                continue
            total_t, n, _ = tracks.shape
            split = _split_for(npz_path)
            for horizon in HORIZONS:
                clip_len = OBS_LEN + horizon
                if total_t < clip_len:
                    skipped[f"H{horizon}_insufficient_frames"] = skipped.get(f"H{horizon}_insufficient_frames", 0) + 1
                    continue
                starts = sorted(set(int(round(x)) for x in np.linspace(0, total_t - clip_len, 3)))
                for start in starts:
                    q = start + OBS_LEN - 1
                    base_valid = vis[q]
                    for m in MS:
                        idx = choose_indices(base_valid, m)
                        if idx.size < m:
                            skipped[f"M{m}_insufficient_points"] = skipped.get(f"M{m}_insufficient_points", 0) + 1
                            continue
                        obs = tracks[start : start + OBS_LEN, idx].transpose(1, 0, 2)
                        fut = tracks[start + OBS_LEN : start + clip_len, idx].transpose(1, 0, 2)
                        obs_vis = vis[start : start + OBS_LEN, idx].transpose(1, 0)
                        fut_vis = vis[start + OBS_LEN : start + clip_len, idx].transpose(1, 0)
                        if float(fut_vis.mean()) < 0.4:
                            skipped[f"M{m}_H{horizon}_valid_future_ratio_lt_0.4"] = skipped.get(
                                f"M{m}_H{horizon}_valid_future_ratio_lt_0.4", 0
                            ) + 1
                            continue
                        combo = f"M{m}_H{horizon}"
                        uid = f"tapvid3d_{npz_path.parent.name}_{npz_path.stem}_{start:05d}_{combo}"
                        rel = Path("outputs/cache/stwm_ostf_v30_external_gt") / DATASET / combo / split / f"{uid}.npz"
                        path = ROOT / rel
                        if path.exists() and path.stat().st_size > 0:
                            written += 1
                            continue
                        save_ostf_npz(
                            path,
                            video_uid=uid,
                            dataset=DATASET,
                            split=split,
                            frame_paths=[f"tapvid3d://{npz_path}/{i:05d}" for i in range(start, start + clip_len)],
                            obs_points=obs,
                            fut_points=fut,
                            obs_vis=obs_vis,
                            fut_vis=fut_vis,
                            point_sampling_method="official_tapvid3d_visible_even_subset",
                            source_path=str(npz_path),
                            coordinate_system="world_xyz",
                            intrinsics=intr,
                        )
                        written += 1
                        if len(examples) < 10:
                            examples.append(str(rel))
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
        "cache_ready": False,
        "partial_cache_ready": bool(any(ready_combos.values())),
        "written_or_existing_cache_count": written,
        "combo_counts": combo_counts,
        "ready_combos": ready_combos,
        "examples": examples,
        "skipped": skipped,
        "exact_blocker": "TAPVid-3D local data is diagnostic/minival; no full official train/val/test main benchmark certified.",
        "source_gt_not_teacher": True,
        "no_future_leakage": True,
    }
    dump_json(REPORT_PATH, payload)
    save_cache_report(DATASET, payload)
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
