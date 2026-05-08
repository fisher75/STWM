#!/usr/bin/env python3
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import dump_json
from stwm.tools.ostf_v30_external_gt_schema_20260508 import EXTERNAL_CACHE_ROOT, ROOT, audit_dataset, save_cache_report, save_ostf_npz, utc_now


DATASET = "tapvid"
REPORT_PATH = ROOT / "reports/stwm_ostf_v30_external_gt_cache_build_tapvid_20260508.json"
OBS_LEN = 8
HORIZONS = (32, 64, 96)
MS = (128, 512, 1024)


def _iter_pickles(root: Path) -> list[Path]:
    return sorted(root.glob("**/*.pkl"))


def _load_items(path: Path):
    with path.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        return [(str(k), v) for k, v in payload.items()]
    return [(f"{path.stem}_{i:05d}", v) for i, v in enumerate(payload)]


def main() -> int:
    audit = audit_dataset(DATASET)
    root = Path(audit["candidate_paths"][0]) if audit["candidate_paths"] else Path("/nonexistent")
    written = 0
    skipped = {}
    examples = []
    if root.exists():
        for pkl_path in _iter_pickles(root):
            try:
                items = _load_items(pkl_path)
            except Exception as exc:
                skipped[f"{pkl_path.name}_parse_error"] = str(exc)
                continue
            for item_idx, (name, item) in enumerate(items[:80]):
                points = np.asarray(item.get("points"), dtype=np.float32) if isinstance(item, dict) and "points" in item else None
                occluded = np.asarray(item.get("occluded")).astype(bool) if isinstance(item, dict) and "occluded" in item else None
                if points is None or occluded is None or points.ndim != 3:
                    skipped["missing_points_or_occluded"] = skipped.get("missing_points_or_occluded", 0) + 1
                    continue
                # TAP-Vid layout is [N,T,2].
                n, total_t, _ = points.shape
                for horizon in HORIZONS:
                    clip_len = OBS_LEN + horizon
                    if total_t < clip_len:
                        skipped[f"H{horizon}_insufficient_frames"] = skipped.get(f"H{horizon}_insufficient_frames", 0) + 1
                        continue
                    for m in MS:
                        if n < m:
                            skipped[f"M{m}_insufficient_points"] = skipped.get(f"M{m}_insufficient_points", 0) + 1
                            continue
                        idx = np.linspace(0, n - 1, m).round().astype(np.int64)
                        uid = f"tapvid_{pkl_path.stem}_{item_idx:05d}_M{m}_H{horizon}"
                        rel = Path("outputs/cache/stwm_ostf_v30_external_gt") / DATASET / f"M{m}_H{horizon}" / "test" / f"{uid}.npz"
                        path = ROOT / rel
                        if path.exists() and path.stat().st_size > 0:
                            written += 1
                            continue
                        obs = points[idx, :OBS_LEN]
                        fut = points[idx, OBS_LEN:clip_len]
                        vis = ~occluded[idx, :clip_len]
                        save_ostf_npz(
                            path,
                            video_uid=uid,
                            dataset=DATASET,
                            split="test",
                            frame_paths=[f"tapvid://{pkl_path.name}/{name}/frame_{i:05d}" for i in range(clip_len)],
                            obs_points=obs,
                            fut_points=fut,
                            obs_vis=vis[:, :OBS_LEN],
                            fut_vis=vis[:, OBS_LEN:clip_len],
                            point_sampling_method="official_tapvid_points_even_subset",
                            source_path=str(pkl_path),
                            coordinate_system="pixel_xy",
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
    payload = {
        "dataset": DATASET,
        "generated_at_utc": utc_now(),
        "cache_ready": False,
        "partial_cache_ready": bool(written),
        "written_or_existing_cache_count": written,
        "combo_counts": combo_counts,
        "examples": examples,
        "skipped": skipped,
        "exact_blocker": "TAP-Vid local GT is sparse; no M128/M512/M1024 cache built unless enough official points exist.",
        "source_gt_not_teacher": True,
        "no_future_leakage": True,
    }
    dump_json(REPORT_PATH, payload)
    save_cache_report(DATASET, payload)
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
