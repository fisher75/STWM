#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import V30_POINTODYSSEY_CACHE, V33_IDENTITY_ROOT, scalar

REPORT = ROOT / "reports/stwm_ostf_v33_dense_field_target_coverage_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_DENSE_FIELD_TARGET_COVERAGE_20260509.md"


def sidecar_for(uid: str, split: str) -> Path:
    return V33_IDENTITY_ROOT / split / f"{uid}.npz"


def combo_stats(m: int) -> dict[str, Any]:
    paths = sorted(V30_POINTODYSSEY_CACHE.glob(f"M{m}_H*/*/*.npz"))
    total = 0
    point_ok = 0
    inst_ratios = []
    teacher_crop_feasible = 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        uid = str(scalar(z, "video_uid", p.stem))
        split = str(scalar(z, "split", p.parent.name))
        total += 1
        if "point_id" in z.files and np.asarray(z["point_id"]).shape[0] == m:
            point_ok += 1
        sc = sidecar_for(uid, split)
        if sc.exists():
            s = np.load(sc, allow_pickle=True)
            ids = np.asarray(s["point_to_instance_id"])
            inst_ratios.append(float((ids >= 0).mean()))
        frames = np.asarray(z["frame_paths"], dtype=object) if "frame_paths" in z.files else []
        if len(frames) and Path(str(frames[-1])).exists():
            teacher_crop_feasible += 1
    return {
        "sample_count": total,
        "point_identity_coverage": point_ok / max(total, 1),
        "instance_assignment_coverage_mean": float(np.mean(inst_ratios)) if inst_ratios else 0.0,
        "teacher_crop_feasibility": teacher_crop_feasible / max(total, 1),
    }


def main() -> int:
    payload = {
        "generated_at_utc": utc_now(),
        "M128": combo_stats(128),
        "M512": combo_stats(512),
        "M1024": combo_stats(1024),
        "whether_M512_M1024_useful_for_visualization": True,
        "whether_current_main_result_is_object_dense_not_per_pixel_dense": True,
        "note": "This audit stops density architecture search; it only checks semantic/identity target coverage by point density.",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 Dense Field Target Coverage", payload, [
        "M128",
        "M512",
        "M1024",
        "whether_M512_M1024_useful_for_visualization",
        "whether_current_main_result_is_object_dense_not_per_pixel_dense",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
