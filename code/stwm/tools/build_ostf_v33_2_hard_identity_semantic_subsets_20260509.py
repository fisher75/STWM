#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import V30_POINTODYSSEY_CACHE, V33_IDENTITY_ROOT, scalar

REPORT = ROOT / "reports/stwm_ostf_v33_2_hard_identity_semantic_subset_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_2_HARD_IDENTITY_SEMANTIC_SUBSET_20260509.md"
MANIFEST = ROOT / "manifests/ostf_v33_2_hard_identity_semantic/H32_M128_seed42.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-name", default="clip_vit_b32_local")
    parser.add_argument("--max-items", type=int, default=128)
    args = parser.parse_args()
    visual_root = ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey" / args.teacher_name / "test"
    entries = []
    pos = neg = hard_neg = occ = conf = same_video_neg = teacher_conf = 0
    for path in sorted((V30_POINTODYSSEY_CACHE / "M128_H32/test").glob("*.npz")):
        z = np.load(path, allow_pickle=True)
        uid = str(scalar(z, "video_uid", path.stem))
        sidecar = V33_IDENTITY_ROOT / "test" / f"{uid}.npz"
        visual = visual_root / f"{uid}.npz"
        if not sidecar.exists() or not visual.exists():
            continue
        s = np.load(sidecar, allow_pickle=True)
        same = np.asarray(s["fut_same_instance_as_obs"]).astype(bool)
        mask = np.asarray(s["fut_instance_available_mask"]).astype(bool)
        neg_count = int((~same & mask).sum())
        pos_count = int((same & mask).sum())
        if neg_count == 0:
            continue
        fut_vis = np.asarray(s["fut_point_visible_target"]).astype(bool)
        obs_vis = np.asarray(z["obs_vis"]).astype(bool)
        reappear = (~obs_vis[:, -1, None] & fut_vis).sum()
        hard_score = neg_count + int(reappear)
        entries.append(
            {
                "uid": uid,
                "cache_path": str(path.relative_to(ROOT)),
                "identity_sidecar_path": str(sidecar.relative_to(ROOT)),
                "visual_sidecar_path": str(visual.relative_to(ROOT)),
                "H": 32,
                "M": 128,
                "hard_negative_count": neg_count,
                "positive_count": pos_count,
                "occlusion_reappearance_count": int(reappear),
                "hard_score": hard_score,
                "tags": ["identity_hard_negative", "visual_teacher_available"] + (["occlusion_reappearance"] if reappear else []),
            }
        )
        pos += pos_count
        neg += neg_count
        hard_neg += neg_count
        occ += int(reappear)
        conf += int(neg_count > 0)
        same_video_neg += neg_count
        teacher_conf += int(visual.exists())
    entries = sorted(entries, key=lambda x: x["hard_score"], reverse=True)[: args.max_items]
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    dump_json(MANIFEST, {"generated_at_utc": utc_now(), "entries": entries})
    total = pos + neg
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "manifest_path": str(MANIFEST.relative_to(ROOT)),
        "item_count": len(entries),
        "positive_ratio": pos / max(total, 1),
        "negative_ratio": neg / max(total, 1),
        "hard_negative_count": hard_neg,
        "occlusion_reappearance_count": occ,
        "confuser_count": conf,
        "same_video_negative_count": same_video_neg,
        "teacher_semantic_confuser_count": teacher_conf,
        "whether_balanced_eval_possible": bool(len(entries) > 0 and neg > 0 and pos > 0),
        "exact_blocker": None if entries else "No test items with both visual sidecar and negative future instance labels.",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.2 Hard Identity Semantic Subset", payload, ["manifest_path", "item_count", "positive_ratio", "negative_ratio", "hard_negative_count", "occlusion_reappearance_count", "confuser_count", "teacher_semantic_confuser_count", "whether_balanced_eval_possible", "exact_blocker"])
    print(REPORT.relative_to(ROOT))
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
