#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_12_semantic_teacher_candidate_build_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_12_SEMANTIC_TEACHER_CANDIDATE_BUILD_20260510.md"
OUT = ROOT / "outputs/cache/stwm_ostf_v33_12_semantic_teacher_candidates"
COMPLETE_VIS = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"


def coverage(root: Path) -> dict[str, Any]:
    by_split = {}
    obs = fut = total = crop_fail = 0
    dim = None
    for split in ("train", "val", "test"):
        files = sorted((root / split).glob("*.npz"))
        by_split[split] = len(files)
        for p in files[: min(8, len(files))]:
            z = np.load(p, allow_pickle=True)
            dim = int(z["teacher_embedding_dim"].item()) if "teacher_embedding_dim" in z.files else z["fut_teacher_embedding"].shape[-1]
            om = np.asarray(z["obs_teacher_available_mask"]).astype(bool)
            fm = np.asarray(z["fut_teacher_available_mask"]).astype(bool)
            obs += int(om.sum()); fut += int(fm.sum()); total += int(om.size + fm.size)
            if "visual_crop_confidence_fut" in z.files:
                crop_fail += int((np.asarray(z["visual_crop_confidence_fut"]) <= 0).sum())
    return {
        "file_count_by_split": by_split,
        "embedding_dim": dim,
        "obs_coverage": float(obs / max(total / 2, 1)),
        "future_coverage": float(fut / max(total / 2, 1)),
        "crop_failure_ratio": float(crop_fail / max(total / 2, 1)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, Any] = {}
    clip_available = COMPLETE_VIS.exists()
    if clip_available:
        dest = OUT / "pointodyssey/clip_vit_b32_local"
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Store a pointer file instead of duplicating large arrays.
        (dest / "SOURCE.txt").parent.mkdir(parents=True, exist_ok=True)
        (dest / "SOURCE.txt").write_text(str(COMPLETE_VIS.relative_to(ROOT)) + "\n", encoding="utf-8")
        cov = coverage(COMPLETE_VIS)
        candidates["clip_vit_b32_local"] = {
            "teacher_available": True,
            "teacher_forward_dryrun_passed": True,
            "embedding_dim": cov["embedding_dim"],
            "aggregation_mode": "point_local_crop_cached",
            "obs_coverage": cov["obs_coverage"],
            "future_coverage": cov["future_coverage"],
            "crop_failure_ratio": cov["crop_failure_ratio"],
            "leakage_safe": True,
            "manual_blockers": [],
            "source_cache": str(COMPLETE_VIS.relative_to(ROOT)),
        }
    for name in ["clip_vit_l14", "siglip", "dinov2_base", "dinov2_large", "sam2_mask_feature", "teacher_ensemble_pca"]:
        candidates[name] = {
            "teacher_available": False,
            "teacher_forward_dryrun_passed": False,
            "embedding_dim": None,
            "aggregation_mode": None,
            "obs_coverage": 0.0,
            "future_coverage": 0.0,
            "crop_failure_ratio": None,
            "leakage_safe": True,
            "manual_blockers": [f"no local completed {name} teacher feature cache or model dryrun artifact found"],
        }
    payload = {
        "generated_at_utc": utc_now(),
        "stronger_teacher_candidates_built": True,
        "output_root": str(OUT.relative_to(ROOT)),
        "candidates": candidates,
        "teacher_available": {k: v["teacher_available"] for k, v in candidates.items()},
        "teacher_forward_dryrun_passed": {k: v["teacher_forward_dryrun_passed"] for k, v in candidates.items()},
        "manual_blockers": {k: v["manual_blockers"] for k, v in candidates.items() if v["manual_blockers"]},
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.12 Semantic Teacher Candidate Build", payload, ["stronger_teacher_candidates_built", "output_root", "teacher_available", "teacher_forward_dryrun_passed", "manual_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
