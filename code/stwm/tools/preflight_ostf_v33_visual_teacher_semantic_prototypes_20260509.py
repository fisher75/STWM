#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import V30_POINTODYSSEY_CACHE, mask_path_for_frame

REPORT = ROOT / "reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_VISUAL_TEACHER_PREFLIGHT_20260509.md"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    sample = next(V30_POINTODYSSEY_CACHE.glob("M128_H32/train/*.npz"), None)
    crop_ok = False
    mask_crop_ok = False
    crop_shape = None
    if sample:
        z = np.load(sample, allow_pickle=True)
        frame = Path(str(np.asarray(z["frame_paths"], dtype=object)[-1]))
        if frame.exists():
            try:
                im = Image.open(frame)
                crop_shape = list(np.asarray(im.crop((0, 0, min(64, im.width), min(64, im.height)))).shape)
                crop_ok = True
            except Exception:
                crop_ok = False
        mp = mask_path_for_frame(frame)
        if mp.exists():
            try:
                _ = Image.open(mp)
                mask_crop_ok = True
            except Exception:
                mask_crop_ok = False
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "DINOv2_available": has_module("dinov2") or has_module("torchvision"),
        "CLIP_open_clip_available": has_module("open_clip") or has_module("clip"),
        "SigLIP_available": has_module("transformers"),
        "SAM2_available": has_module("sam2"),
        "rgb_crop_extraction_available": crop_ok,
        "mask_crop_extraction_available": mask_crop_ok,
        "example_crop_shape": crop_shape,
        "teacher_cache_cost_estimate": {
            "M128_H32_samples": len(list(V30_POINTODYSSEY_CACHE.glob("M128_H32/*/*.npz"))),
            "future_supervision_embeddings_expensive": True,
            "recommended_first_teacher": "DINOv2_or_torchvision_feature_if_available",
        },
        "observed_teacher_embeddings_as_input_allowed": True,
        "future_teacher_embeddings_input_allowed": False,
        "future_teacher_embeddings_supervision_only": True,
        "visual_teacher_semantic_targets_available": False,
        "exact_blocker": "No teacher prototype cache has been materialized in this round; this is preflight only.",
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33 Visual Teacher Semantic Prototype Preflight", payload, [
        "DINOv2_available",
        "CLIP_open_clip_available",
        "SigLIP_available",
        "SAM2_available",
        "rgb_crop_extraction_available",
        "mask_crop_extraction_available",
        "visual_teacher_semantic_targets_available",
        "future_teacher_embeddings_supervision_only",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
