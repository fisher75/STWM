#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import V30_POINTODYSSEY_CACHE, mask_path_for_frame

REPORT = ROOT / "reports/stwm_ostf_v33_visual_teacher_preflight_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_VISUAL_TEACHER_PREFLIGHT_20260509.md"
REPORT_V33_2 = ROOT / "reports/stwm_ostf_v33_2_visual_teacher_preflight_20260509.json"
DOC_V33_2 = ROOT / "docs/STWM_OSTF_V33_2_VISUAL_TEACHER_PREFLIGHT_20260509.md"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def clip_cache_path() -> Path | None:
    for root in (Path("/home/chen034/.cache/clip"), Path("/raid/chen034/.cache/clip")):
        path = root / "ViT-B-32.pt"
        if path.exists():
            return path
    return None


def clip_dryrun(crop: Image.Image) -> dict[str, Any]:
    cache = clip_cache_path()
    if cache is None or not has_module("clip"):
        return {
            "teacher_model_loaded": False,
            "teacher_forward_dryrun_passed": False,
            "teacher_embedding_dim": None,
            "recommended_first_teacher": None,
            "exact_blocker": "OpenAI CLIP package or local ViT-B-32.pt is missing; manual_model_download_needed.",
        }
    try:
        import clip  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(cache.parent))
        model.eval()
        with torch.no_grad():
            x = preprocess(crop.convert("RGB")).unsqueeze(0).to(device)
            emb = model.encode_image(x).float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return {
            "teacher_model_loaded": True,
            "teacher_forward_dryrun_passed": True,
            "teacher_embedding_dim": int(emb.shape[-1]),
            "recommended_first_teacher": "clip_vit_b32_local",
            "teacher_checkpoint_path": str(cache),
            "device": device,
            "exact_blocker": None,
        }
    except Exception as exc:
        return {
            "teacher_model_loaded": False,
            "teacher_forward_dryrun_passed": False,
            "teacher_embedding_dim": None,
            "recommended_first_teacher": None,
            "teacher_checkpoint_path": str(cache),
            "exact_blocker": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    sample = next(V30_POINTODYSSEY_CACHE.glob("M128_H32/train/*.npz"), None)
    crop_ok = False
    mask_crop_ok = False
    crop_shape = None
    crop_img = Image.new("RGB", (64, 64), "white")
    if sample:
        z = np.load(sample, allow_pickle=True)
        frame = Path(str(np.asarray(z["frame_paths"], dtype=object)[-1]))
        if frame.exists():
            try:
                im = Image.open(frame)
                crop_img = im.crop((0, 0, min(64, im.width), min(64, im.height))).resize((64, 64))
                crop_shape = list(np.asarray(crop_img).shape)
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
    clip_status = clip_dryrun(crop_img)
    dinov2_local = has_module("dinov2") and bool(list((ROOT / "models").glob("*dinov2*")))
    siglip_local = False
    sam2_local = has_module("sam2") or (ROOT / "third_party/sam2").exists()
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "DINOv2_available": dinov2_local,
        "CLIP_open_clip_available": bool(clip_status["teacher_model_loaded"]),
        "SigLIP_available": siglip_local,
        "SAM2_available": sam2_local,
        "rgb_crop_extraction_available": crop_ok,
        "mask_crop_extraction_available": mask_crop_ok,
        "example_crop_shape": crop_shape,
        "teacher_cache_cost_estimate": {
            "M128_H32_samples": len(list(V30_POINTODYSSEY_CACHE.glob("M128_H32/*/*.npz"))),
            "future_supervision_embeddings_expensive": True,
            "recommended_first_teacher": clip_status["recommended_first_teacher"],
        },
        "class_semantic_available": False,
        "visual_teacher_semantic_needed": True,
        "recommended_first_teacher": clip_status["recommended_first_teacher"],
        "teacher_model_loaded": bool(clip_status["teacher_model_loaded"]),
        "teacher_forward_dryrun_passed": bool(clip_status["teacher_forward_dryrun_passed"]),
        "teacher_embedding_dim": clip_status["teacher_embedding_dim"],
        "teacher_checkpoint_path": clip_status.get("teacher_checkpoint_path"),
        "observed_teacher_embeddings_as_input_allowed": True,
        "future_teacher_embeddings_input_allowed": False,
        "future_teacher_embeddings_supervision_only": True,
        "visual_teacher_semantic_targets_available": False,
        "manual_model_download_needed": not bool(clip_status["teacher_model_loaded"]),
        "exact_blocker": clip_status["exact_blocker"] or "Teacher dryrun passed; prototype cache has not been materialized by this preflight.",
    }
    dump_json(REPORT, payload)
    dump_json(REPORT_V33_2, payload)
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
    write_doc(DOC_V33_2, "STWM OSTF V33.2 Visual Teacher Semantic Prototype Preflight", payload, [
        "DINOv2_available",
        "CLIP_open_clip_available",
        "SigLIP_available",
        "SAM2_available",
        "teacher_model_loaded",
        "teacher_forward_dryrun_passed",
        "teacher_embedding_dim",
        "recommended_first_teacher",
        "future_teacher_embeddings_input_allowed",
        "future_teacher_embeddings_supervision_only",
        "exact_blocker",
    ])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
