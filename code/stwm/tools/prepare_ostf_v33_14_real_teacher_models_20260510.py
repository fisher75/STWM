#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v33_14_real_teacher_model_prepare_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_14_REAL_TEACHER_MODEL_PREPARE_20260510.md"
MODEL_ROOT = ROOT / "models/semantic_teachers"


TEACHERS = {
    "dinov2_base": {"hf_id": "facebook/dinov2-base", "kind": "auto", "size": 224},
    "dinov2_large": {"hf_id": "facebook/dinov2-large", "kind": "auto", "size": 224},
    "siglip_base": {"hf_id": "google/siglip-base-patch16-224", "kind": "auto", "size": 224},
    "clip_vit_l14": {"hf_id": "openai/clip-vit-large-patch14", "kind": "auto", "size": 224},
}


def pkg_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def try_transformers_teacher(name: str, spec: dict[str, Any], *, local_only: bool, device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {
        "model_available": False,
        "weights_available": False,
        "package_available": pkg_available("transformers"),
        "forward_dryrun_passed": False,
        "embedding_dim": None,
        "preprocessing": {"input": "[1,3,224,224]", "normalization": "teacher default not used for dryrun random tensor"},
        "model_path": str((MODEL_ROOT / name).relative_to(ROOT)),
        "hf_id": spec["hf_id"],
        "exact_blocker": None,
    }
    if not out["package_available"]:
        out["exact_blocker"] = "transformers package unavailable"
        return out
    try:
        from transformers import AutoModel
        if name == "clip_vit_l14":
            from transformers import CLIPVisionModel

            model_cls = CLIPVisionModel
        elif name == "siglip_base":
            from transformers import SiglipVisionModel

            model_cls = SiglipVisionModel
        else:
            model_cls = AutoModel

        cache_dir = MODEL_ROOT / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        model = model_cls.from_pretrained(spec["hf_id"], cache_dir=str(cache_dir), local_files_only=local_only)
        model.eval().to(device)
        x = torch.rand(1, 3, int(spec.get("size", 224)), int(spec.get("size", 224)), device=device)
        with torch.no_grad():
            y = model(pixel_values=x)
        if hasattr(y, "pooler_output") and y.pooler_output is not None:
            emb = y.pooler_output
        elif hasattr(y, "last_hidden_state"):
            emb = y.last_hidden_state.mean(dim=1)
        else:
            first = y[0] if isinstance(y, (tuple, list)) else y
            emb = first.mean(dim=1) if first.ndim == 3 else first
        out.update(
            {
                "model_available": True,
                "weights_available": True,
                "forward_dryrun_passed": True,
                "embedding_dim": int(emb.shape[-1]),
                "exact_blocker": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - report exact blocker for reproducibility
        out["exact_blocker"] = f"{type(exc).__name__}: {exc}"
        if "Connection" in str(exc) or "offline" in str(exc).lower() or "not the path" in str(exc).lower():
            out["manual_download_needed"] = True
    finally:
        try:
            del model  # type: ignore[name-defined]
            torch.cuda.empty_cache()
        except Exception:
            pass
    return out


def try_sam2(device: torch.device) -> dict[str, Any]:
    out = {
        "model_available": False,
        "weights_available": False,
        "package_available": pkg_available("sam2"),
        "forward_dryrun_passed": False,
        "embedding_dim": None,
        "preprocessing": {"input": "RGB/mask feature extraction"},
        "exact_blocker": None,
    }
    if not out["package_available"]:
        out["exact_blocker"] = "sam2 package unavailable"
    return out


def main() -> int:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    local_only = os.environ.get("STWM_TEACHER_LOCAL_ONLY", "0") == "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teachers: dict[str, Any] = {}
    for name, spec in TEACHERS.items():
        teachers[name] = try_transformers_teacher(name, spec, local_only=local_only, device=device)
    teachers["sam2_mask_feature"] = try_sam2(device)
    available = [k for k, v in teachers.items() if v.get("forward_dryrun_passed")]
    ensembles = {
        "ensemble_dinov2_siglip": all(t in available for t in ["dinov2_base", "siglip_base"]),
        "ensemble_dinov2_clip_l14": all(t in available for t in ["dinov2_base", "clip_vit_l14"]),
        "ensemble_dinov2_siglip_clip_l14": all(t in available for t in ["dinov2_base", "siglip_base", "clip_vit_l14"]),
    }
    payload = {
        "generated_at_utc": utc_now(),
        "stronger_teacher_model_prepare_done": True,
        "device": str(device),
        "local_files_only": local_only,
        "teachers": teachers,
        "available_teachers": available,
        "ensemble_candidates": ensembles,
        "stronger_teacher_forward_dryrun_passed": bool(available),
        "any_stronger_teacher_available": bool(available),
        "recommended_next_step": "build_teacher_feature_cache" if available else "fix_teacher_model_availability",
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V33.14 Real Teacher Model Prepare",
        payload,
        ["stronger_teacher_model_prepare_done", "device", "local_files_only", "available_teachers", "stronger_teacher_forward_dryrun_passed", "ensemble_candidates", "recommended_next_step", "teachers"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
