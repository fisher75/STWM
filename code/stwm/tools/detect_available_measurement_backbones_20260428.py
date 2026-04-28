#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import importlib.util
import json
import os


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Available Measurement Backbones V7",
        "",
        f"- selected_backbone: `{payload.get('selected_backbone')}`",
        f"- selection_reason: `{payload.get('selection_reason')}`",
        f"- no_internet_download_attempted: `{payload.get('no_internet_download_attempted')}`",
        f"- fallback_if_none: `{payload.get('fallback_if_none')}`",
        f"- available_backbones: `{payload.get('available_backbones')}`",
        f"- unavailable_backbones: `{payload.get('unavailable_backbones')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _clip_cache_files(home: Path) -> list[str]:
    candidates = [
        home / ".cache" / "clip" / "ViT-B-32.pt",
        home / ".cache" / "clip" / "ViT-B-16.pt",
    ]
    return [str(p) for p in candidates if p.exists() and p.stat().st_size > 0]


def _hf_model_cache(home: Path, needle: str) -> list[str]:
    hub = home / ".cache" / "huggingface" / "hub"
    if not hub.exists():
        return []
    return [str(p) for p in hub.glob(f"models--*{needle}*") if p.exists()]


def detect(repo_root: Path, output: Path, doc: Path) -> dict[str, Any]:
    home = Path(os.environ.get("HOME", str(Path.home()))).expanduser()
    clip_files = _clip_cache_files(home)
    hf_clip = _hf_model_cache(home, "clip")
    hf_dino = _hf_model_cache(home, "dinov2") + _hf_model_cache(home, "dino")
    hf_siglip = _hf_model_cache(home, "siglip")
    sam2_path = repo_root / "third_party" / "sam2"

    probes = {
        "local_openai_clip_vit_b_32": {
            "available": bool(clip_files) and _module_available("clip"),
            "cache_or_weight_paths": clip_files,
            "python_module_available": _module_available("clip"),
        },
        "huggingface_clip_cache": {
            "available": bool(hf_clip) and _module_available("transformers"),
            "cache_or_weight_paths": hf_clip,
            "python_module_available": _module_available("transformers"),
        },
        "huggingface_dinov2_cache": {
            "available": bool(hf_dino) and _module_available("transformers"),
            "cache_or_weight_paths": hf_dino,
            "python_module_available": _module_available("transformers"),
        },
        "huggingface_siglip_cache": {
            "available": bool(hf_siglip) and _module_available("transformers"),
            "cache_or_weight_paths": hf_siglip,
            "python_module_available": _module_available("transformers"),
        },
        "local_sam2_code_or_cache": {
            "available": sam2_path.exists(),
            "cache_or_weight_paths": [str(sam2_path)] if sam2_path.exists() else [],
            "python_module_available": False,
        },
        "stwm_crop_visual_encoder": {
            "available": (repo_root / "code" / "stwm").exists(),
            "cache_or_weight_paths": [],
            "python_module_available": True,
        },
    }
    priority = [
        "huggingface_dinov2_cache",
        "huggingface_siglip_cache",
        "local_openai_clip_vit_b_32",
        "huggingface_clip_cache",
        "local_sam2_code_or_cache",
        "stwm_crop_visual_encoder",
    ]
    selected = "stwm_crop_visual_encoder"
    reason = "fallback to existing STWM crop visual encoder"
    for name in priority:
        if bool(probes[name]["available"]):
            if name == "local_sam2_code_or_cache":
                # Code without an already wired feature extractor is not selected
                # above a directly usable crop encoder.
                continue
            selected = name
            reason = "highest-priority locally available frozen measurement backbone"
            break

    payload = {
        "generated_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "no_internet_download_attempted": True,
        "available_backbones": {k: v for k, v in probes.items() if bool(v.get("available"))},
        "unavailable_backbones": {k: v for k, v in probes.items() if not bool(v.get("available"))},
        "selected_backbone": selected,
        "selection_reason": reason,
        "fallback_if_none": "crop_encoder_feature_only",
        "probes": probes,
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--repo-root", default=os.environ.get("STWM_ROOT", "."))
    parser.add_argument("--output", default="reports/stwm_available_measurement_backbones_v7_20260428.json")
    parser.add_argument("--doc", default="docs/STWM_AVAILABLE_MEASUREMENT_BACKBONES_V7_20260428.md")
    args = parser.parse_args()
    detect(Path(args.repo_root).expanduser().resolve(), Path(args.output), Path(args.doc))


if __name__ == "__main__":
    main()
