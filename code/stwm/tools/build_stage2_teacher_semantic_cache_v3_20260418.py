#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import importlib.util
import json

import numpy as np
from PIL import Image
import torch


def _repo_root() -> Path:
    for candidate in [Path("/raid/chen034/workspace/stwm"), Path("/home/chen034/workspace/stwm")]:
        if candidate.exists():
            return candidate
    raise RuntimeError("unable to resolve STWM repo root")


ROOT = _repo_root()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in str(value))


def _apply_mask(rgb_chw: np.ndarray, mask_chw: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_chw, dtype=np.float32)
    mask = np.asarray(mask_chw, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    return np.clip(rgb * np.clip(mask, 0.0, 1.0)[None, ...], 0.0, 1.0)


def _to_pil(rgb_chw: np.ndarray) -> Image.Image:
    arr = np.transpose(np.asarray(rgb_chw, dtype=np.float32), (1, 2, 0))
    arr = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _load_clip_backend(device: str) -> Tuple[Any, Any, Any, torch.device, str, Dict[str, str]]:
    blocked: Dict[str, str] = {
        "dinov2_like": "backend_not_available_in_current_env",
        "siglip_like": "backend_not_available_in_current_env",
    }
    if importlib.util.find_spec("clip") is None:
        raise RuntimeError("clip module not installed in current environment")
    import clip  # type: ignore

    target = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    errors: Dict[str, str] = {}
    for backbone in ["ViT-B/16", "ViT-B/32"]:
        try:
            model, preprocess = clip.load(backbone, device=str(target), jit=False)
            model.eval()
            blocked["siglip_like"] = blocked.get("siglip_like", "backend_not_available_in_current_env")
            return clip, model, preprocess, target, backbone, {**blocked, **errors}
        except Exception as exc:
            errors[f"clip_{backbone.lower().replace('/', '_')}"] = repr(exc)
    raise RuntimeError(f"no CLIP backbone available: {errors}")


def parse_args() -> Any:
    p = ArgumentParser(description="Build stronger frozen teacher semantic cache v3 for TUSB context-aligned repair")
    p.add_argument("--predecode-cache-root", default=str(ROOT / "data/processed/stage2_tusb_v2_predecode_cache_20260418"))
    p.add_argument("--teacher-cache-root", default=str(ROOT / "data/processed/stage2_teacher_semantic_cache_v3_20260418"))
    p.add_argument("--output-json", default=str(ROOT / "reports/stage2_tusb_teacher_prior_v3_20260418.json"))
    p.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_TUSB_TEACHER_PRIOR_V3_20260418.md"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predecode_root = Path(args.predecode_cache_root)
    teacher_root = Path(args.teacher_cache_root)
    index_path = predecode_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing predecode cache index: {index_path}")
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index_payload.get("entries", {}) if isinstance(index_payload.get("entries", {}), dict) else {}

    clip_mod, clip_model, clip_preprocess, clip_device, chosen_backbone, blocked = _load_clip_backend(str(args.device))
    teacher_root.mkdir(parents=True, exist_ok=True)

    out_entries: Dict[str, str] = {}
    written = 0
    reused = 0
    per_dataset_counts: Dict[str, int] = {}
    feature_dim = 512
    chosen_teacher_prior_v3 = f"clip_{chosen_backbone.lower().replace('/', '_')}_temporal_weighted_masked_mean_v3"

    for key, source_path in sorted(entries.items()):
        npz_path = Path(str(source_path))
        if not npz_path.exists():
            continue
        out_path = teacher_root / f"{_safe_name(key)}.npz"
        out_entries[str(key)] = str(out_path)
        if out_path.exists():
            reused += 1
            continue
        with np.load(npz_path, allow_pickle=True) as payload:
            rgb_temporal = np.asarray(payload["semantic_rgb_crop_temporal"], dtype=np.float32)
            mask_temporal = np.asarray(payload["semantic_mask_crop_temporal"], dtype=np.float32)
            temporal_valid = np.asarray(payload["semantic_temporal_valid"], dtype=bool)
            rgb_now = np.asarray(payload["semantic_rgb_crop"], dtype=np.float32)
            mask_now = np.asarray(payload["semantic_mask_crop"], dtype=np.float32)
            meta = payload["meta_json"].item()
        dataset_name = str(meta.get("dataset", "unknown"))
        per_dataset_counts[dataset_name] = per_dataset_counts.get(dataset_name, 0) + 1
        entity_count = int(rgb_now.shape[0])
        priors = np.zeros((entity_count, feature_dim), dtype=np.float32)
        for entity_idx in range(entity_count):
            images: List[Image.Image] = []
            weights: List[float] = []
            for step in range(rgb_temporal.shape[1]):
                if not bool(temporal_valid[entity_idx, step]):
                    continue
                masked = _apply_mask(rgb_temporal[entity_idx, step], mask_temporal[entity_idx, step])
                images.append(_to_pil(masked))
                fg_ratio = float(np.asarray(mask_temporal[entity_idx, step], dtype=np.float32).mean())
                weights.append(max(fg_ratio, 1e-3) * (1.0 + 0.10 * float(step + 1)))
            if not images:
                masked_now = _apply_mask(rgb_now[entity_idx], mask_now[entity_idx])
                images.append(_to_pil(masked_now))
                weights.append(max(float(np.asarray(mask_now[entity_idx], dtype=np.float32).mean()), 1e-3))
            with torch.no_grad():
                tensors = torch.stack([clip_preprocess(img) for img in images], dim=0)
                tensors = tensors.to(next(clip_model.parameters()).device)
                feats = clip_model.encode_image(tensors).float()
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                weight_t = torch.tensor(weights, dtype=feats.dtype, device=feats.device)
                weight_t = weight_t / weight_t.sum().clamp_min(1e-6)
                prior = (feats * weight_t[:, None]).sum(dim=0)
                prior = prior / prior.norm().clamp_min(1e-6)
            priors[entity_idx] = prior.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(
            out_path,
            semantic_teacher_prior=priors,
            backend=np.asarray(chosen_teacher_prior_v3, dtype=object),
            feature_dim=np.asarray(int(feature_dim), dtype=np.int64),
            source_cache_path=np.asarray(str(npz_path), dtype=object),
        )
        written += 1

    write_json(
        teacher_root / "index.json",
        {
            "generated_at_utc": now_iso(),
            "teacher_cache_root": str(teacher_root),
            "backend": chosen_teacher_prior_v3,
            "feature_dim": int(feature_dim),
            "entries": out_entries,
        },
    )

    payload = {
        "generated_at_utc": now_iso(),
        "teacher_cache_root": str(teacher_root),
        "teacher_cache_index": str(teacher_root / "index.json"),
        "source_predecode_cache_root": str(predecode_root),
        "current_env_blocked_backends": blocked,
        "chosen_teacher_prior_v3": chosen_teacher_prior_v3,
        "why_this_is_strongest_available_frozen_prior_in_current_env": (
            "DINOv2-like and SigLIP-like backends are not available in the current environment; "
            f"{chosen_backbone} is the strongest CLIP-family backbone that loaded successfully, "
            "and v3 uses temporal weighted masked aggregation instead of plain temporal mean."
        ),
        "feature_dim": int(feature_dim),
        "cached_entry_count": int(len(out_entries)),
        "newly_written_count": int(written),
        "reused_existing_count": int(reused),
        "per_dataset_counts": per_dataset_counts,
        "frozen_semantic_source_only": True,
        "teacher_is_mainline_semantic_source": False,
    }
    write_json(args.output_json, payload)
    write_md(
        args.output_md,
        [
            "# Stage2 TUSB Teacher Prior V3 20260418",
            "",
            f"- chosen_teacher_prior_v3: {payload['chosen_teacher_prior_v3']}",
            f"- feature_dim: {payload['feature_dim']}",
            f"- cached_entry_count: {payload['cached_entry_count']}",
            f"- newly_written_count: {payload['newly_written_count']}",
            f"- reused_existing_count: {payload['reused_existing_count']}",
            "- teacher_is_mainline_semantic_source: false",
            "",
            "## Current Env Blocked Backends",
            "",
            *[f"- {name}: {reason}" for name, reason in sorted(payload["current_env_blocked_backends"].items())],
            "",
            "## Per Dataset Counts",
            "",
            *[f"- {name}: {count}" for name, count in sorted(per_dataset_counts.items())],
            "",
            "## Selection Rationale",
            "",
            f"- {payload['why_this_is_strongest_available_frozen_prior_in_current_env']}",
        ],
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
