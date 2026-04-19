#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import torch

from stwm.tools import build_stage2_teacher_semantic_cache_v3_20260418 as prev


ROOT = prev.ROOT


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _masked_rgb_signature(rgb_chw: np.ndarray, mask_chw: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_chw, dtype=np.float32)
    mask = np.asarray(mask_chw, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    denom = float(np.clip(mask.sum(), 1e-6, None))
    vec = (rgb * mask[None, ...]).reshape(3, -1).sum(axis=-1) / denom
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.zeros((3,), dtype=np.float32)
    return (vec / norm).astype(np.float32)


def _weighted_temporal_prior(feats: torch.Tensor, fg_weights: List[float], early_late_cosine: float) -> torch.Tensor:
    step_count = int(feats.shape[0])
    if step_count <= 1:
        prior = feats[0]
        return prior / prior.norm().clamp_min(1e-6)
    drift = max(0.0, 1.0 - float(early_late_cosine))
    time_pos = torch.linspace(0.0, 1.0, steps=step_count, device=feats.device, dtype=feats.dtype)
    temporal_bias = 1.0 + float(drift) * time_pos
    center = torch.nn.functional.normalize(feats.mean(dim=0), dim=-1)
    stability = ((feats * center[None, :]).sum(dim=-1).clamp(-1.0, 1.0) + 1.0) * 0.5
    fg = torch.tensor(fg_weights, dtype=feats.dtype, device=feats.device)
    weights = fg * temporal_bias * (0.5 + 0.5 * stability)
    weights = weights / weights.sum().clamp_min(1e-6)
    prior = (feats * weights[:, None]).sum(dim=0)
    return prior / prior.norm().clamp_min(1e-6)


def parse_args() -> Any:
    p = ArgumentParser(description="Build appearance-drift-aware teacher semantic cache v4 for TUSB-v3.1")
    p.add_argument("--predecode-cache-root", default=str(ROOT / "data/processed/stage2_tusb_v2_predecode_cache_20260418"))
    p.add_argument("--teacher-cache-root", default=str(ROOT / "data/processed/stage2_teacher_semantic_cache_v4_appearance_20260418"))
    p.add_argument("--output-json", default=str(ROOT / "reports/stage2_tusb_v3p1_appearance_teacher_20260418.json"))
    p.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_TUSB_V3P1_APPEARANCE_TEACHER_20260418.md"))
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    prev._apply_process_title_normalization()
    args = parse_args()
    predecode_root = Path(args.predecode_cache_root)
    teacher_root = Path(args.teacher_cache_root)
    index_path = predecode_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing predecode cache index: {index_path}")
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index_payload.get("entries", {}) if isinstance(index_payload.get("entries", {}), dict) else {}
    clip_mod, clip_model, clip_preprocess, clip_device, chosen_backbone, blocked = prev._load_clip_backend(str(args.device))
    teacher_root.mkdir(parents=True, exist_ok=True)

    out_entries: Dict[str, str] = {}
    reused = 0
    written = 0
    per_dataset_counts: Dict[str, int] = {}
    feature_dim = 512
    chosen_teacher_prior_v4 = f"clip_{chosen_backbone.lower().replace('/', '_')}_temporal_drift_aware_masked_mean_v4"
    early_late_cosines: List[float] = []

    for key, source_path in sorted(entries.items()):
        npz_path = Path(str(source_path))
        if not npz_path.exists():
            continue
        out_path = teacher_root / f"{prev._safe_name(key)}.npz"
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
        entity_cos = np.zeros((entity_count,), dtype=np.float32)
        for entity_idx in range(entity_count):
            images: List[Any] = []
            fg_weights: List[float] = []
            rgb_sigs: List[np.ndarray] = []
            for step in range(rgb_temporal.shape[1]):
                if not bool(temporal_valid[entity_idx, step]):
                    continue
                masked = prev._apply_mask(rgb_temporal[entity_idx, step], mask_temporal[entity_idx, step])
                images.append(prev._to_pil(masked))
                fg_ratio = float(np.asarray(mask_temporal[entity_idx, step], dtype=np.float32).mean())
                fg_weights.append(max(fg_ratio, 1e-3))
                rgb_sigs.append(_masked_rgb_signature(rgb_temporal[entity_idx, step], mask_temporal[entity_idx, step]))
            if not images:
                masked_now = prev._apply_mask(rgb_now[entity_idx], mask_now[entity_idx])
                images.append(prev._to_pil(masked_now))
                fg_weights.append(max(float(np.asarray(mask_now[entity_idx], dtype=np.float32).mean()), 1e-3))
                rgb_sigs.append(_masked_rgb_signature(rgb_now[entity_idx], mask_now[entity_idx]))
            early_late_cosine = 1.0
            if len(rgb_sigs) >= 2:
                early_late_cosine = float(np.clip(np.dot(rgb_sigs[0], rgb_sigs[-1]), -1.0, 1.0))
            entity_cos[entity_idx] = float(early_late_cosine)
            early_late_cosines.append(float(early_late_cosine))
            with torch.no_grad():
                tensors = torch.stack([clip_preprocess(img) for img in images], dim=0).to(next(clip_model.parameters()).device)
                feats = clip_model.encode_image(tensors).float()
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                prior = _weighted_temporal_prior(feats, fg_weights, early_late_cosine)
            priors[entity_idx] = prior.detach().cpu().numpy().astype(np.float32)
        np.savez_compressed(
            out_path,
            semantic_teacher_prior=priors,
            early_late_teacher_cosine=entity_cos.astype(np.float32),
            backend=np.asarray(chosen_teacher_prior_v4, dtype=object),
            feature_dim=np.asarray(int(feature_dim), dtype=np.int64),
            source_cache_path=np.asarray(str(npz_path), dtype=object),
        )
        written += 1

    prev.write_json(
        teacher_root / "index.json",
        {
            "generated_at_utc": now_iso(),
            "teacher_cache_root": str(teacher_root),
            "backend": chosen_teacher_prior_v4,
            "feature_dim": int(feature_dim),
            "entries": out_entries,
        },
    )

    cosine_arr = np.asarray(early_late_cosines, dtype=np.float32) if early_late_cosines else np.zeros((0,), dtype=np.float32)
    cosine_stats = {
        "count": int(cosine_arr.size),
        "mean": float(cosine_arr.mean()) if cosine_arr.size else 0.0,
        "std": float(cosine_arr.std()) if cosine_arr.size else 0.0,
        "min": float(cosine_arr.min()) if cosine_arr.size else 0.0,
        "max": float(cosine_arr.max()) if cosine_arr.size else 0.0,
    }
    payload = {
        "generated_at_utc": now_iso(),
        "teacher_cache_root": str(teacher_root),
        "teacher_cache_index": str(teacher_root / "index.json"),
        "source_predecode_cache_root": str(predecode_root),
        "current_env_blocked_backends": blocked,
        "chosen_teacher_prior_v4": chosen_teacher_prior_v4,
        "appearance_drift_signal_available": bool(cosine_arr.size > 0),
        "early_late_teacher_cosine_stats": cosine_stats,
        "feature_dim": int(feature_dim),
        "cached_entry_count": int(len(out_entries)),
        "newly_written_count": int(written),
        "reused_existing_count": int(reused),
        "per_dataset_counts": per_dataset_counts,
        "teacher_is_mainline_semantic_source": False,
        "current_env_blocked_backends_reasoning": (
            "DINOv2-like and SigLIP-like backends are still unavailable in the current environment; "
            "v4 therefore remains CLIP-family but adds drift-aware temporal weighting."
        ),
    }
    prev.write_json(args.output_json, payload)
    prev.write_md(
        args.output_md,
        [
            "# Stage2 TUSB-V3.1 Appearance Teacher 20260418",
            "",
            f"- chosen_teacher_prior_v4: {payload['chosen_teacher_prior_v4']}",
            f"- appearance_drift_signal_available: {payload['appearance_drift_signal_available']}",
            f"- early_late_teacher_cosine_mean: {payload['early_late_teacher_cosine_stats']['mean']:.6f}",
            f"- early_late_teacher_cosine_std: {payload['early_late_teacher_cosine_stats']['std']:.6f}",
            f"- cached_entry_count: {payload['cached_entry_count']}",
            f"- newly_written_count: {payload['newly_written_count']}",
            f"- reused_existing_count: {payload['reused_existing_count']}",
            "",
            "## Blocked Backends",
            "",
            *[f"- {name}: {reason}" for name, reason in sorted(payload["current_env_blocked_backends"].items())],
        ],
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
