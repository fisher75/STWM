#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np

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


def _masked_rgb_mean_std(rgb_chw: np.ndarray, mask_chw: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_chw, dtype=np.float32)
    mask = np.asarray(mask_chw, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[0]
    denom = float(np.clip(mask.sum(), 1e-6, None))
    weighted = (rgb * mask[None, ...]).reshape(3, -1)
    mean = weighted.sum(axis=-1) / denom
    std = np.sqrt((((rgb - mean[:, None, None]) ** 2) * mask[None, ...]).reshape(3, -1).sum(axis=-1) / denom)
    return np.concatenate([mean, std], axis=0).astype(np.float32)


def _quantiles(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size <= 0:
        return {k: 0.0 for k in ["q50", "q70", "q80", "q90", "q95"]}
    return {
        "q50": float(np.quantile(arr, 0.50)),
        "q70": float(np.quantile(arr, 0.70)),
        "q80": float(np.quantile(arr, 0.80)),
        "q90": float(np.quantile(arr, 0.90)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def parse_args() -> Any:
    p = ArgumentParser(description="Build drift-calibrated teacher semantic cache v5 for TUSB-v3.2")
    p.add_argument("--predecode-cache-root", default=str(ROOT / "data/processed/stage2_tusb_v3_predecode_cache_20260418"))
    p.add_argument("--source-teacher-cache-root", default=str(ROOT / "data/processed/stage2_teacher_semantic_cache_v4_appearance_20260418"))
    p.add_argument("--teacher-cache-root", default=str(ROOT / "data/processed/stage2_teacher_semantic_cache_v5_driftcal_20260419"))
    p.add_argument("--output-json", default=str(ROOT / "reports/stage2_tusb_v3p2_appearance_signal_20260419.json"))
    p.add_argument("--output-md", default=str(ROOT / "docs/STAGE2_TUSB_V3P2_APPEARANCE_SIGNAL_20260419.md"))
    return p.parse_args()


def main() -> None:
    prev._apply_process_title_normalization()
    args = parse_args()
    predecode_root = Path(args.predecode_cache_root)
    source_teacher_root = Path(args.source_teacher_cache_root)
    teacher_root = Path(args.teacher_cache_root)
    index_path = predecode_root / "index.json"
    source_index_path = source_teacher_root / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing predecode cache index: {index_path}")
    if not source_index_path.exists():
        raise FileNotFoundError(f"missing source teacher cache index: {source_index_path}")

    predecode_index = json.loads(index_path.read_text(encoding="utf-8"))
    source_index = json.loads(source_index_path.read_text(encoding="utf-8"))
    entries = predecode_index.get("entries", {}) if isinstance(predecode_index.get("entries", {}), dict) else {}
    source_entries = source_index.get("entries", {}) if isinstance(source_index.get("entries", {}), dict) else {}
    teacher_root.mkdir(parents=True, exist_ok=True)

    v4_report = ROOT / "reports/stage2_tusb_v3p1_appearance_teacher_20260418.json"
    blocked = {}
    if v4_report.exists():
        try:
            blocked = json.loads(v4_report.read_text(encoding="utf-8")).get("current_env_blocked_backends", {})
        except Exception:
            blocked = {}

    out_entries: Dict[str, str] = {}
    per_dataset_teacher_drift: Dict[str, List[float]] = {}
    per_dataset_local_delta: Dict[str, List[float]] = {}
    per_dataset_combined: Dict[str, List[float]] = {}
    all_teacher_drift: List[float] = []
    all_local_delta: List[float] = []
    all_combined: List[float] = []
    written = 0
    reused = 0

    for key, source_path in sorted(entries.items()):
        predecode_npz = Path(str(source_path))
        source_teacher_npz = Path(str(source_entries.get(str(key), "")))
        if not predecode_npz.exists():
            continue
        out_path = teacher_root / f"{prev._safe_name(key)}.npz"
        out_entries[str(key)] = str(out_path)
        if out_path.exists():
            reused += 1
            continue

        with np.load(predecode_npz, allow_pickle=True) as payload:
            rgb_temporal = np.asarray(payload["semantic_rgb_crop_temporal"], dtype=np.float32)
            mask_temporal = np.asarray(payload["semantic_mask_crop_temporal"], dtype=np.float32)
            temporal_valid = np.asarray(payload["semantic_temporal_valid"], dtype=bool)
            meta = dict(payload["meta_json"].item())
        teacher_prior = None
        early_late_teacher_cosine = None
        if source_teacher_npz.exists():
            with np.load(source_teacher_npz, allow_pickle=True) as teacher_payload:
                if "semantic_teacher_prior" in teacher_payload:
                    teacher_prior = np.asarray(teacher_payload["semantic_teacher_prior"], dtype=np.float32)
                if "early_late_teacher_cosine" in teacher_payload:
                    early_late_teacher_cosine = np.asarray(teacher_payload["early_late_teacher_cosine"], dtype=np.float32)
        if teacher_prior is None:
            entity_count = int(rgb_temporal.shape[0])
            teacher_prior = np.zeros((entity_count, 512), dtype=np.float32)
        entity_count = int(teacher_prior.shape[0])
        local_delta = np.zeros((entity_count,), dtype=np.float32)
        if early_late_teacher_cosine is None or int(np.asarray(early_late_teacher_cosine).shape[0]) != entity_count:
            early_late_teacher_cosine = np.ones((entity_count,), dtype=np.float32)

        dataset_name = str(meta.get("dataset", "unknown"))
        ds_teacher = per_dataset_teacher_drift.setdefault(dataset_name, [])
        ds_local = per_dataset_local_delta.setdefault(dataset_name, [])
        ds_combined = per_dataset_combined.setdefault(dataset_name, [])

        for entity_idx in range(entity_count):
            valid_steps = np.flatnonzero(temporal_valid[entity_idx])
            if valid_steps.size >= 2:
                first_idx = int(valid_steps[0])
                last_idx = int(valid_steps[-1])
                early_sig = _masked_rgb_mean_std(rgb_temporal[entity_idx, first_idx], mask_temporal[entity_idx, first_idx])
                late_sig = _masked_rgb_mean_std(rgb_temporal[entity_idx, last_idx], mask_temporal[entity_idx, last_idx])
                local_delta_val = float(np.mean(np.abs(early_sig - late_sig)))
                local_delta[entity_idx] = local_delta_val
            teacher_drift = float(1.0 - float(early_late_teacher_cosine[entity_idx]))
            combined = max(teacher_drift, 0.60 * teacher_drift + 0.40 * min(local_delta[entity_idx] / 0.25, 2.0))
            ds_teacher.append(teacher_drift)
            ds_local.append(float(local_delta[entity_idx]))
            ds_combined.append(float(combined))
            all_teacher_drift.append(teacher_drift)
            all_local_delta.append(float(local_delta[entity_idx]))
            all_combined.append(float(combined))

        np.savez_compressed(
            out_path,
            semantic_teacher_prior=teacher_prior.astype(np.float32),
            early_late_teacher_cosine=np.asarray(early_late_teacher_cosine, dtype=np.float32),
            local_appearance_delta=local_delta.astype(np.float32),
            combined_appearance_drift=np.asarray(ds_combined[-entity_count:], dtype=np.float32),
            backend=np.asarray("clip_vit-b_16_temporal_weighted_masked_mean_v5_driftcal", dtype=object),
            feature_dim=np.asarray(int(teacher_prior.shape[-1]), dtype=np.int64),
            source_cache_path=np.asarray(str(predecode_npz), dtype=object),
        )
        written += 1

    prev.write_json(
        teacher_root / "index.json",
        {
            "generated_at_utc": now_iso(),
            "teacher_cache_root": str(teacher_root),
            "backend": "clip_vit-b_16_temporal_weighted_masked_mean_v5_driftcal",
            "feature_dim": 512,
            "entries": out_entries,
        },
    )

    per_dataset_quantiles = {
        ds: {
            "teacher_drift_quantiles": _quantiles(vals),
            "local_appearance_delta_quantiles": _quantiles(per_dataset_local_delta.get(ds, [])),
            "combined_drift_quantiles": _quantiles(per_dataset_combined.get(ds, [])),
        }
        for ds, vals in sorted(per_dataset_teacher_drift.items())
    }
    global_combined = np.asarray(all_combined, dtype=np.float32)
    global_threshold = float(np.quantile(global_combined, 0.80)) if global_combined.size > 0 else 0.18
    appearance_high_ratio = float(np.mean(global_combined >= global_threshold)) if global_combined.size > 0 else 0.0

    payload = {
        "generated_at_utc": now_iso(),
        "teacher_cache_root": str(teacher_root),
        "teacher_cache_index": str(teacher_root / "index.json"),
        "source_teacher_cache_root": str(source_teacher_root),
        "source_predecode_cache_root": str(predecode_root),
        "current_env_blocked_backends": blocked,
        "chosen_teacher_prior_v5": "clip_vit-b_16_temporal_weighted_masked_mean_v5_driftcal",
        "appearance_drift_signal_available": bool(global_combined.size > 0),
        "appearance_drift_high_ratio": float(appearance_high_ratio),
        "global_combined_drift_high_threshold": float(global_threshold),
        "per_dataset_drift_quantiles": per_dataset_quantiles,
        "early_late_teacher_cosine_stats": _quantiles([1.0 - x for x in all_teacher_drift]),
        "local_appearance_delta_stats": _quantiles(all_local_delta),
        "cached_entry_count": int(len(out_entries)),
        "newly_written_count": int(written),
        "reused_existing_count": int(reused),
        "current_env_blocked_backends_reasoning": (
            "DINOv2-like and SigLIP-like backends remain unavailable in the current environment; "
            "v5 therefore strengthens the existing CLIP-family prior via drift calibration instead of changing thesis."
        ),
    }
    prev.write_json(args.output_json, payload)
    prev.write_md(
        args.output_md,
        [
            "# Stage2 TUSB-V3.2 Appearance Signal 20260419",
            "",
            f"- chosen_teacher_prior_v5: {payload['chosen_teacher_prior_v5']}",
            f"- appearance_drift_signal_available: {payload['appearance_drift_signal_available']}",
            f"- appearance_drift_high_ratio: {payload['appearance_drift_high_ratio']:.6f}",
            f"- global_combined_drift_high_threshold: {payload['global_combined_drift_high_threshold']:.6f}",
            "",
            "## Per-Dataset Drift Quantiles",
            "",
            *[
                f"- {name}: combined_q80={stats['combined_drift_quantiles']['q80']:.6f}, local_q80={stats['local_appearance_delta_quantiles']['q80']:.6f}"
                for name, stats in sorted(payload["per_dataset_drift_quantiles"].items())
            ],
        ],
    )
    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
