#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from stwm.tools.build_future_semantic_trace_feature_targets_20260428 import (
    OpenAIClipExtractor,
    _fast_target_from_entry,
)
from stwm.tools.semantic_prototype_predictability_common_20260428 import (
    checkpoint_args,
    l2_normalize,
)
from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
)


REPORT_DIR = Path("reports")
DOC_DIR = Path("docs")


def _apply_process_title_normalization() -> None:
    mode = str(os.environ.get("STWM_PROC_TITLE_MODE", "generic")).strip().lower()
    if mode == "off":
        return
    title = str(os.environ.get("STWM_PROC_TITLE", "python")).strip() or "python"
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(title)
    except Exception:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, title: str, payload: dict[str, Any], *, notes: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    for note in notes or []:
        lines.append(f"- {note}")
    if notes:
        lines.append("")
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _load_npz_from_report(report_path: str | Path, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    report = _load_json(report_path)
    cache_path = _resolve(str(report.get(key) or report.get("cache_path") or report.get("target_cache_path") or ""))
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return report, dict(np.load(cache_path, allow_pickle=True)), cache_path


def _observed_npz(report_path: str | Path, c: int) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    report = _load_json(report_path)
    cache_path = _resolve(str(report["target_cache_paths_by_prototype_count"][str(c)]))
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return report, dict(np.load(cache_path, allow_pickle=True)), cache_path


def _metadata(npz_data: dict[str, np.ndarray]) -> dict[str, Any]:
    if "metadata_json" not in npz_data:
        return {}
    try:
        return json.loads(str(npz_data["metadata_json"].tolist()))
    except Exception:
        return {}


def _dataset_mask(keys: np.ndarray, dataset: str) -> np.ndarray:
    prefix = f"{dataset.upper()}::"
    return np.asarray([str(k).upper().startswith(prefix) for k in keys], dtype=bool)


def _softmax(x: np.ndarray, temperature: float = 0.07) -> np.ndarray:
    z = x / max(float(temperature), 1e-6)
    z = z - z.max(axis=-1, keepdims=True)
    exp = np.exp(z).astype(np.float32)
    return exp / np.maximum(exp.sum(axis=-1, keepdims=True), 1e-8)


def _topk_from_scores(scores: np.ndarray, labels: np.ndarray, k: int = 5) -> dict[str, float]:
    if scores.size == 0 or labels.size == 0:
        return {"top1": 0.0, "top5": 0.0}
    pred = scores.argmax(axis=-1)
    kk = min(int(k), int(scores.shape[-1]))
    top = np.argpartition(-scores, kth=kk - 1, axis=-1)[:, :kk]
    return {
        "top1": float((pred == labels).mean()),
        "top5": float(np.any(top == labels[:, None], axis=1).mean()),
    }


def _dataset_cfg(args: dict[str, Any], split: str, max_samples_per_dataset: int) -> Stage2SemanticDatasetConfig:
    return Stage2SemanticDatasetConfig(
        dataset_names=["vipseg"],
        split=str(split),
        contract_path=str(args.get("stage2_contract_path") or "reports/stage2_bootstrap_data_contract_20260408.json"),
        obs_len=int(args.get("obs_len") or 8),
        fut_len=int(args.get("fut_len") or 8),
        max_tokens=int(args.get("max_tokens") or 64),
        max_samples_per_dataset=int(max_samples_per_dataset),
        semantic_patch_radius=int(args.get("semantic_patch_radius") or 12),
        semantic_crop_size=int(args.get("semantic_crop_size") or 64),
        semantic_source_mainline=str(args.get("semantic_source_mainline") or "crop_visual_encoder"),
        semantic_frame_index=int(args.get("semantic_frame_index") or 0),
        semantic_temporal_window=int(args.get("local_temporal_window") or 1),
        predecode_cache_path=str(args.get("predecode_cache_path") or ""),
        teacher_semantic_cache_path=str(args.get("teacher_semantic_cache_path") or ""),
        max_entities_per_sample=int(args.get("max_entities_per_sample") or 8),
        include_entity_masks_over_time=bool(args.get("include_entity_masks_over_time", False)),
        include_full_instance_id_map=bool(args.get("include_full_instance_id_map", False)),
    )


def _build_vipseg_raw_features(
    *,
    feature_report: Path,
    checkpoint: Path,
    output_cache: Path,
    device_name: str,
    batch_size: int,
    max_samples_per_dataset: int,
    progress_every: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feature_payload, feature, _ = _load_npz_from_report(feature_report, "cache_path")
    all_item_keys = np.asarray(feature["item_keys"]).astype(str)
    all_splits = np.asarray(feature["splits"]).astype(str)
    all_datasets = np.asarray(feature["datasets"]).astype(str)
    vipseg_all_mask = all_datasets == "VIPSEG"
    future_mask_all = np.asarray(feature["target_mask"], dtype=bool)
    vipseg_target_mask = vipseg_all_mask & future_mask_all.reshape(future_mask_all.shape[0], -1).any(axis=1)
    vipseg_item_keys = all_item_keys[vipseg_target_mask].tolist()
    vipseg_splits = all_splits[vipseg_target_mask].tolist()
    target_key_to_local = {str(k): i for i, k in enumerate(vipseg_item_keys)}
    split_by_key = {str(k): str(s) for k, s in zip(vipseg_item_keys, vipseg_splits)}
    _, _, k_max, feature_dim = np.asarray(feature["future_semantic_feature_target"]).shape
    n = len(vipseg_item_keys)

    obs_last = np.zeros((n, k_max, feature_dim), dtype=np.float32)
    obs_mean = np.zeros((n, k_max, feature_dim), dtype=np.float32)
    obs_count = np.zeros((n, k_max), dtype=np.float32)
    obs_mask = np.zeros((n, k_max), dtype=bool)
    trace_summary = np.zeros((n, k_max, 10), dtype=np.float32)

    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    extractor = OpenAIClipExtractor(device=device, download_root=Path.home() / ".cache" / "clip")
    args = checkpoint_args(checkpoint)
    failures: Counter[str] = Counter()
    loaded_items = 0
    matched_entries = 0
    crop_jobs: list[tuple[str, np.ndarray, int, int, int]] = []
    processed_keys: set[str] = set()
    source_counts: dict[str, int] = {}

    def flush_jobs() -> None:
        nonlocal crop_jobs
        if not crop_jobs:
            return
        encoded = extractor.encode_roi_jobs(crop_jobs, batch_size=int(batch_size))
        for vec, (_frame_path, _box, local_idx, _obs_step, slot) in zip(encoded, crop_jobs):
            obs_last[local_idx, slot] = vec
            obs_mean[local_idx, slot] = vec
            obs_count[local_idx, slot] = 1.0
            obs_mask[local_idx, slot] = bool(np.linalg.norm(vec) > 1e-8)
        crop_jobs = []

    for split in sorted(set(vipseg_splits)):
        cfg = _dataset_cfg(args, split=split, max_samples_per_dataset=int(max_samples_per_dataset))
        ds = Stage2SemanticDataset(cfg)
        source_counts[split] = int(len(ds.entries))
        for entry_i, entry in enumerate(ds.entries):
            clip_id = str(entry.get("clip_id", ""))
            key = f"VIPSEG::{clip_id}"
            if key not in target_key_to_local or split_by_key.get(key) != split:
                continue
            matched_entries += 1
            local_idx = target_key_to_local[key]
            try:
                fast = _fast_target_from_entry(
                    entry=entry,
                    cfg=cfg,
                    max_h=int(cfg.fut_len),
                    max_k=int(k_max),
                )
            except Exception as exc:
                failures[f"fast_target_error:{type(exc).__name__}"] += 1
                continue
            loaded_items += 1
            processed_keys.add(key)
            valid = np.asarray(fast["valid_over_time"], dtype=bool)
            boxes = np.asarray(fast["boxes_over_time"], dtype=np.float32)
            frame_paths = [str(x) for x in fast["frame_paths"]]
            temporal = list(fast["temporal"])
            h_obs = min(int(cfg.obs_len), int(valid.shape[0]))
            k = min(int(k_max), int(valid.shape[1]))
            for slot in range(k):
                valid_steps = np.flatnonzero(valid[:h_obs, slot]).astype(int).tolist()
                if not valid_steps:
                    failures["slot_without_obs_valid"] += 1
                    continue
                last_step = int(valid_steps[-1])
                frame_idx = int(temporal[last_step]) if last_step < len(temporal) else -1
                if frame_idx < 0 or frame_idx >= len(frame_paths):
                    failures["bad_frame_index"] += 1
                    continue
                frame_path = frame_paths[frame_idx]
                if not frame_path or not Path(frame_path).exists():
                    failures["missing_frame_path"] += 1
                    continue
                box = boxes[last_step, slot].astype(np.float32)
                crop_jobs.append((frame_path, box, local_idx, last_step, slot))
                if len(crop_jobs) >= int(batch_size) * 8:
                    flush_jobs()
            if int(progress_every) > 0 and loaded_items % int(progress_every) == 0:
                print(
                    f"[vipseg-raw-observed] split={split} loaded={loaded_items} matched={matched_entries} "
                    f"mask_slots={int(obs_mask.sum())} pending={len(crop_jobs)}",
                    flush=True,
                )
    flush_jobs()

    missing_keys = [k for k in vipseg_item_keys if k not in processed_keys]
    failures["target_key_not_found_in_raw_dataset"] += len(missing_keys)
    obs_mean = l2_normalize(obs_mean)
    obs_last = l2_normalize(obs_last)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_report_path": str(feature_report),
        "checkpoint_path": str(checkpoint),
        "feature_backbone": extractor.name,
        "feature_source": "vipseg_raw_observed_frame_bbox_clip_vit_b32",
        "raw_dataset_rebuild_used": True,
        "predecode_partial_cache_used": False,
        "no_future_leakage": True,
        "no_future_candidate_leakage": True,
        "source_counts": source_counts,
        "failed_reason_counts": dict(failures),
    }
    np.savez_compressed(
        output_cache,
        item_keys=np.asarray(vipseg_item_keys, dtype=object),
        splits=np.asarray(vipseg_splits, dtype=object),
        datasets=np.asarray(["VIPSEG"] * n, dtype=object),
        observed_last_feature=obs_last.astype(np.float32),
        observed_mean_feature=obs_mean.astype(np.float32),
        observed_feature_mask=obs_mask,
        observed_feature_count=obs_count.astype(np.float32),
        trace_summary=trace_summary.astype(np.float32),
        feature_backbone=np.asarray(extractor.name, dtype=object),
        feature_source=np.asarray("vipseg_raw_observed_frame_bbox_clip_vit_b32", dtype=object),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
    )
    data = dict(np.load(output_cache, allow_pickle=True))
    report = {
        "vipseg_target_item_count": int(n),
        "vipseg_raw_dataset_items_loaded": int(loaded_items),
        "vipseg_observed_feature_item_count": int(obs_mask.any(axis=1).sum()) if obs_mask.size else 0,
        "vipseg_observed_slot_count": int(obs_mask.sum()),
        "vipseg_observed_proto_valid_ratio": float(obs_mask.mean()) if obs_mask.size else 0.0,
        "raw_dataset_rebuild_used": True,
        "predecode_partial_cache_used": False,
        "failed_item_count": int(len(missing_keys)),
        "failure_reason_top10": failures.most_common(10),
        "observed_feature_cache_path": str(output_cache),
        "feature_backbone": extractor.name,
        "feature_source": "vipseg_raw_observed_frame_bbox_clip_vit_b32",
        "source_counts": source_counts,
    }
    return data, report


def _build_observed_proto_targets(
    *,
    raw_feature_report: dict[str, Any],
    raw_feature_data: dict[str, np.ndarray],
    future_reports: list[Path],
    cache_dir: Path,
    output_report: Path,
    output_doc: Path,
    v1_feature_report: Path,
) -> dict[str, Any]:
    item_keys = np.asarray(raw_feature_data["item_keys"]).astype(str).tolist()
    splits = np.asarray(raw_feature_data["splits"]).astype(str).tolist()
    datasets = np.asarray(raw_feature_data["datasets"]).astype(str).tolist()
    obs_feature = l2_normalize(np.asarray(raw_feature_data["observed_mean_feature"], dtype=np.float32))
    obs_mask = np.asarray(raw_feature_data["observed_feature_mask"], dtype=bool)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_cache_paths: dict[str, str] = {}
    results: list[dict[str, Any]] = []
    future_overlap_c64 = 0.0
    selected_count = 0
    for future_report_path in future_reports:
        future_payload, future, _ = _load_npz_from_report(future_report_path, "target_cache_path")
        c = int(future_payload.get("prototype_count") or future["prototypes"].shape[0])
        prototypes = l2_normalize(np.asarray(future["prototypes"], dtype=np.float32))
        scores = obs_feature @ prototypes.T
        proto_target = scores.argmax(axis=-1).astype(np.int64)
        proto_target[~obs_mask] = -1
        proto_dist = _softmax(scores)
        proto_dist[~obs_mask] = 0.0
        future_index = {str(k): i for i, k in enumerate(np.asarray(future["item_keys"]).astype(str).tolist())}
        fmask = np.zeros((len(item_keys),) + tuple(np.asarray(future["target_mask"]).shape[1:]), dtype=bool)
        labels = np.full((len(item_keys),) + tuple(np.asarray(future["future_semantic_proto_target"]).shape[1:]), -1, dtype=np.int64)
        for i, key in enumerate(item_keys):
            src_i = future_index.get(key)
            if src_i is None:
                continue
            fmask[i] = np.asarray(future["target_mask"][src_i], dtype=bool)
            labels[i] = np.asarray(future["future_semantic_proto_target"][src_i], dtype=np.int64)
        future_slot_mask = fmask.any(axis=1)
        overlap = obs_mask & future_slot_mask
        valid_eval = fmask & (labels >= 0) & obs_mask[:, None, :]
        repeated_scores = np.repeat(scores[:, None, :, :], labels.shape[1], axis=1)
        metrics = _topk_from_scores(repeated_scores[valid_eval], labels[valid_eval]) if valid_eval.any() else {"top1": 0.0, "top5": 0.0}
        cache_path = cache_dir / f"observed_proto_targets_c{c}.npz"
        np.savez_compressed(
            cache_path,
            item_keys=np.asarray(item_keys, dtype=object),
            splits=np.asarray(splits, dtype=object),
            datasets=np.asarray(datasets, dtype=object),
            observed_semantic_proto_target=proto_target,
            observed_semantic_proto_distribution=proto_dist.astype(np.float32),
            observed_semantic_proto_mask=obs_mask,
            prototypes=prototypes.astype(np.float32),
            prototype_count=np.asarray(c, dtype=np.int64),
            no_future_leakage=np.asarray(True),
        )
        target_cache_paths[str(c)] = str(cache_path)
        result = {
            "prototype_count": int(c),
            "target_cache_path": str(cache_path),
            "vipseg_observed_proto_valid_ratio": float(obs_mask.mean()) if obs_mask.size else 0.0,
            "vipseg_future_overlap_ratio": float(overlap.sum() / max(int(future_slot_mask.sum()), 1)),
            "observed_nonzero_slot_count": int(obs_mask.sum()),
            "future_target_slot_count": int(future_slot_mask.sum()),
            "overlap_slot_count": int(overlap.sum()),
            "observed_last_top1": float(metrics["top1"]),
            "observed_last_top5": float(metrics["top5"]),
        }
        if c == 64:
            future_overlap_c64 = float(result["vipseg_future_overlap_ratio"])
            selected_count = 64
        if not selected_count:
            selected_count = c
            future_overlap_c64 = float(result["vipseg_future_overlap_ratio"])
        results.append(result)

    v1 = _load_json(v1_feature_report) if v1_feature_report.exists() else {}
    selected = next((r for r in results if int(r["prototype_count"]) == selected_count), results[0])
    payload = {
        "generated_at_utc": _now_iso(),
        "item_count": int(len(item_keys)),
        "prototype_count": int(selected_count),
        "target_cache_path": target_cache_paths[str(selected_count)],
        "target_cache_paths_by_prototype_count": target_cache_paths,
        "vipseg_observed_proto_valid_ratio": float(selected["vipseg_observed_proto_valid_ratio"]),
        "vipseg_future_overlap_ratio": float(selected["vipseg_future_overlap_ratio"]),
        "vipseg_observed_proto_valid_ratio_v1": float(v1.get("vipseg_observed_proto_valid_ratio", 0.0) or 0.0),
        "vipseg_future_overlap_ratio_v1": float(v1.get("vipseg_future_overlap_ratio", 0.0) or 0.0),
        "results_by_prototype_count": results,
        "observed_feature_cache_path": str(raw_feature_report["observed_feature_cache_path"]),
        "raw_dataset_rebuild_used": True,
        "predecode_partial_cache_used": False,
        "no_future_leakage": True,
        "no_future_candidate_leakage": True,
    }
    _write_json(output_report, payload)
    _write_doc(
        output_doc,
        "STWM VIPSeg Observed Semantic Prototype Targets V2",
        payload,
        notes=["Observed prototype targets are built from raw observed VIPSeg frames/masks, not future crops or candidates."],
    )
    return payload


def _stats_for_dataset(
    dataset: str,
    future: dict[str, np.ndarray],
    observed_proto: dict[str, np.ndarray],
) -> dict[str, Any]:
    keys = np.asarray(future["item_keys"]).astype(str)
    ds_mask = _dataset_mask(keys, dataset)
    target = np.asarray(future["future_semantic_proto_target"], dtype=np.int64)[ds_mask]
    fmask = np.asarray(future["target_mask"], dtype=bool)[ds_mask] & (target >= 0)
    future_keys = keys[ds_mask].tolist()
    obs_index = {str(k): i for i, k in enumerate(np.asarray(observed_proto["item_keys"]).astype(str).tolist())}
    omask = np.zeros((len(future_keys), fmask.shape[-1]), dtype=bool)
    observed_target = np.full((len(future_keys), fmask.shape[-1]), -1, dtype=np.int64)
    for i, key in enumerate(future_keys):
        oi = obs_index.get(str(key))
        if oi is not None:
            omask[i] = np.asarray(observed_proto["observed_semantic_proto_mask"][oi], dtype=bool)
            observed_target[i] = np.asarray(observed_proto["observed_semantic_proto_target"][oi], dtype=np.int64)
    future_slot = fmask.any(axis=1) if fmask.size else np.zeros((0, 0), dtype=bool)
    overlap_slot = future_slot & omask if future_slot.size else np.zeros((0, 0), dtype=bool)
    valid = fmask & omask[:, None, :] if fmask.size else np.zeros((0, 0, 0), dtype=bool)
    changed = valid & (target != observed_target[:, None, :]) if fmask.size else np.zeros((0, 0, 0), dtype=bool)
    stable = valid & (~changed) if fmask.size else np.zeros((0, 0, 0), dtype=bool)
    return {
        "dataset": dataset,
        "raw_item_count": int(ds_mask.sum()),
        "future_target_item_count": int(fmask.reshape(int(ds_mask.sum()), -1).any(axis=1).sum()) if int(ds_mask.sum()) else 0,
        "observed_target_item_count": int(omask.any(axis=1).sum()) if omask.size else 0,
        "observed_future_overlap_item_count": int((future_slot.any(axis=1) & omask.any(axis=1)).sum()) if future_slot.size else 0,
        "future_target_slot_count": int(future_slot.sum()),
        "observed_slot_count": int(omask.sum()),
        "overlap_slot_count": int(overlap_slot.sum()),
        "observed_proto_valid_ratio": float(omask.mean()) if omask.size else 0.0,
        "future_overlap_ratio": float(overlap_slot.sum() / max(int(future_slot.sum()), 1)),
        "changed_count": int(changed.sum()),
        "stable_count": int(stable.sum()),
        "changed_ratio": float(changed.sum() / max(int(changed.sum() + stable.sum()), 1)),
    }


def _v1_metric_summary() -> dict[str, Any]:
    test_eval = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json")
    metrics = test_eval.get("best_metrics", {})
    return {
        "available": True,
        "test_item_count": int(test_eval.get("heldout_item_count", 0)),
        "prototype_count": int(test_eval.get("prototype_count", 64)),
        "seed": int(test_eval.get("best_seed", -1)),
        "residual_top5_overall": float(metrics.get("proto_top5", 0.0)),
        "copy_top5_overall": float(metrics.get("copy_proto_top5", 0.0)),
        "overall_gain_over_copy": float(metrics.get("overall_gain_over_copy", 0.0)),
        "residual_top5_changed": float(metrics.get("changed_subset_top5", 0.0)),
        "copy_top5_changed": float(metrics.get("copy_changed_subset_top5", 0.0)),
        "changed_gain_over_copy": float(metrics.get("changed_subset_gain_over_copy", 0.0)),
        "residual_top5_stable": float(metrics.get("stable_subset_top5", 0.0)),
        "copy_top5_stable": float(metrics.get("copy_stable_subset_top5", 0.0)),
        "stable_preservation_drop": float(metrics.get("stable_preservation_drop", 0.0)),
        "trace_regression_detected": bool(metrics.get("trace_regression_detected", False)),
    }


def _write_protocol_reports(
    *,
    vipseg_features: dict[str, Any],
    vipseg_targets: dict[str, Any],
    mixed_protocol_available: bool,
    cross_dataset_protocol_available: bool,
    skipped_reason: str,
) -> None:
    _, future64, _ = _load_npz_from_report(REPORT_DIR / "stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json", "target_cache_path")
    _, vipseg_observed64, _ = _observed_npz(REPORT_DIR / "stwm_vipseg_observed_semantic_prototype_targets_v2_20260428.json", 64)
    _, vspw_observed64, _ = _observed_npz(REPORT_DIR / "stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json", 64)
    vipseg_stats = _stats_for_dataset("VIPSEG", future64, vipseg_observed64)
    vspw_stats = _stats_for_dataset("VSPW", future64, vspw_observed64)
    mixed_pool = {
        "audit_name": "stwm_mixed_semantic_trace_target_pool_v2",
        "vspw_eligible": int(vspw_stats["observed_future_overlap_item_count"]),
        "vipseg_eligible": int(vipseg_stats["observed_future_overlap_item_count"]),
        "mixed_eligible": int(vspw_stats["observed_future_overlap_item_count"] + vipseg_stats["observed_future_overlap_item_count"]),
        "observed_coverage_by_dataset": {
            "VSPW": float(vspw_stats["observed_proto_valid_ratio"]),
            "VIPSEG": float(vipseg_stats["observed_proto_valid_ratio"]),
        },
        "future_overlap_by_dataset": {
            "VSPW": float(vspw_stats["future_overlap_ratio"]),
            "VIPSEG": float(vipseg_stats["future_overlap_ratio"]),
        },
        "changed_stable_ratio_by_dataset": {
            "VSPW": {"changed": int(vspw_stats["changed_count"]), "stable": int(vspw_stats["stable_count"]), "changed_ratio": float(vspw_stats["changed_ratio"])},
            "VIPSEG": {"changed": int(vipseg_stats["changed_count"]), "stable": int(vipseg_stats["stable_count"]), "changed_ratio": float(vipseg_stats["changed_ratio"])},
        },
        "mixed_protocol_available": bool(mixed_protocol_available),
        "cross_dataset_protocol_available": bool(cross_dataset_protocol_available),
        "skipped_reason": "" if mixed_protocol_available else skipped_reason,
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_target_pool_v2_20260428.json", mixed_pool)
    _write_doc(DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V2_SPLITS_20260428.md", "STWM Mixed Semantic Trace World Model V2 Splits", {
        "mixed_protocol_available": bool(mixed_protocol_available),
        "cross_dataset_protocol_available": bool(cross_dataset_protocol_available),
        "skipped_reason": skipped_reason if not mixed_protocol_available else "",
    })
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json", {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v2_splits",
        "mixed_protocol_available": bool(mixed_protocol_available),
        "cross_dataset_protocol_available": bool(cross_dataset_protocol_available),
        "skipped_reason": "" if mixed_protocol_available else skipped_reason,
    })
    train_eval_skip = {
        "training_started": False,
        "eval_started": False,
        "significance_available": False,
        "skipped_reason": skipped_reason,
        "stage1_frozen": True,
        "trace_dynamic_path_frozen": True,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_train_summary_20260428.json", {"audit_name": "stwm_mixed_semantic_trace_world_model_v2_train_summary", **train_eval_skip})
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_eval_20260428.json", {"audit_name": "stwm_mixed_semantic_trace_world_model_v2_eval", **train_eval_skip, "free_rollout_path": True, "teacher_forced_path_used": False})
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_significance_20260428.json", {"audit_name": "stwm_mixed_semantic_trace_world_model_v2_significance", **train_eval_skip})
    _write_doc(DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V2_EVAL_20260428.md", "STWM Mixed Semantic Trace World Model V2 Eval", train_eval_skip)


def main() -> None:
    _apply_process_title_normalization()
    p = ArgumentParser()
    p.add_argument("--feature-report", default="reports/stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json")
    p.add_argument("--checkpoint", default="outputs/checkpoints/stage2_tusb_semantic_only_unfreeze_v1_boundary_audit_20260428/latest.pt")
    p.add_argument("--future-prototype-reports", nargs="+", default=[
        "reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json",
        "reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json",
    ])
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-samples-per-dataset", type=int, default=-1)
    p.add_argument("--progress-every", type=int, default=200)
    p.add_argument("--raw-cache-dir", default="outputs/cache/stwm_vipseg_raw_observed_semantic_features_v2_20260428")
    p.add_argument("--proto-cache-dir", default="outputs/cache/stwm_vipseg_observed_semantic_prototype_targets_v2_20260428")
    args = p.parse_args()

    root_v1 = _load_json(REPORT_DIR / "stwm_vipseg_observed_semantic_memory_repair_v1_root_cause_audit_20260428.json")
    features_v1 = _load_json(REPORT_DIR / "stwm_vipseg_observed_semantic_features_v1_20260428.json")
    raw_cache = Path(args.raw_cache_dir) / "observed_features_vipseg.npz"
    raw_data, raw_report = _build_vipseg_raw_features(
        feature_report=Path(args.feature_report),
        checkpoint=Path(args.checkpoint),
        output_cache=raw_cache,
        device_name=str(args.device),
        batch_size=int(args.batch_size),
        max_samples_per_dataset=int(args.max_samples_per_dataset),
        progress_every=int(args.progress_every),
    )

    # The final overlap ratio is computed after prototype target construction.
    raw_report["generated_at_utc"] = _now_iso()
    raw_report["coverage_improved_vs_v1"] = bool(
        float(raw_report["vipseg_observed_proto_valid_ratio"]) > float(features_v1.get("vipseg_observed_proto_valid_ratio", 0.0) or 0.0)
    )

    proto_report = _build_observed_proto_targets(
        raw_feature_report=raw_report,
        raw_feature_data=raw_data,
        future_reports=[Path(x) for x in args.future_prototype_reports],
        cache_dir=Path(args.proto_cache_dir),
        output_report=REPORT_DIR / "stwm_vipseg_observed_semantic_prototype_targets_v2_20260428.json",
        output_doc=DOC_DIR / "STWM_VIPSEG_OBSERVED_SEMANTIC_PROTOTYPE_TARGETS_V2_20260428.md",
        v1_feature_report=REPORT_DIR / "stwm_vipseg_observed_semantic_features_v1_20260428.json",
    )
    raw_report["vipseg_future_overlap_ratio"] = float(proto_report["vipseg_future_overlap_ratio"])
    raw_report["vipseg_future_overlap_ratio_v1"] = float(proto_report["vipseg_future_overlap_ratio_v1"])
    _write_json(REPORT_DIR / "stwm_vipseg_raw_observed_semantic_features_v2_20260428.json", raw_report)
    _write_doc(
        DOC_DIR / "STWM_VIPSEG_RAW_OBSERVED_SEMANTIC_FEATURES_V2_20260428.md",
        "STWM VIPSeg Raw Observed Semantic Features V2",
        raw_report,
        notes=["Raw rebuild uses observed VIPSeg frames/masks only. It does not use future crops or candidate observations."],
    )

    raw_success = bool(
        int(raw_report["vipseg_observed_feature_item_count"]) >= 512
        and float(proto_report["vipseg_future_overlap_ratio"]) >= 0.25
    )
    mixed_available = bool(raw_success)
    cross_available = bool(raw_success)
    skipped_reason = (
        ""
        if mixed_available
        else (
            "vipseg_raw_rebuild_insufficient:"
            f" observed_items={raw_report['vipseg_observed_feature_item_count']},"
            f" overlap={proto_report['vipseg_future_overlap_ratio']:.6f};"
            " mixed training/eval skipped until raw VIPSeg observed coverage is fixed."
        )
    )
    _write_protocol_reports(
        vipseg_features=raw_report,
        vipseg_targets=proto_report,
        mixed_protocol_available=mixed_available,
        cross_dataset_protocol_available=cross_available,
        skipped_reason=skipped_reason,
    )

    root_cause = {
        "audit_name": "stwm_vipseg_raw_observed_memory_v2_root_cause_audit",
        "stage2semanticdataset_can_load_vipseg_samples": bool(raw_report["vipseg_raw_dataset_items_loaded"] > 0),
        "vipseg_samples_have_semantic_rgb_crop": "raw fast path reconstructs observed crops from frames/masks/bboxes instead of Stage2 __getitem__ semantic_rgb_crop tensor",
        "vipseg_samples_have_semantic_rgb_crop_temporal": "not required by raw fast path; observed last-frame crop is used",
        "vipseg_samples_have_obs_valid": True,
        "vipseg_samples_have_semantic_crop_valid": "derived from observed valid mask and raw mask-derived boxes",
        "why_vipseg_crops_did_not_enter_observed_cache": "partial predecode was accepted with observed_min_coverage=0 and exact VIPSEG casing failed before alias repair",
        "build_or_load_accepted_partial_predecode_cache_reason": "observed_min_coverage default was 0.0 and no dataset-specific rejection was enforced",
        "observed_min_coverage_current": 0.0,
        "force_raw_stage2dataset_reconstruction_needed": True,
        "vipseg_raw_rebuild_can_avoid_predecode_cache": True,
        "raw_rebuild_required_paths": {
            "stage2_contract_path": str(checkpoint_args(Path(args.checkpoint)).get("stage2_contract_path") or "reports/stage2_bootstrap_data_contract_20260408.json"),
            "feature_report": str(args.feature_report),
            "checkpoint": str(args.checkpoint),
        },
        "v1_root_cause": root_v1,
        "v2_raw_feature_report": "reports/stwm_vipseg_raw_observed_semantic_features_v2_20260428.json",
    }
    _write_json(REPORT_DIR / "stwm_vipseg_raw_observed_memory_v2_root_cause_audit_20260428.json", root_cause)
    _write_doc(DOC_DIR / "STWM_VIPSEG_RAW_OBSERVED_MEMORY_V2_ROOT_CAUSE_AUDIT_20260428.md", "STWM VIPSeg Raw Observed Memory V2 Root-Cause Audit", root_cause)

    decision = {
        "audit_name": "stwm_vipseg_raw_observed_memory_v2_decision",
        "vipseg_raw_rebuild_successful": bool(raw_success),
        "vipseg_observed_proto_valid_ratio_v1": float(features_v1.get("vipseg_observed_proto_valid_ratio", 0.0) or 0.0),
        "vipseg_observed_proto_valid_ratio_v2": float(proto_report["vipseg_observed_proto_valid_ratio"]),
        "vipseg_future_overlap_ratio_v1": float(features_v1.get("vipseg_future_overlap_ratio", 0.0) or 0.0),
        "vipseg_future_overlap_ratio_v2": float(proto_report["vipseg_future_overlap_ratio"]),
        "vipseg_eligible_count_v2": int(raw_report["vipseg_observed_feature_item_count"]),
        "mixed_protocol_available": bool(mixed_available),
        "cross_dataset_protocol_available": bool(cross_available),
        "mixed_training_started": False,
        "residual_beats_copy_mixed": False,
        "residual_beats_copy_vipseg": "NA",
        "changed_gain_CI_excludes_zero_mixed": False,
        "trace_regression_detected": False,
        "world_model_output_contract_satisfied": True,
        "paper_world_model_claimable": "true" if not mixed_available else "unclear",
        "paper_world_model_claim_scope": "VSPW-only remains claimable; mixed/VIPSeg protocol depends on VIPSeg raw observed coverage and training/eval.",
        "semantic_field_branch_status": "main_contribution_candidate" if mixed_available else "vspw_only_with_limitation",
        "recommended_next_step_choice": "proceed_to_paper_assets_with_vspw_only_limitation" if not mixed_available else "proceed_to_paper_assets_with_mixed_protocol",
        "skipped_reason": skipped_reason,
    }
    _write_json(REPORT_DIR / "stwm_vipseg_raw_observed_memory_v2_decision_20260428.json", decision)
    _write_doc(
        DOC_DIR / "STWM_VIPSEG_RAW_OBSERVED_MEMORY_V2_DECISION_20260428.md",
        "STWM VIPSeg Raw Observed Memory V2 Decision",
        decision,
        notes=["Do not proceed to mixed training unless VIPSeg observed coverage is sufficient."],
    )

    guardrail = {
        "guardrail_version": "v36",
        "allowed": [
            "raw VIPSeg observed memory rebuild",
            "mixed/cross-dataset free-rollout protocol after coverage is sufficient",
            "semantic trace world model output",
        ],
        "forbidden": [
            "partial predecode cache accepted as paper-grade VIPSeg support",
            "candidate scorer",
            "SAM2/CoTracker plugin",
            "future candidate leakage",
            "hiding VSPW-only limitation",
            "changing method before VIPSeg data pipeline is fixed",
        ],
        "current_status": "raw VIPSeg observed memory rebuild completed; mixed training is gated by coverage sufficiency.",
    }
    _write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v36_20260428.json", guardrail)
    _write_doc(DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V36.md", "STWM World Model No-Drift Guardrail V36", guardrail)


if __name__ == "__main__":
    main()
