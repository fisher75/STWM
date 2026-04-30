from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import Stage2SemanticDataset, Stage2SemanticDatasetConfig
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, title: str, payload: dict[str, Any], bullets: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    for bullet in bullets or []:
        lines.append(f"- {bullet}")
    if bullets:
        lines.append("")
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


class LocalClipExtractor:
    def __init__(self, device_name: str = "cuda") -> None:
        import clip  # type: ignore

        root = Path.home() / ".cache" / "clip"
        ckpt = root / "ViT-B-32.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"local OpenAI CLIP ViT-B/32 weights not found: {ckpt}")
        device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=str(device), download_root=str(root))
        self.model.eval()
        self.device = device
        self.feature_dim = 512
        self.name = "openai_clip_vit_b_32_local"
        self.source = str(ckpt)

    @torch.no_grad()
    def encode(self, images: list[Any], batch_size: int = 128) -> np.ndarray:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + int(batch_size)]
            tensor = torch.stack([self.preprocess(im) for im in batch], dim=0).to(self.device)
            feat = F.normalize(self.model.encode_image(tensor).float(), dim=-1).detach().cpu()
            chunks.append(feat)
        if not chunks:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return torch.cat(chunks, dim=0).numpy().astype(np.float32)


def load_npz_from_report(report_path: Path, key: str = "target_cache_path") -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cache_path = Path(str(payload.get(key) or payload.get("cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return payload, dict(np.load(cache_path, allow_pickle=True)), cache_path


def checkpoint_args(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and isinstance(payload.get("args"), dict):
        return dict(payload["args"])
    return {}


def dataset_for_split(args: dict[str, Any], split: str, max_samples_per_dataset: int) -> Stage2SemanticDataset:
    return Stage2SemanticDataset(
        Stage2SemanticDatasetConfig(
            dataset_names=list(args.get("dataset_names") or ["vspw", "vipseg"]),
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
    )


def _dataset_name_aliases(dataset: str) -> list[str]:
    """Return filename aliases for dataset names used by older caches.

    STWM item keys use `VIPSEG`, while the Stage2 predecode cache was written as
    `VIPSeg`.  Treating those as distinct silently drops VIPSeg observed memory.
    """
    aliases = [str(dataset)]
    if str(dataset).upper() == "VIPSEG":
        aliases.extend(["VIPSeg", "vipseg"])
    if str(dataset).upper() == "VSPW":
        aliases.extend(["VSPW", "vspw"])
    out: list[str] = []
    for name in aliases:
        if name and name not in out:
            out.append(name)
    return out


def _tensor_crop_to_pil(crop: torch.Tensor) -> Image.Image | None:
    if not isinstance(crop, torch.Tensor) or crop.ndim != 3 or int(crop.shape[0]) != 3:
        return None
    arr = crop.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    if float(np.abs(arr).sum()) <= 1e-8:
        return None
    return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")


def build_or_load_observed_feature_cache(
    *,
    feature_report: Path,
    checkpoint_path: Path,
    output_cache: Path,
    device: str = "cuda",
    batch_size: int = 128,
    max_samples_per_dataset: int = 512,
    force_rebuild: bool = False,
    min_required_coverage: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    feature_payload, feature_data, _ = load_npz_from_report(feature_report, key="cache_path")
    item_keys = [str(x) for x in feature_data["item_keys"].tolist()]
    splits = [str(x) for x in feature_data["splits"].tolist()]
    datasets = [str(x) for x in feature_data["datasets"].tolist()] if "datasets" in feature_data else []
    requested_meta = {
        "feature_report_path": str(feature_report),
        "feature_report_mtime": float(feature_report.stat().st_mtime) if feature_report.exists() else None,
        "checkpoint_path": str(checkpoint_path),
        "dataset_names": sorted(set(datasets)),
        "splits": sorted(set(splits)),
        "max_samples_per_dataset": int(max_samples_per_dataset),
        "item_keys_count": int(len(item_keys)),
    }

    rebuild_reason = ""
    if output_cache.exists() and not bool(force_rebuild):
        with np.load(output_cache, allow_pickle=True) as data:
            cached = {k: data[k] for k in data.files}
        cached_metadata: dict[str, Any] = {}
        if "metadata_json" in cached:
            try:
                cached_metadata = json.loads(str(cached["metadata_json"].tolist()))
            except Exception:
                cached_metadata = {}
        observed_mask = np.asarray(cached.get("observed_feature_mask", np.zeros((0, 0), dtype=bool)), dtype=bool)
        observed_ratio = float(observed_mask.mean()) if observed_mask.size else 0.0
        mismatch_reasons: list[str] = []
        if not cached_metadata:
            mismatch_reasons.append("missing_cache_fingerprint")
        for key in ("feature_report_path", "checkpoint_path", "max_samples_per_dataset", "item_keys_count"):
            if cached_metadata and cached_metadata.get(key) != requested_meta.get(key):
                mismatch_reasons.append(f"fingerprint_mismatch:{key}")
        if cached_metadata and sorted(cached_metadata.get("splits", [])) != requested_meta["splits"]:
            mismatch_reasons.append("fingerprint_mismatch:splits")
        if cached_metadata and sorted(cached_metadata.get("dataset_names", [])) != requested_meta["dataset_names"]:
            mismatch_reasons.append("fingerprint_mismatch:dataset_names")
        if observed_ratio < float(min_required_coverage):
            mismatch_reasons.append("coverage_below_min_required")
        if not mismatch_reasons:
            cached_metadata.update(
                {
                    "observed_feature_cache_path": str(output_cache),
                    "observed_feature_cache_reused": True,
                    "feature_backbone": str(cached.get("feature_backbone", np.asarray("")).tolist()),
                    "feature_dim": int(cached["observed_last_feature"].shape[-1]),
                    "observed_slot_feature_available_ratio": observed_ratio,
                    "observed_nonzero_count": int(observed_mask.sum()),
                    "cache_rebuild_reason": "",
                    "requested_fingerprint": requested_meta,
                }
            )
            return cached, cached_metadata
        rebuild_reason = ",".join(mismatch_reasons)
    elif bool(force_rebuild):
        rebuild_reason = "force_rebuild_requested"

    if output_cache.exists() and rebuild_reason:
        # Overwrite in place after recording why the existing cache was rejected.
        pass

    key_to_idx = {key: idx for idx, key in enumerate(item_keys)}
    split_by_key = {key: splits[idx] for idx, key in enumerate(item_keys)}
    n = len(item_keys)
    _, _, k_max, feature_dim = feature_data["future_semantic_feature_target"].shape
    obs_last = np.zeros((n, k_max, feature_dim), dtype=np.float32)
    obs_mean_sum = np.zeros((n, k_max, feature_dim), dtype=np.float32)
    obs_count = np.zeros((n, k_max), dtype=np.float32)
    obs_last_mask = np.zeros((n, k_max), dtype=bool)
    trace_summary = np.zeros((n, k_max, 10), dtype=np.float32)

    args = checkpoint_args(checkpoint_path)
    predecode_cache_path = Path(str(args.get("predecode_cache_path") or ""))
    if predecode_cache_path.exists():
        extractor = LocalClipExtractor(device_name=device)
        crops: list[Any] = []
        refs: list[tuple[int, int, str]] = []
        predecode_hits = 0
        predecode_hits_by_dataset: dict[str, int] = {}
        predecode_file_hits_by_dataset: dict[str, int] = {}
        predecode_missing_by_dataset: dict[str, int] = {}
        for idx, key in enumerate(item_keys):
            dataset, clip = key.split("::", 1) if "::" in key else ("", key)
            split = str(split_by_key.get(key) or splits[idx])
            candidates = []
            for dataset_alias in _dataset_name_aliases(dataset):
                candidates.extend(
                    [
                        predecode_cache_path / split / f"{dataset_alias}__{split}__{clip}.npz",
                        predecode_cache_path / "train" / f"{dataset_alias}__train__{clip}.npz",
                        predecode_cache_path / "val" / f"{dataset_alias}__val__{clip}.npz",
                    ]
                )
            cache_file = next((p for p in candidates if p.exists()), None)
            if cache_file is None:
                matches = []
                for dataset_alias in _dataset_name_aliases(dataset):
                    matches.extend(predecode_cache_path.glob(f"*/*{dataset_alias}__*__{clip}.npz"))
                cache_file = matches[0] if matches else None
            if cache_file is None:
                predecode_missing_by_dataset[dataset] = predecode_missing_by_dataset.get(dataset, 0) + 1
                continue
            predecode_file_hits_by_dataset[dataset] = predecode_file_hits_by_dataset.get(dataset, 0) + 1
            try:
                with np.load(cache_file, allow_pickle=True) as predecode:
                    rgb = np.asarray(predecode["semantic_rgb_crop"], dtype=np.float32)
                    valid = np.asarray(predecode.get("semantic_crop_valid", np.ones((rgb.shape[0],), dtype=bool)), dtype=bool)
                    obs_valid = np.asarray(predecode.get("obs_valid", np.zeros((1, rgb.shape[0]), dtype=bool)), dtype=bool)
            except Exception:
                continue
            if rgb.ndim != 4 or rgb.shape[1] != 3:
                continue
            kk = min(k_max, int(rgb.shape[0]))
            any_hit = False
            for slot in range(kk):
                slot_valid = bool(valid[slot]) if slot < valid.shape[0] else True
                if obs_valid.ndim == 2 and slot < obs_valid.shape[1]:
                    slot_valid = slot_valid and bool(obs_valid[:, slot].any())
                if not slot_valid:
                    continue
                crop_tensor = torch.from_numpy(rgb[slot])
                crop = _tensor_crop_to_pil(crop_tensor)
                if crop is None:
                    continue
                crops.append(crop)
                refs.append((idx, slot, "last"))
                any_hit = True
            predecode_hits += int(any_hit)
            predecode_hits_by_dataset[dataset] = predecode_hits_by_dataset.get(dataset, 0) + int(any_hit)
        encoded = extractor.encode(crops, batch_size=int(batch_size)) if crops else np.zeros((0, feature_dim), dtype=np.float32)
        for vec, (idx, slot, _mode) in zip(encoded, refs):
            obs_last[idx, slot] = vec
            obs_mean_sum[idx, slot] += vec
            obs_count[idx, slot] += 1.0
            obs_last_mask[idx, slot] = True
        if float(obs_last_mask.mean()) >= float(min_required_coverage):
            obs_mean = obs_mean_sum / np.maximum(obs_count[..., None], 1.0)
            obs_mean = l2_normalize(obs_mean)
            obs_last = l2_normalize(obs_last)
            metadata = {
                **requested_meta,
                "feature_report_mtime": float(feature_report.stat().st_mtime) if feature_report.exists() else None,
                "observed_nonzero_count": int(obs_last_mask.sum()),
                "observed_slot_feature_available_ratio": float(obs_last_mask.mean()) if obs_last_mask.size else 0.0,
                "cache_rebuild_reason": rebuild_reason,
                "observed_feature_fast_path": "predecode_crop_clip",
                "predecode_cache_path": str(predecode_cache_path),
                "direct_cache_item_hits": int(predecode_hits),
                "direct_cache_item_hits_by_dataset": predecode_hits_by_dataset,
                "predecode_file_hits_by_dataset": predecode_file_hits_by_dataset,
                "predecode_missing_by_dataset": predecode_missing_by_dataset,
                "dataset_name_aliases_enabled": True,
            }
            output_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_cache,
                item_keys=np.asarray(item_keys, dtype=object),
                splits=np.asarray(splits, dtype=object),
                observed_last_feature=obs_last.astype(np.float32),
                observed_mean_feature=obs_mean.astype(np.float32),
                observed_feature_mask=obs_last_mask,
                observed_feature_count=obs_count.astype(np.float32),
                trace_summary=trace_summary.astype(np.float32),
                feature_backbone=np.asarray(extractor.name, dtype=object),
                feature_source=np.asarray("stage2_predecode_observed_semantic_crop_clip_vit_b32", dtype=object),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
            )
            data = dict(np.load(output_cache, allow_pickle=True))
            meta = {
                "observed_feature_cache_path": str(output_cache),
                "observed_feature_cache_reused": False,
                "feature_backbone": extractor.name,
                "feature_dim": int(feature_dim),
                "observed_slot_feature_available_ratio": float(obs_last_mask.mean()),
                "observed_nonzero_count": int(obs_last_mask.sum()),
                "item_keys_count": int(len(item_keys)),
                "max_samples_per_dataset": int(max_samples_per_dataset),
                "cache_rebuild_reason": rebuild_reason,
                "metadata": metadata,
                "no_network_download_attempted": True,
                "observed_feature_source_note": "uses local Stage2 predecode observed semantic crops encoded by local CLIP; no future candidate or future target crop is used",
            }
            return data, meta

    teacher_cache_path = Path(str(args.get("teacher_semantic_cache_path") or ""))
    direct_cache_hits = 0
    if teacher_cache_path.exists():
        # Fast path: reuse the observed semantic prior already consumed by Stage2.
        # This is observed-state memory, not a future candidate or future target input.
        for idx, key in enumerate(item_keys):
            dataset, clip = key.split("::", 1) if "::" in key else ("", key)
            split = str(split_by_key.get(key) or splits[idx])
            candidates = []
            for dataset_alias in _dataset_name_aliases(dataset):
                candidates.extend(
                    [
                        teacher_cache_path / f"{dataset_alias}__{split}__{clip}.npz",
                        teacher_cache_path / f"{dataset_alias}__train__{clip}.npz",
                        teacher_cache_path / f"{dataset_alias}__val__{clip}.npz",
                    ]
                )
            cache_file = next((p for p in candidates if p.exists()), None)
            if cache_file is None:
                matches = []
                for dataset_alias in _dataset_name_aliases(dataset):
                    matches.extend(teacher_cache_path.glob(f"{dataset_alias}__*__{clip}.npz"))
                cache_file = matches[0] if matches else None
            if cache_file is None:
                continue
            try:
                with np.load(cache_file, allow_pickle=True) as teacher_payload:
                    prior = np.asarray(teacher_payload["semantic_teacher_prior"], dtype=np.float32)
            except Exception:
                continue
            if prior.ndim != 2 or prior.shape[-1] != feature_dim:
                continue
            kk = min(k_max, int(prior.shape[0]))
            if kk <= 0:
                continue
            prior = l2_normalize(prior[:kk])
            valid = np.linalg.norm(prior, axis=-1) > 1e-8
            obs_last[idx, :kk] = prior
            obs_mean_sum[idx, :kk] += prior
            obs_count[idx, :kk] += valid.astype(np.float32)
            obs_last_mask[idx, :kk] = valid
            direct_cache_hits += int(valid.any())

    if float(obs_last_mask.mean()) >= float(min_required_coverage):
        obs_mean = obs_mean_sum / np.maximum(obs_count[..., None], 1.0)
        obs_mean = l2_normalize(obs_mean)
        obs_last = l2_normalize(obs_last)
        metadata = {
            **requested_meta,
            "feature_report_mtime": float(feature_report.stat().st_mtime) if feature_report.exists() else None,
            "observed_nonzero_count": int(obs_last_mask.sum()),
            "observed_slot_feature_available_ratio": float(obs_last_mask.mean()) if obs_last_mask.size else 0.0,
            "cache_rebuild_reason": rebuild_reason,
            "observed_feature_fast_path": "teacher_semantic_cache",
            "teacher_semantic_cache_path": str(teacher_cache_path),
            "direct_cache_item_hits": int(direct_cache_hits),
        }
        output_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_cache,
            item_keys=np.asarray(item_keys, dtype=object),
            splits=np.asarray(splits, dtype=object),
            observed_last_feature=obs_last.astype(np.float32),
            observed_mean_feature=obs_mean.astype(np.float32),
            observed_feature_mask=obs_last_mask,
            observed_feature_count=obs_count.astype(np.float32),
            trace_summary=trace_summary.astype(np.float32),
            feature_backbone=np.asarray("stage2_teacher_semantic_cache_prior", dtype=object),
            feature_source=np.asarray("stage2_observed_semantic_teacher_prior_cache", dtype=object),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
        )
        data = dict(np.load(output_cache, allow_pickle=True))
        meta = {
            "observed_feature_cache_path": str(output_cache),
            "observed_feature_cache_reused": False,
            "feature_backbone": "stage2_teacher_semantic_cache_prior",
            "feature_dim": int(feature_dim),
            "observed_slot_feature_available_ratio": float(obs_last_mask.mean()),
            "observed_nonzero_count": int(obs_last_mask.sum()),
            "item_keys_count": int(len(item_keys)),
            "max_samples_per_dataset": int(max_samples_per_dataset),
            "cache_rebuild_reason": rebuild_reason,
            "metadata": metadata,
            "no_network_download_attempted": True,
            "observed_feature_source_note": "uses local Stage2 observed semantic teacher-prior cache; no future candidate or future target crop is used",
        }
        return data, meta

    extractor = LocalClipExtractor(device_name=device)
    crops: list[Any] = []
    refs: list[tuple[int, int, str]] = []
    for split in sorted(set(splits)):
        ds = dataset_for_split(args, split=split, max_samples_per_dataset=int(max_samples_per_dataset))
        for sample_idx in range(len(ds)):
            sample = ds[sample_idx]
            key = stage2_item_key(sample.get("meta", {}))
            if key not in key_to_idx or split_by_key.get(key) != split:
                continue
            idx = key_to_idx[key]
            obs_valid = sample["obs_valid"].cpu().numpy().astype(bool)
            obs_state = sample["obs_state"].cpu().numpy().astype(np.float32)
            k = min(k_max, int(obs_valid.shape[1]))
            for slot in range(k):
                valid_steps = np.where(obs_valid[:, slot])[0].tolist()
                if not valid_steps:
                    continue
                first = int(valid_steps[0])
                last = int(valid_steps[-1])
                last_state = obs_state[last, slot]
                first_state = obs_state[first, slot]
                trace_summary[idx, slot] = np.asarray(
                    [
                        float(first_state[0]),
                        float(first_state[1]),
                        float(last_state[0]),
                        float(last_state[1]),
                        float(last_state[0] - first_state[0]),
                        float(last_state[1] - first_state[1]),
                        float(len(valid_steps) / max(int(args.get("obs_len") or 8), 1)),
                        float(obs_valid[-1, slot]),
                        float(last_state[6]) if last_state.shape[0] > 6 else 0.0,
                        float(last_state[7]) if last_state.shape[0] > 7 else 0.0,
                    ],
                    dtype=np.float32,
                )
                crop = _tensor_crop_to_pil(sample["semantic_rgb_crop"][slot])
                if crop is not None:
                    crops.append(crop)
                    refs.append((idx, slot, "last"))
                    obs_last_mask[idx, slot] = True
                temporal_crop = sample.get("semantic_rgb_crop_temporal")
                temporal_valid = sample.get("semantic_temporal_valid")
                if isinstance(temporal_crop, torch.Tensor) and temporal_crop.ndim == 5:
                    tw = int(temporal_crop.shape[1])
                    any_temporal = False
                    for step in range(tw):
                        if isinstance(temporal_valid, torch.Tensor) and temporal_valid.ndim == 2 and not bool(temporal_valid[slot, step].item()):
                            continue
                        crop_t = _tensor_crop_to_pil(temporal_crop[slot, step])
                        if crop_t is None:
                            continue
                        crops.append(crop_t)
                        refs.append((idx, slot, "mean"))
                        any_temporal = True
                    if not any_temporal and crop is not None:
                        crops.append(crop)
                        refs.append((idx, slot, "mean"))
                elif crop is not None:
                    crops.append(crop)
                    refs.append((idx, slot, "mean"))
    encoded = extractor.encode(crops, batch_size=int(batch_size)) if crops else np.zeros((0, feature_dim), dtype=np.float32)
    for vec, (idx, slot, mode) in zip(encoded, refs):
        if mode == "last":
            obs_last[idx, slot] = vec
        obs_mean_sum[idx, slot] += vec
        obs_count[idx, slot] += 1.0
    obs_mean = obs_mean_sum / np.maximum(obs_count[..., None], 1.0)
    obs_mean = l2_normalize(obs_mean)
    obs_last = l2_normalize(obs_last)
    metadata = {
        **requested_meta,
        "feature_report_mtime": float(feature_report.stat().st_mtime) if feature_report.exists() else None,
        "observed_nonzero_count": int(obs_last_mask.sum()),
        "observed_slot_feature_available_ratio": float(obs_last_mask.mean()) if obs_last_mask.size else 0.0,
        "cache_rebuild_reason": rebuild_reason,
    }
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_cache,
        item_keys=np.asarray(item_keys, dtype=object),
        splits=np.asarray(splits, dtype=object),
        observed_last_feature=obs_last.astype(np.float32),
        observed_mean_feature=obs_mean.astype(np.float32),
        observed_feature_mask=obs_last_mask,
        observed_feature_count=obs_count.astype(np.float32),
        trace_summary=trace_summary.astype(np.float32),
        feature_backbone=np.asarray(extractor.name, dtype=object),
        feature_source=np.asarray("stage2_observed_semantic_crop_clip_vit_b32", dtype=object),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True), dtype=object),
    )
    data = dict(np.load(output_cache, allow_pickle=True))
    meta = {
        "observed_feature_cache_path": str(output_cache),
        "observed_feature_cache_reused": False,
        "feature_backbone": extractor.name,
        "feature_dim": int(feature_dim),
        "observed_slot_feature_available_ratio": float(obs_last_mask.mean()),
        "observed_nonzero_count": int(obs_last_mask.sum()),
        "item_keys_count": int(len(item_keys)),
        "max_samples_per_dataset": int(max_samples_per_dataset),
        "cache_rebuild_reason": rebuild_reason,
        "metadata": metadata,
        "no_network_download_attempted": True,
        "observed_feature_source_note": "uses Stage2 prebuilt observed semantic crop/temporal crop, not future candidate or future target crop",
    }
    return data, meta


def topk_metrics(scores: np.ndarray, labels: np.ndarray, topk: int = 5) -> dict[str, float]:
    if scores.size == 0 or labels.size == 0:
        return {"top1": 0.0, "top5": 0.0, "ce": 0.0}
    pred = scores.argmax(axis=-1)
    k = min(int(topk), int(scores.shape[-1]))
    top = np.argpartition(-scores, kth=k - 1, axis=-1)[:, :k]
    top5 = np.any(top == labels[:, None], axis=1)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs = probs / np.maximum(probs.sum(axis=-1, keepdims=True), 1e-8)
    ce = -np.log(np.maximum(probs[np.arange(labels.shape[0]), labels], 1e-8)).mean()
    return {"top1": float((pred == labels).mean()), "top5": float(top5.mean()), "ce": float(ce)}


def frequency_scores(train_labels: np.ndarray, eval_count: int, prototype_count: int) -> np.ndarray:
    counts = np.bincount(train_labels.astype(np.int64), minlength=int(prototype_count)).astype(np.float32)
    scores = np.log(np.maximum(counts / max(counts.sum(), 1.0), 1e-8))
    return np.repeat(scores[None, :], int(eval_count), axis=0)
