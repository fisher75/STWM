#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from stwm.tracewm_v2_stage2.datasets.stage2_semantic_dataset import (
    Stage2SemanticDataset,
    Stage2SemanticDatasetConfig,
    _temporal_indices,
)


_IMAGE_CACHE: OrderedDict[str, Image.Image] = OrderedDict()
_IMAGE_CACHE_MAX = 8


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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Future Semantic Trace Feature Targets V1",
        "",
        f"- feature_backbone: `{payload.get('feature_backbone')}`",
        f"- feature_source: `{payload.get('feature_source')}`",
        f"- feature_dim: `{payload.get('feature_dim')}`",
        f"- item_count: `{payload.get('item_count')}`",
        f"- target_shape: `{payload.get('target_shape')}`",
        f"- target_mask_shape: `{payload.get('target_mask_shape')}`",
        f"- valid_target_ratio: `{payload.get('valid_target_ratio')}`",
        f"- no_future_candidate_leakage: `{payload.get('no_future_candidate_leakage')}`",
        f"- cache_path: `{payload.get('cache_path')}`",
        "",
        "Future GT crops are used only as supervised targets. They are not inserted into rollout inputs.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_rgb_cached(path: str) -> Image.Image:
    cached = _IMAGE_CACHE.get(path)
    if cached is not None:
        _IMAGE_CACHE.move_to_end(path)
        return cached
    with Image.open(path) as im:
        rgb = im.convert("RGB").copy()
    _IMAGE_CACHE[path] = rgb
    _IMAGE_CACHE.move_to_end(path)
    while len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
        _IMAGE_CACHE.popitem(last=False)
    return rgb


def _crop(frame_path: str, box_xyxy: np.ndarray) -> Image.Image | None:
    try:
        im = _load_rgb_cached(frame_path)
        w, h = im.size
        x0, y0, x1, y1 = [float(x) for x in box_xyxy.tolist()]
        x0 = max(0, min(int(round(x0)), w - 1))
        y0 = max(0, min(int(round(y0)), h - 1))
        x1 = max(x0 + 1, min(int(round(x1)), w))
        y1 = max(y0 + 1, min(int(round(y1)), h))
        return im.crop((x0, y0, x1, y1)).copy()
    except Exception:
        return None


class OpenAIClipExtractor:
    def __init__(self, device: torch.device, download_root: Path) -> None:
        import clip  # type: ignore

        self.model, self.preprocess = clip.load("ViT-B/32", device=str(device), download_root=str(download_root))
        self.model.eval()
        self.device = device
        self.name = "openai_clip_vit_b_32_local"
        self.source = str(download_root / "ViT-B-32.pt")

    @torch.no_grad()
    def encode(self, images: list[Image.Image], batch_size: int) -> np.ndarray:
        out: list[torch.Tensor] = []
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + int(batch_size)]
            tensor = torch.stack([self.preprocess(im) for im in batch], dim=0).to(self.device)
            feat = F.normalize(self.model.encode_image(tensor).float(), dim=-1).detach().cpu()
            out.append(feat)
        if not out:
            return np.zeros((0, 512), dtype=np.float32)
        return torch.cat(out, dim=0).numpy().astype(np.float32)


def _select_backbone(backbone_report: Path, device_name: str) -> tuple[OpenAIClipExtractor | None, str, str, int, bool]:
    payload = load_json(backbone_report) if backbone_report.exists() else {}
    selected = str(payload.get("selected_backbone") or "")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    if selected == "local_openai_clip_vit_b_32":
        clip_root = Path.home() / ".cache" / "clip"
        if (clip_root / "ViT-B-32.pt").exists():
            extractor = OpenAIClipExtractor(device=device, download_root=clip_root)
            return extractor, extractor.name, "future_gt_bbox_crop_clip_vit_b32", 512, True
    return None, "stwm_crop_stats_fallback", "future_gt_bbox_crop_rgb_bbox_stats", 14, False


def _stats_feature(crop: Image.Image | None, box: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    x0, y0, x1, y1 = [float(x) for x in box.tolist()]
    bw = max((x1 - x0) / max(float(width), 1.0), 0.0)
    bh = max((y1 - y0) / max(float(height), 1.0), 0.0)
    area = bw * bh
    mean_rgb = np.zeros((3,), dtype=np.float32)
    std_rgb = np.zeros((3,), dtype=np.float32)
    if crop is not None:
        arr = np.asarray(crop.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        mean_rgb = flat.mean(axis=0)
        std_rgb = flat.std(axis=0)
    return np.asarray(
        [
            ((x0 + x1) * 0.5) / max(float(width), 1.0),
            ((y0 + y1) * 0.5) / max(float(height), 1.0),
            bw,
            bh,
            area,
            np.sqrt(max(area, 0.0)),
            *mean_rgb.tolist(),
            *std_rgb.tolist(),
            1.0,
            0.0,
        ],
        dtype=np.float32,
    )


def _build_split(
    *,
    split: str,
    cfg_base: dict[str, Any],
    max_samples_per_dataset: int,
    extractor: OpenAIClipExtractor | None,
    feature_dim: int,
    batch_size: int,
) -> tuple[list[str], list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cfg = Stage2SemanticDatasetConfig(
        split=split,
        max_samples_per_dataset=int(max_samples_per_dataset),
        **cfg_base,
    )
    dataset = Stage2SemanticDataset(cfg)
    keys: list[str] = []
    splits: list[str] = []
    datasets: list[str] = []
    all_crops: list[Image.Image] = []
    crop_refs: list[tuple[int, int, int]] = []
    samples: list[dict[str, Any]] = []
    max_h = int(cfg.fut_len)
    max_k = int(cfg.max_entities_per_sample)
    features = np.zeros((len(dataset), max_h, max_k, int(feature_dim)), dtype=np.float32)
    target_mask = np.zeros((len(dataset), max_h, max_k), dtype=bool)
    visibility = np.zeros((len(dataset), max_h, max_k), dtype=bool)
    reappearance = np.zeros((len(dataset), max_h, max_k), dtype=bool)
    identity = np.full((len(dataset), max_k), -1, dtype=np.int64)
    extent = np.zeros((len(dataset), max_h, max_k, 4), dtype=np.float32)
    encode_flush_threshold = max(int(batch_size) * 8, int(batch_size))

    def _flush_crops() -> None:
        if extractor is None or not all_crops:
            return
        encoded = extractor.encode(all_crops, batch_size=int(batch_size))
        for vec, (ref_idx, ref_fh, ref_slot) in zip(encoded, crop_refs):
            features[ref_idx, ref_fh, ref_slot, : int(vec.shape[0])] = vec
        all_crops.clear()
        crop_refs.clear()

    for idx in range(len(dataset)):
        sample = dataset[idx]
        meta = sample.get("meta", {})
        dataset_name = str(meta.get("dataset", "")).upper()
        clip_id = str(meta.get("clip_id", ""))
        key = f"{dataset_name}::{clip_id}"
        keys.append(key)
        splits.append(split)
        datasets.append(dataset_name)
        samples.append({"item_id": key, "split": split, "dataset": dataset_name, "clip_id": clip_id})
        point_ids = sample.get("point_ids")
        if isinstance(point_ids, torch.Tensor):
            k_ids = min(max_k, int(point_ids.shape[0]))
            identity[idx, :k_ids] = point_ids[:k_ids].cpu().numpy().astype(np.int64)
        fut_valid = sample["fut_valid"].cpu().numpy().astype(bool)
        fut_state = sample["fut_state"].cpu().numpy().astype(np.float32)
        h = min(max_h, int(fut_valid.shape[0]))
        k = min(max_k, int(fut_valid.shape[1]))
        visibility[idx, :h, :k] = fut_valid[:h, :k]
        extent[idx, :h, :k, 0:2] = fut_state[:h, :k, 0:2]
        extent[idx, :h, :k, 2:4] = fut_state[:h, :k, 6:8]
        obs_valid = sample["obs_valid"].cpu().numpy().astype(bool)
        obs_seen_any = obs_valid[: int(cfg.obs_len), :k].any(axis=0)
        obs_endpoint_visible = obs_valid[int(cfg.obs_len) - 1, :k] if int(cfg.obs_len) > 0 else np.zeros((k,), dtype=bool)
        obs_occluded = obs_seen_any & (~obs_valid[: int(cfg.obs_len), :k].all(axis=0))
        reappearance_gate = ((~obs_endpoint_visible) | obs_occluded) & obs_seen_any
        reappearance[idx, :h, :k] = fut_valid[:h, :k] & reappearance_gate[None, :]

        entry = dataset.entries[idx]
        frame_paths = [str(x) for x in entry.get("frame_paths", [])]
        temporal = _temporal_indices(len(frame_paths), int(cfg.obs_len) + int(cfg.fut_len))
        boxes_over_time = sample["entity_boxes_over_time"].cpu().numpy().astype(np.float32)
        for fh in range(h):
            frame_idx = temporal[int(cfg.obs_len) + fh]
            frame_path = frame_paths[frame_idx] if 0 <= frame_idx < len(frame_paths) else ""
            image_size = (1, 1)
            if frame_path:
                try:
                    im = _load_rgb_cached(frame_path)
                    image_size = (int(im.width), int(im.height))
                except Exception:
                    pass
            for slot in range(k):
                if not bool(fut_valid[fh, slot]):
                    continue
                box = boxes_over_time[int(cfg.obs_len) + fh, slot]
                crop = _crop(frame_path, box) if frame_path else None
                if extractor is not None and crop is not None:
                    all_crops.append(crop)
                    crop_refs.append((idx, fh, slot))
                    target_mask[idx, fh, slot] = True
                    if len(all_crops) >= encode_flush_threshold:
                        _flush_crops()
                elif extractor is None:
                    features[idx, fh, slot, :14] = _stats_feature(crop, box, image_size)
                    target_mask[idx, fh, slot] = True

    _flush_crops()

    return keys, splits, datasets, features, target_mask, visibility, reappearance, identity, extent, samples


def build_targets(
    *,
    output: Path,
    doc: Path,
    cache_dir: Path,
    stage2_contract_path: Path,
    backbone_report: Path,
    dataset_names: list[str],
    splits: list[str],
    max_samples_train: int,
    max_samples_val: int,
    obs_len: int,
    fut_len: int,
    max_tokens: int,
    max_entities_per_sample: int,
    semantic_crop_size: int,
    semantic_patch_radius: int,
    predecode_cache_path: str,
    teacher_semantic_cache_path: str,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    _apply_process_title_normalization()
    extractor, feature_backbone, feature_source, feature_dim, frozen_available = _select_backbone(backbone_report, device)
    cfg_base = {
        "dataset_names": dataset_names,
        "contract_path": str(stage2_contract_path),
        "obs_len": int(obs_len),
        "fut_len": int(fut_len),
        "max_tokens": int(max_tokens),
        "semantic_patch_radius": int(semantic_patch_radius),
        "semantic_crop_size": int(semantic_crop_size),
        "semantic_source_mainline": "crop_visual_encoder",
        "semantic_temporal_window": 1,
        "predecode_cache_path": str(predecode_cache_path),
        "teacher_semantic_cache_path": str(teacher_semantic_cache_path),
        "max_entities_per_sample": int(max_entities_per_sample),
    }
    all_keys: list[str] = []
    all_splits: list[str] = []
    all_datasets: list[str] = []
    all_features: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_visibility: list[np.ndarray] = []
    all_reappearance: list[np.ndarray] = []
    all_identity: list[np.ndarray] = []
    all_extent: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []
    for split in splits:
        limit = int(max_samples_train if split == "train" else max_samples_val)
        block = _build_split(
            split=str(split),
            cfg_base=cfg_base,
            max_samples_per_dataset=limit,
            extractor=extractor,
            feature_dim=int(feature_dim),
            batch_size=int(batch_size),
        )
        keys, split_names, dataset_names_out, features, target_mask, visibility, reappearance, identity, extent, sample_rows = block
        all_keys.extend(keys)
        all_splits.extend(split_names)
        all_datasets.extend(dataset_names_out)
        all_features.append(features)
        all_masks.append(target_mask)
        all_visibility.append(visibility)
        all_reappearance.append(reappearance)
        all_identity.append(identity)
        all_extent.append(extent)
        samples.extend(sample_rows)

    feature_arr = np.concatenate(all_features, axis=0) if all_features else np.zeros((0, fut_len, max_entities_per_sample, feature_dim), dtype=np.float32)
    mask_arr = np.concatenate(all_masks, axis=0) if all_masks else np.zeros((0, fut_len, max_entities_per_sample), dtype=bool)
    visibility_arr = np.concatenate(all_visibility, axis=0) if all_visibility else np.zeros_like(mask_arr)
    reappearance_arr = np.concatenate(all_reappearance, axis=0) if all_reappearance else np.zeros_like(mask_arr)
    identity_arr = np.concatenate(all_identity, axis=0) if all_identity else np.zeros((0, max_entities_per_sample), dtype=np.int64)
    extent_arr = np.concatenate(all_extent, axis=0) if all_extent else np.zeros((0, fut_len, max_entities_per_sample, 4), dtype=np.float32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "future_semantic_trace_feature_targets_v1.npz"
    np.savez_compressed(
        cache_path,
        item_keys=np.asarray(all_keys, dtype=object),
        splits=np.asarray(all_splits, dtype=object),
        datasets=np.asarray(all_datasets, dtype=object),
        future_semantic_feature_target=feature_arr,
        target_mask=mask_arr,
        future_visibility_target=visibility_arr,
        future_reappearance_target=reappearance_arr,
        identity_target=identity_arr,
        extent_box_target=extent_arr,
    )
    valid_target_ratio = float(mask_arr.mean()) if mask_arr.size else 0.0
    payload = {
        "generated_at_utc": now_iso(),
        "stage2_contract_path": str(stage2_contract_path),
        "splits": splits,
        "dataset_names": dataset_names,
        "item_count": int(feature_arr.shape[0]),
        "feature_dim": int(feature_dim),
        "feature_backbone": feature_backbone,
        "feature_source": feature_source,
        "frozen_feature_backbone_available": bool(frozen_available),
        "target_shape": list(feature_arr.shape),
        "target_mask_shape": list(mask_arr.shape),
        "valid_target_ratio": valid_target_ratio,
        "future_semantic_feature_targets_available": bool(feature_arr.shape[0] > 0 and valid_target_ratio > 0.0),
        "target_quality": "future_gt_bbox_crop_frozen_clip" if frozen_available else "future_gt_bbox_crop_stats_fallback",
        "future_visibility_target_available": bool(visibility_arr.size > 0),
        "future_reappearance_target_available": bool(reappearance_arr.size > 0),
        "identity_target_available": bool(identity_arr.size > 0),
        "extent_box_target_available": bool(extent_arr.size > 0),
        "no_future_candidate_leakage": True,
        "cache_path": str(cache_path),
        "sample_items": samples[:20],
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--stage2-contract-path", default="reports/stage2_bootstrap_data_contract_20260408.json")
    p.add_argument("--backbone-report", default="reports/stwm_available_measurement_backbones_v7_20260428.json")
    p.add_argument("--dataset-names", nargs="*", default=["vspw", "vipseg"])
    p.add_argument("--splits", nargs="*", default=["train", "val"])
    p.add_argument("--max-samples-train", type=int, default=24)
    p.add_argument("--max-samples-val", type=int, default=12)
    p.add_argument("--obs-len", type=int, default=8)
    p.add_argument("--fut-len", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--max-entities-per-sample", type=int, default=8)
    p.add_argument("--semantic-crop-size", type=int, default=64)
    p.add_argument("--semantic-patch-radius", type=int, default=12)
    p.add_argument("--predecode-cache-path", default="")
    p.add_argument("--teacher-semantic-cache-path", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--cache-dir", default="outputs/cache/stwm_future_semantic_trace_feature_targets_v1_20260428")
    p.add_argument("--output", default="reports/stwm_future_semantic_trace_feature_targets_v1_20260428.json")
    p.add_argument("--doc", default="docs/STWM_FUTURE_SEMANTIC_TRACE_FEATURE_TARGETS_V1_20260428.md")
    args = p.parse_args()
    build_targets(
        output=Path(args.output),
        doc=Path(args.doc),
        cache_dir=Path(args.cache_dir),
        stage2_contract_path=Path(args.stage2_contract_path),
        backbone_report=Path(args.backbone_report),
        dataset_names=[str(x) for x in args.dataset_names],
        splits=[str(x) for x in args.splits],
        max_samples_train=int(args.max_samples_train),
        max_samples_val=int(args.max_samples_val),
        obs_len=int(args.obs_len),
        fut_len=int(args.fut_len),
        max_tokens=int(args.max_tokens),
        max_entities_per_sample=int(args.max_entities_per_sample),
        semantic_crop_size=int(args.semantic_crop_size),
        semantic_patch_radius=int(args.semantic_patch_radius),
        predecode_cache_path=str(args.predecode_cache_path),
        teacher_semantic_cache_path=str(args.teacher_semantic_cache_path),
        device=str(args.device),
        batch_size=int(args.batch_size),
    )


if __name__ == "__main__":
    main()
