#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import os

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


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
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM External Candidate Frozen Measurement Features V7",
        "",
        f"- selected_backbone: `{payload.get('selected_backbone')}`",
        f"- frozen_measurement_feature_available: `{payload.get('frozen_measurement_feature_available')}`",
        f"- frozen_backbone: `{payload.get('frozen_backbone')}`",
        f"- feature_dim: `{payload.get('feature_dim')}`",
        f"- item_count: `{payload.get('item_count')}`",
        f"- candidate_record_count: `{payload.get('candidate_record_count')}`",
        f"- feature_record_count: `{payload.get('feature_record_count')}`",
        f"- failed_feature_count: `{payload.get('failed_feature_count')}`",
        f"- future_candidate_used_as_input: `{payload.get('future_candidate_used_as_input')}`",
        f"- candidate_feature_used_for_rollout: `{payload.get('candidate_feature_used_for_rollout')}`",
        f"- candidate_feature_used_for_scoring: `{payload.get('candidate_feature_used_for_scoring')}`",
        f"- no_internet_download_attempted: `{payload.get('no_internet_download_attempted')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def _bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except Exception:
        return None


def _load_rgb_image_cached(frame_path: str) -> Image.Image:
    cached = _IMAGE_CACHE.get(frame_path)
    if cached is not None:
        _IMAGE_CACHE.move_to_end(frame_path)
        return cached
    with Image.open(frame_path) as im:
        rgb = im.convert("RGB").copy()
    _IMAGE_CACHE[frame_path] = rgb
    _IMAGE_CACHE.move_to_end(frame_path)
    while len(_IMAGE_CACHE) > _IMAGE_CACHE_MAX:
        _IMAGE_CACHE.popitem(last=False)
    return rgb


def _frame_path_for_index(frame_paths: list[Any], frame_index: Any) -> str | None:
    if not frame_paths:
        return None
    try:
        idx = int(frame_index)
    except Exception:
        idx = len(frame_paths) - 1
    idx = max(0, min(idx, len(frame_paths) - 1))
    return str(frame_paths[idx])


def _crop_image(frame_path: str | None, bbox: list[float] | None) -> Image.Image | None:
    if not frame_path or bbox is None:
        return None
    try:
        im = _load_rgb_image_cached(frame_path)
        w, h = im.size
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(x1 + 1, min(int(round(x2)), w))
        y2 = max(y1 + 1, min(int(round(y2)), h))
        return im.crop((x1, y1, x2, y2)).copy()
    except Exception:
        return None


def _stats_feature(frame_path: str | None, bbox: list[float] | None, image_size: Any) -> list[float] | None:
    if bbox is None:
        return None
    width = height = 1.0
    if isinstance(image_size, list) and len(image_size) >= 2:
        width, height = float(image_size[0]), float(image_size[1])
    elif isinstance(image_size, dict):
        width = float(image_size.get("width", 1.0))
        height = float(image_size.get("height", 1.0))
    x1, y1, x2, y2 = bbox
    bw = max((x2 - x1) / max(width, 1.0), 0.0)
    bh = max((y2 - y1) / max(height, 1.0), 0.0)
    area = bw * bh
    aspect = bw / max(bh, 1e-6)
    mean_rgb = [0.0, 0.0, 0.0]
    std_rgb = [0.0, 0.0, 0.0]
    crop = _crop_image(frame_path, bbox)
    if crop is not None:
        arr = np.asarray(crop.resize((64, 64), Image.BILINEAR), dtype=np.float32) / 255.0
        flat = arr.reshape(-1, 3)
        mean_rgb = [float(x) for x in flat.mean(axis=0).tolist()]
        std_rgb = [float(x) for x in flat.std(axis=0).tolist()]
    return [
        float(((x1 + x2) * 0.5) / max(width, 1.0)),
        float(((y1 + y2) * 0.5) / max(height, 1.0)),
        float(bw),
        float(bh),
        float(area),
        float(min(aspect, 10.0) / 10.0),
        float(math.sqrt(max(area, 0.0))),
        *mean_rgb,
        *std_rgb,
        1.0,
    ]


class OpenAIClipExtractor:
    def __init__(self, device: torch.device, download_root: Path) -> None:
        import clip  # type: ignore

        self.model, self.preprocess = clip.load("ViT-B/32", device=str(device), download_root=str(download_root))
        self.model.eval()
        self.device = device
        self.name = "openai_clip_vit_b_32_local"
        self.source = str(download_root / "ViT-B-32.pt")

    @torch.no_grad()
    def encode(self, images: list[Image.Image], batch_size: int) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(images), int(batch_size)):
            batch = images[start : start + int(batch_size)]
            tensor = torch.stack([self.preprocess(im) for im in batch], dim=0).to(self.device)
            feat = self.model.encode_image(tensor).float()
            feat = F.normalize(feat, dim=-1).detach().cpu()
            out.extend([[round(float(x), 6) for x in row.tolist()] for row in feat])
        return out


def _cosine01(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    dim = min(len(a), len(b))
    ta = torch.tensor(a[:dim], dtype=torch.float32)
    tb = torch.tensor(b[:dim], dtype=torch.float32)
    denom = ta.norm() * tb.norm()
    if float(denom.item()) <= 1e-12:
        return None
    return float(((torch.dot(ta, tb) / denom).clamp(-1.0, 1.0) * 0.5 + 0.5).item())


def _load_items(candidate_manifest: Path, max_items: int) -> list[dict[str, Any]]:
    data = load_json(candidate_manifest)
    records = data.get("records") if isinstance(data, dict) else []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        if isinstance(rec, dict):
            grouped.setdefault(str(rec.get("item_id")), []).append(rec)
    items: list[dict[str, Any]] = []
    for item_id, recs in grouped.items():
        first = recs[0]
        items.append(
            {
                "item_id": item_id,
                "frame_paths": first.get("frame_paths"),
                "observed_frame_indices": first.get("observed_frame_indices"),
                "future_frame_index": first.get("future_frame_index"),
                "observed_target": first.get("observed_target"),
                "image_size": first.get("image_size"),
                "future_candidates": [r.get("candidate") for r in recs],
                "candidate_ids": [str((r.get("candidate") or {}).get("candidate_id")) for r in recs],
                "subset_tags": first.get("subset_tags") if isinstance(first.get("subset_tags"), dict) else {},
            }
        )
        if int(max_items) > 0 and len(items) >= int(max_items):
            break
    return items


def build_cache(
    *,
    candidate_manifest: Path,
    backbone_report: Path,
    output: Path,
    doc: Path,
    device_name: str,
    batch_size: int,
    max_items: int,
) -> dict[str, Any]:
    _apply_process_title_normalization()
    backbone = load_json(backbone_report)
    selected = str(backbone.get("selected_backbone") or "stwm_crop_visual_encoder")
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    extractor: OpenAIClipExtractor | None = None
    selected_name = selected
    checkpoint_source = None
    frozen_available = False
    if selected == "local_openai_clip_vit_b_32":
        clip_root = Path(os.environ.get("HOME", str(Path.home()))) / ".cache" / "clip"
        if (clip_root / "ViT-B-32.pt").exists():
            extractor = OpenAIClipExtractor(device=device, download_root=clip_root)
            selected_name = extractor.name
            checkpoint_source = extractor.source
            frozen_available = True

    items = _load_items(candidate_manifest, int(max_items))
    features_by_item: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    feature_dim: int | None = None
    record_count = 0
    global_encoded: dict[tuple[str, str], list[float]] = {}

    if extractor is not None:
        crop_images: list[Image.Image] = []
        crop_keys: list[tuple[str, str]] = []
        for item in items:
            item_id = str(item.get("item_id"))
            frame_paths = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
            observed_indices = item.get("observed_frame_indices") if isinstance(item.get("observed_frame_indices"), list) else [0]
            obs_frame = _frame_path_for_index(frame_paths, observed_indices[0] if observed_indices else 0)
            fut_frame = _frame_path_for_index(frame_paths, item.get("future_frame_index"))
            obs_bbox = _bbox((item.get("observed_target") or {}).get("bbox") if isinstance(item.get("observed_target"), dict) else None)
            obs_crop = _crop_image(obs_frame, obs_bbox)
            if obs_crop is not None:
                crop_images.append(obs_crop)
                crop_keys.append((item_id, "__observed_target__"))
            for cid, cand in zip(item.get("candidate_ids") or [], item.get("future_candidates") or []):
                if not isinstance(cand, dict):
                    failures.append({"item_id": item_id, "candidate_id": cid, "failure_reason": "candidate_not_dict"})
                    continue
                crop = _crop_image(fut_frame, _bbox(cand.get("bbox")))
                if crop is None:
                    failures.append({"item_id": item_id, "candidate_id": cid, "failure_reason": "candidate_crop_unavailable"})
                    continue
                crop_images.append(crop)
                crop_keys.append((item_id, str(cid)))
        if crop_images:
            vectors = extractor.encode(crop_images, batch_size=int(batch_size))
            global_encoded = {key: vec for key, vec in zip(crop_keys, vectors)}

    for item in items:
        frame_paths = item.get("frame_paths") if isinstance(item.get("frame_paths"), list) else []
        observed_indices = item.get("observed_frame_indices") if isinstance(item.get("observed_frame_indices"), list) else [0]
        obs_frame = _frame_path_for_index(frame_paths, observed_indices[0] if observed_indices else 0)
        fut_frame = _frame_path_for_index(frame_paths, item.get("future_frame_index"))
        obs_bbox = _bbox((item.get("observed_target") or {}).get("bbox") if isinstance(item.get("observed_target"), dict) else None)
        encoded: dict[str, list[float]] = {}
        item_id = str(item.get("item_id"))
        if extractor is not None:
            encoded["__observed_target__"] = global_encoded.get((item_id, "__observed_target__"), [])
            for cid in item.get("candidate_ids") or []:
                encoded[str(cid)] = global_encoded.get((item_id, str(cid)), [])
        else:
            # Explicit fallback is deterministic and not called a frozen VLM.
            obs_vec = _stats_feature(obs_frame, obs_bbox, item.get("image_size"))
            if obs_vec is not None:
                encoded["__observed_target__"] = [round(float(x), 6) for x in obs_vec]
            for cid, cand in zip(item.get("candidate_ids") or [], item.get("future_candidates") or []):
                if isinstance(cand, dict):
                    vec = _stats_feature(fut_frame, _bbox(cand.get("bbox")), item.get("image_size"))
                    if vec is not None:
                        encoded[str(cid)] = [round(float(x), 6) for x in vec]
        obs_feature = encoded.get("__observed_target__", [])
        candidates: dict[str, Any] = {}
        for cid in item.get("candidate_ids") or []:
            cand_feature = encoded.get(str(cid), [])
            if cand_feature:
                feature_dim = feature_dim or len(cand_feature)
                record_count += 1
            candidates[str(cid)] = {
                "candidate_frozen_feature": cand_feature,
                "target_candidate_frozen_similarity": _cosine01(obs_feature, cand_feature),
                "future_candidate_feature_available": bool(cand_feature),
                "observed_target_feature_available": bool(obs_feature),
                "feature_dim": len(cand_feature),
                "candidate_feature_used_for_scoring": True,
                "candidate_feature_used_for_rollout": False,
                "future_candidate_used_as_input": False,
            }
        features_by_item[str(item.get("item_id"))] = {
            "observed_target_frozen_feature": obs_feature,
            "candidates": candidates,
            "subset_tags": item.get("subset_tags"),
        }

    payload = {
        "generated_at_utc": now_iso(),
        "candidate_manifest": str(candidate_manifest),
        "backbone_report": str(backbone_report),
        "selected_backbone": selected_name,
        "checkpoint_or_cache_source": checkpoint_source,
        "frozen_measurement_feature_available": bool(frozen_available),
        "frozen_backbone": bool(frozen_available),
        "feature_dim": feature_dim,
        "item_count": len(items),
        "candidate_record_count": sum(len((item.get("candidate_ids") or [])) for item in items),
        "feature_record_count": record_count,
        "failed_feature_count": len(failures),
        "failures": failures[:50],
        "no_internet_download_attempted": True,
        "future_candidate_used_as_input": False,
        "candidate_feature_used_for_rollout": False,
        "candidate_feature_used_for_scoring": True,
        "features_by_item": features_by_item,
    }
    write_json(output, payload)
    write_doc(doc, payload)
    return payload


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--candidate-manifest", default="reports/stwm_external_hardcase_candidate_expanded_manifest_v2_20260428.json")
    parser.add_argument("--backbone-report", default="reports/stwm_available_measurement_backbones_v7_20260428.json")
    parser.add_argument("--output", default="reports/stwm_external_candidate_frozen_measurement_features_v7_20260428.json")
    parser.add_argument("--doc", default="docs/STWM_EXTERNAL_CANDIDATE_FROZEN_MEASUREMENT_FEATURES_V7_20260428.md")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-items", type=int, default=389)
    args = parser.parse_args()
    build_cache(
        candidate_manifest=Path(args.candidate_manifest),
        backbone_report=Path(args.backbone_report),
        output=Path(args.output),
        doc=Path(args.doc),
        device_name=str(args.device),
        batch_size=int(args.batch_size),
        max_items=int(args.max_items),
    )


if __name__ == "__main__":
    main()
