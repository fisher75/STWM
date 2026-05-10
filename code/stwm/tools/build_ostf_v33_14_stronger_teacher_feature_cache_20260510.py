#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


PREPARE = ROOT / "reports/stwm_ostf_v33_14_real_teacher_model_prepare_20260510.json"
COMPLETE = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128"
OLD_VIS = COMPLETE / "visual_teacher_prototypes/pointodyssey/clip_vit_b32_local"
OUT = ROOT / "outputs/cache/stwm_ostf_v33_14_teacher_features/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v33_14_teacher_feature_cache_build_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_14_TEACHER_FEATURE_CACHE_BUILD_20260510.md"


HF_IDS = {
    "dinov2_base": ("facebook/dinov2-base", "auto"),
    "dinov2_large": ("facebook/dinov2-large", "auto"),
    "siglip_base": ("google/siglip-base-patch16-224", "siglip"),
    "clip_vit_l14": ("openai/clip-vit-large-patch14", "clip"),
}


def load_teacher(name: str, device: torch.device) -> tuple[Any, Any, int]:
    from transformers import AutoImageProcessor, AutoModel, CLIPVisionModel, SiglipVisionModel

    hf_id, kind = HF_IDS[name]
    cache_dir = ROOT / "models/semantic_teachers" / name
    processor = AutoImageProcessor.from_pretrained(hf_id, cache_dir=str(cache_dir))
    if kind == "clip":
        model = CLIPVisionModel.from_pretrained(hf_id, cache_dir=str(cache_dir))
    elif kind == "siglip":
        model = SiglipVisionModel.from_pretrained(hf_id, cache_dir=str(cache_dir))
    else:
        model = AutoModel.from_pretrained(hf_id, cache_dir=str(cache_dir))
    model.eval().to(device)
    with torch.no_grad():
        dummy = torch.rand(1, 3, 224, 224, device=device)
        y = model(pixel_values=dummy)
        tokens = y.last_hidden_state[:, 1:, :] if y.last_hidden_state.shape[1] > 1 else y.last_hidden_state
    return model, processor, int(tokens.shape[-1])


def frame_tokens(model: Any, processor: Any, paths: list[str], device: torch.device) -> tuple[np.ndarray, list[tuple[int, int]]]:
    images = []
    sizes = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        sizes.append(img.size)
        images.append(img)
    inputs = processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        y = model(pixel_values=pixel_values)
        tokens = y.last_hidden_state
        if tokens.shape[1] > 1:
            tokens = tokens[:, 1:, :]
        tokens = torch.nn.functional.normalize(tokens, dim=-1)
    return tokens.detach().cpu().numpy().astype(np.float16), sizes


def sample_tokens(tokens: np.ndarray, sizes: list[tuple[int, int]], points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    # tokens [T, P, D], points [M,T,2]
    t, p, d = tokens.shape
    grid = int(math.sqrt(p))
    if grid * grid != p:
        grid = int(round(math.sqrt(p)))
    out = np.zeros((*points.shape[:2], d), dtype=np.float16)
    for ti in range(t):
        w, h = sizes[ti]
        xy = points[:, ti]
        ix = np.clip((xy[:, 0] / max(w, 1) * grid).astype(np.int64), 0, grid - 1)
        iy = np.clip((xy[:, 1] / max(h, 1) * grid).astype(np.int64), 0, grid - 1)
        idx = np.clip(iy * grid + ix, 0, p - 1)
        out[:, ti] = tokens[ti, idx]
        out[~valid[:, ti], ti] = 0
    return out


def iter_old(split: str) -> list[Path]:
    return sorted((OLD_VIS / split).glob("*.npz"))


def build_for_teacher(name: str, args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, processor, dim = load_teacher(name, device)
    start = time.time()
    by_split: dict[str, Any] = {}
    total_obs = total_fut = total_slots = 0
    for split in ("train", "val", "test"):
        out_dir = OUT / name / "point_local_crop" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        files = iter_old(split)
        if args.max_samples_per_split > 0:
            files = files[: args.max_samples_per_split]
        obs_cov = fut_cov = slots = 0
        for old_path in files:
            old = np.load(old_path, allow_pickle=True)
            source = ROOT / str(old["source_npz"])
            base = np.load(source, allow_pickle=True)
            uid = str(old["sample_uid"].item() if hasattr(old["sample_uid"], "item") else old["sample_uid"])
            frame_paths = [str(x) for x in base["frame_paths"]]
            obs_points = np.asarray(base["obs_points"], dtype=np.float32)
            fut_points = np.asarray(base["fut_points"], dtype=np.float32)
            obs_vis = np.asarray(base["obs_vis"]).astype(bool)
            fut_vis = np.asarray(base["fut_vis"]).astype(bool)
            paths = frame_paths[: obs_points.shape[1] + fut_points.shape[1]]
            tokens_all, sizes = frame_tokens(model, processor, paths, device)
            obs_tokens = sample_tokens(tokens_all[: obs_points.shape[1]], sizes[: obs_points.shape[1]], obs_points, obs_vis)
            fut_tokens = sample_tokens(tokens_all[obs_points.shape[1] :], sizes[obs_points.shape[1] :], fut_points, fut_vis)
            point_to_instance_id = np.asarray(old["point_to_instance_id"], dtype=np.int64) if "point_to_instance_id" in old.files else np.full((obs_points.shape[0],), -1)
            obs_inst = np.zeros((obs_points.shape[0], dim), dtype=np.float16)
            for inst in np.unique(point_to_instance_id[point_to_instance_id >= 0]):
                mask = point_to_instance_id == inst
                vals = obs_tokens[mask][obs_vis[mask]]
                if vals.size:
                    obs_inst[mask] = vals.reshape(-1, dim).mean(axis=0).astype(np.float16)
            np.savez_compressed(
                out_dir / f"{uid}.npz",
                sample_uid=uid,
                split=split,
                teacher_name=name,
                aggregation="point_local_crop",
                implementation_detail="full-frame teacher patch token sampled at point coordinate",
                embedding_dim=dim,
                point_id=np.asarray(old["point_id"], dtype=np.int64),
                point_to_instance_id=point_to_instance_id,
                obs_teacher_embedding=obs_tokens,
                obs_teacher_available_mask=obs_vis,
                obs_instance_teacher_embedding=obs_inst,
                fut_teacher_embedding=fut_tokens,
                fut_teacher_available_mask=fut_vis,
                crop_confidence_obs=obs_vis.astype(np.float16),
                crop_confidence_fut=fut_vis.astype(np.float16),
                leakage_safe=True,
                observed_embeddings_input_allowed=True,
                future_embeddings_input_allowed=False,
                future_embeddings_supervision_only=True,
                source_npz=str(source.relative_to(ROOT)),
            )
            obs_cov += int(obs_vis.sum())
            fut_cov += int(fut_vis.sum())
            slots += int(obs_vis.size + fut_vis.size)
        by_split[split] = {"sample_count": len(files), "obs_embedding_coverage": float(obs_cov / max(slots / 2, 1)), "future_embedding_coverage": float(fut_cov / max(slots / 2, 1))}
        total_obs += obs_cov
        total_fut += fut_cov
        total_slots += slots
    del model
    torch.cuda.empty_cache()
    return {
        "teacher_name": name,
        "aggregation": "point_local_crop",
        "embedding_dim": dim,
        "by_split": by_split,
        "obs_embedding_coverage": float(total_obs / max(total_slots / 2, 1)),
        "future_embedding_coverage": float(total_fut / max(total_slots / 2, 1)),
        "crop_failure_ratio": float(1.0 - ((total_obs + total_fut) / max(total_slots, 1))),
        "mask_crop_coverage": 0.0,
        "point_crop_coverage": float((total_obs + total_fut) / max(total_slots, 1)),
        "cache_size_bytes": sum(p.stat().st_size for p in (OUT / name / "point_local_crop").rglob("*.npz")),
        "time_cost_seconds": float(time.time() - start),
        "leakage_safe": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="*", default=None)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    prepare = json.loads(PREPARE.read_text(encoding="utf-8")) if PREPARE.exists() else {}
    available = [t for t in prepare.get("available_teachers", []) if t in HF_IDS]
    teachers = args.teachers or available
    rows = []
    blockers = []
    if not teachers:
        blockers.append("no stronger teacher with forward dryrun passed")
    for teacher in teachers:
        if teacher not in available:
            blockers.append(f"{teacher} was not available in prepare report")
            continue
        rows.append(build_for_teacher(teacher, args))
    payload = {
        "generated_at_utc": utc_now(),
        "teacher_feature_cache_built": bool(rows),
        "output_root": str(OUT.relative_to(ROOT)),
        "teachers_built": [r["teacher_name"] for r in rows],
        "rows": rows,
        "exact_blockers": blockers,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.14 Teacher Feature Cache Build", payload, ["teacher_feature_cache_built", "output_root", "teachers_built", "rows", "exact_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
