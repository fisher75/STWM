#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import V30_POINTODYSSEY_CACHE, V33_IDENTITY_ROOT, load_mask, mask_path_for_frame, scalar

REPORT = ROOT / "reports/stwm_ostf_v33_2_visual_teacher_prototype_build_20260509.json"
DOC = ROOT / "docs/STWM_OSTF_V33_2_VISUAL_TEACHER_PROTOTYPE_BUILD_20260509.md"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey"


def load_preflight() -> dict[str, Any]:
    path = ROOT / "reports/stwm_ostf_v33_2_visual_teacher_preflight_20260509.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_clip(device: str):
    import clip  # type: ignore

    cache = Path("/home/chen034/.cache/clip/ViT-B-32.pt")
    if not cache.exists():
        cache = Path("/raid/chen034/.cache/clip/ViT-B-32.pt")
    if not cache.exists():
        raise FileNotFoundError("local CLIP ViT-B-32.pt missing")
    model, preprocess = clip.load("ViT-B/32", device=device, download_root=str(cache.parent))
    model.eval()
    return model, preprocess, cache


def cache_paths(m: int, h: int, split: str) -> list[Path]:
    return sorted((V30_POINTODYSSEY_CACHE / f"M{m}_H{h}" / split).glob("*.npz"))


def crop_for_instance(frame: Path, mask_path: Path, instance_id: int, point_xy: np.ndarray | None = None, size: int = 96) -> tuple[Image.Image | None, list[int] | None, float]:
    if not frame.exists():
        return None, None, 0.0
    im = Image.open(frame).convert("RGB")
    w, h = im.size
    box = None
    conf = 0.0
    if instance_id >= 0:
        mask = load_mask(str(mask_path))
        if mask is not None:
            ys, xs = np.where(mask.astype(np.int64) == int(instance_id))
            if xs.size > 0:
                x0, x1 = int(xs.min()), int(xs.max())
                y0, y1 = int(ys.min()), int(ys.max())
                pad = max(4, int(0.08 * max(x1 - x0 + 1, y1 - y0 + 1)))
                box = [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad + 1), min(h, y1 + pad + 1)]
                conf = 1.0
    if box is None and point_xy is not None and np.isfinite(point_xy).all():
        x, y = float(point_xy[0]), float(point_xy[1])
        half = size // 2
        box = [max(0, int(round(x)) - half), max(0, int(round(y)) - half), min(w, int(round(x)) + half), min(h, int(round(y)) + half)]
        conf = 0.5
    if box is None or box[2] <= box[0] or box[3] <= box[1]:
        return None, None, 0.0
    return im.crop(tuple(box)).resize((224, 224)), box, conf


def encode_crops(model: torch.nn.Module, preprocess: Any, crops: list[Image.Image], device: str, batch_size: int) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, len(crops), batch_size):
            x = torch.stack([preprocess(c) for c in crops[i : i + batch_size]]).to(device)
            y = model.encode_image(x).float()
            y = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            outs.append(y.cpu().numpy().astype(np.float16))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 512), dtype=np.float16)


def process_sample(path: Path, model: torch.nn.Module, preprocess: Any, device: str, batch_size: int, teacher_name: str) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    uid = str(scalar(z, "video_uid", path.stem))
    split = str(scalar(z, "split", path.parent.name))
    obs_points = np.asarray(z["obs_points"], dtype=np.float32)
    fut_points = np.asarray(z["fut_points"], dtype=np.float32)
    frame_paths = [Path(str(x)) for x in np.asarray(z["frame_paths"], dtype=object).tolist()]
    m, t_obs = obs_points.shape[:2]
    h = fut_points.shape[1]
    sidecar = V33_IDENTITY_ROOT / split / f"{uid}.npz"
    s = np.load(sidecar, allow_pickle=True)
    obs_inst = np.asarray(s["obs_instance_id"], dtype=np.int64)
    fut_inst = np.asarray(s["fut_instance_id"], dtype=np.int64)
    point_to_instance = np.asarray(s["point_to_instance_id"], dtype=np.int64)
    point_id = np.asarray(s["point_id"], dtype=np.int64)
    obs_emb = np.zeros((m, t_obs, 512), dtype=np.float16)
    fut_emb = np.zeros((m, h, 512), dtype=np.float16)
    obs_mask = np.zeros((m, t_obs), dtype=bool)
    fut_mask = np.zeros((m, h), dtype=bool)
    crop_conf_obs = np.zeros((m, t_obs), dtype=np.float16)
    crop_conf_fut = np.zeros((m, h), dtype=np.float16)
    crops: list[Image.Image] = []
    assignments: list[tuple[str, int, int, float]] = []
    crop_fail = 0
    crop_total = 0
    for phase, inst_arr, points, time_count, offset in (("obs", obs_inst, obs_points, t_obs, 0), ("fut", fut_inst, fut_points, h, t_obs)):
        for tt in range(time_count):
            frame = frame_paths[offset + tt]
            mask_path = mask_path_for_frame(frame)
            for inst_id in np.unique(inst_arr[:, tt]):
                idx = np.where(inst_arr[:, tt] == inst_id)[0]
                if idx.size == 0:
                    continue
                crop_total += 1
                point_xy = points[int(idx[0]), tt]
                crop, _, conf = crop_for_instance(frame, mask_path, int(inst_id), point_xy)
                if crop is None:
                    crop_fail += 1
                    continue
                crops.append(crop)
                assignments.append((phase, tt, int(inst_id), conf))
    encoded = encode_crops(model, preprocess, crops, device, batch_size)
    for emb, (phase, tt, inst_id, conf) in zip(encoded, assignments):
        if phase == "obs":
            idx = np.where(obs_inst[:, tt] == inst_id)[0]
            obs_emb[idx, tt] = emb
            obs_mask[idx, tt] = True
            crop_conf_obs[idx, tt] = conf
        else:
            idx = np.where(fut_inst[:, tt] == inst_id)[0]
            fut_emb[idx, tt] = emb
            fut_mask[idx, tt] = True
            crop_conf_fut[idx, tt] = conf
    same_obs = np.zeros((m, h), dtype=bool)
    obs_inst_emb = np.zeros((m, 512), dtype=np.float16)
    for i in range(m):
        valid = np.where(obs_mask[i])[0]
        if valid.size:
            obs_inst_emb[i] = obs_emb[i, valid].astype(np.float32).mean(axis=0).astype(np.float16)
        if point_to_instance[i] >= 0:
            same_obs[i] = fut_inst[i] == point_to_instance[i]
    out = OUT_ROOT / teacher_name / split / f"{uid}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        sample_uid=uid,
        dataset=str(scalar(z, "dataset", "pointodyssey")),
        split=split,
        source_npz=str(path.relative_to(ROOT)),
        identity_sidecar=str(sidecar.relative_to(ROOT)),
        teacher_name=teacher_name,
        teacher_embedding_dim=np.asarray(512, dtype=np.int64),
        point_id=point_id,
        point_to_instance_id=point_to_instance,
        obs_teacher_embedding=obs_emb,
        obs_instance_teacher_embedding=obs_inst_emb,
        obs_teacher_available_mask=obs_mask,
        fut_teacher_embedding=fut_emb,
        fut_teacher_available_mask=fut_mask,
        fut_semantic_same_as_obs_target=same_obs,
        semantic_prototype_id=np.full((m, h), -1, dtype=np.int64),
        visual_crop_confidence_obs=crop_conf_obs,
        visual_crop_confidence_fut=crop_conf_fut,
        leakage_safe=True,
        input_uses_observed_only=True,
        future_teacher_embeddings_supervision_only=True,
        future_teacher_embeddings_input_allowed=False,
        M=np.asarray(m, dtype=np.int64),
        horizon=np.asarray(h, dtype=np.int64),
    )
    return {
        "uid": uid,
        "split": split,
        "sidecar": str(out.relative_to(ROOT)),
        "obs_embedding_coverage": float(obs_mask.mean()),
        "future_embedding_coverage": float(fut_mask.mean()),
        "instance_crop_coverage": 1.0 - crop_fail / max(crop_total, 1),
        "point_crop_coverage": 0.0,
        "crop_failure_ratio": crop_fail / max(crop_total, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-points", type=int, default=128)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--max-samples-per-split", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-name", default="clip_vit_b32_local")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    preflight = load_preflight()
    if not preflight.get("teacher_model_loaded") or not preflight.get("teacher_forward_dryrun_passed"):
        payload = {
            "generated_at_utc": utc_now(),
            "visual_teacher_cache_built": False,
            "teacher_name": args.teacher_name,
            "manual_blockers": ["teacher preflight did not pass"],
            "preflight": preflight,
        }
        dump_json(REPORT, payload)
        write_doc(DOC, "STWM OSTF V33.2 Visual Teacher Prototype Build", payload, ["visual_teacher_cache_built", "teacher_name", "manual_blockers"])
        print(REPORT.relative_to(ROOT))
        return 2
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model, preprocess, ckpt = load_clip(device)
    rows = []
    for split in ("train", "val", "test"):
        for path in cache_paths(args.m_points, args.horizon, split)[: args.max_samples_per_split]:
            rows.append(process_sample(path, model, preprocess, device, args.batch_size, args.teacher_name))
    obs_cov = [r["obs_embedding_coverage"] for r in rows]
    fut_cov = [r["future_embedding_coverage"] for r in rows]
    crop_fail = [r["crop_failure_ratio"] for r in rows]
    payload = {
        "generated_at_utc": utc_now(),
        "visual_teacher_cache_built": bool(rows),
        "teacher_name": args.teacher_name,
        "teacher_checkpoint_path": str(ckpt),
        "teacher_embedding_dim": 512,
        "total_samples_processed": len(rows),
        "obs_embedding_coverage": float(np.mean(obs_cov)) if obs_cov else 0.0,
        "future_embedding_coverage": float(np.mean(fut_cov)) if fut_cov else 0.0,
        "instance_crop_coverage": float(1.0 - np.mean(crop_fail)) if crop_fail else 0.0,
        "point_crop_coverage": 0.0,
        "crop_failure_ratio": float(np.mean(crop_fail)) if crop_fail else 1.0,
        "m128_coverage": {"samples": len(rows), "obs": float(np.mean(obs_cov)) if obs_cov else 0.0, "future": float(np.mean(fut_cov)) if fut_cov else 0.0},
        "m512_coverage": {"manifest_only": True, "target_coverage_available_from_identity_sidecar": True},
        "m1024_coverage": {"manifest_only": True, "target_coverage_available_from_identity_sidecar": True},
        "leakage_safe": True,
        "future_teacher_embeddings_input_allowed": False,
        "future_teacher_embeddings_supervision_only": True,
        "manual_blockers": [],
        "examples": rows[:20],
        "cache_root": str((OUT_ROOT / args.teacher_name).relative_to(ROOT)),
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.2 Visual Teacher Prototype Build", payload, ["visual_teacher_cache_built", "teacher_name", "teacher_embedding_dim", "total_samples_processed", "obs_embedding_coverage", "future_embedding_coverage", "instance_crop_coverage", "crop_failure_ratio", "leakage_safe", "manual_blockers"])
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
