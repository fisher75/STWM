#!/usr/bin/env python3
"""从 video-derived CoTracker trace cache 构建 V35 observed semantic measurement cache。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import clip  # type: ignore
import numpy as np
import setproctitle
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_10_video_observed_semantic_measurement_bank/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_10_video_observed_semantic_measurement_cache_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_10_VIDEO_OBSERVED_SEMANTIC_MEASUREMENT_CACHE_20260515.md"
PAD_DIM = 768


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def scalar(x: np.ndarray) -> Any:
    return np.asarray(x).item()


def list_npz(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.npz"))


def crop_for_object(z: Any, obj_idx: int, t: int, pad: float = 0.08) -> Image.Image | None:
    frame_paths = np.asarray(z["frame_paths"], dtype=object)
    if t >= len(frame_paths):
        return None
    frame_path = Path(str(frame_paths[t]))
    if not frame_path.exists():
        return None
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    vis = np.asarray(z["visibility"]).astype(bool)
    pts = tracks[obj_idx, :, t]
    keep = vis[obj_idx, :, t]
    if not keep.any():
        keep = np.ones((pts.shape[0],), dtype=bool)
    pts = pts[keep]
    raw_w, raw_h = [float(x) for x in np.asarray(z["raw_size"]).tolist()]
    resized_w, resized_h = [float(x) for x in np.asarray(z["resized_size"]).tolist()]
    sx = raw_w / max(resized_w, 1.0)
    sy = raw_h / max(resized_h, 1.0)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    bw = max(x2 - x1, 4.0)
    bh = max(y2 - y1, 4.0)
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad
    box = (
        int(max(0, min(raw_w - 1, round(x1 * sx)))),
        int(max(0, min(raw_h - 1, round(y1 * sy)))),
        int(max(1, min(raw_w, round(x2 * sx)))),
        int(max(1, min(raw_h, round(y2 * sy)))),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    with Image.open(frame_path) as im:
        return im.convert("RGB").crop(box).copy()


@torch.no_grad()
def encode_crops(model: Any, preprocess: Any, device: torch.device, crops: list[Image.Image], batch_size: int) -> np.ndarray:
    feats: list[torch.Tensor] = []
    for start in range(0, len(crops), batch_size):
        batch = torch.stack([preprocess(im) for im in crops[start : start + batch_size]], dim=0).to(device)
        feat = F.normalize(model.encode_image(batch).float(), dim=-1).detach().cpu()
        feats.append(feat)
    x = torch.cat(feats, dim=0).numpy().astype(np.float32) if feats else np.zeros((0, 512), dtype=np.float32)
    if x.shape[1] < PAD_DIM:
        x = np.pad(x, ((0, 0), (0, PAD_DIM - x.shape[1])), mode="constant")
    elif x.shape[1] > PAD_DIM:
        x = x[:, :PAD_DIM]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-6)
    return x.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", type=str, default=str(CACHE_ROOT))
    ap.add_argument("--out-root", type=str, default=str(OUT_ROOT))
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=str(device), download_root=str(Path.home() / ".cache" / "clip"))
    model.eval()
    paths = list_npz(cache_root)
    if args.max_samples > 0:
        paths = paths[: args.max_samples]
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    for p in paths:
        try:
            z = np.load(p, allow_pickle=True)
            split = str(scalar(z["split"]))
            out_dir = out_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
            vis = np.asarray(z["visibility"]).astype(bool)
            conf = np.asarray(z["confidence"], dtype=np.float32)
            obs_len = int(scalar(z["obs_len"]))
            horizon = int(scalar(z["horizon"]))
            obj_n, per_obj_m, _, _ = tracks.shape
            point_n = obj_n * per_obj_m
            crops: list[Image.Image] = []
            crop_index: list[tuple[int, int]] = []
            for obj in range(obj_n):
                for t in range(obs_len):
                    crop = crop_for_object(z, obj, t)
                    if crop is not None:
                        crops.append(crop)
                        crop_index.append((obj, t))
            encoded = encode_crops(model, preprocess, device, crops, args.batch_size)
            obj_obs = np.zeros((obj_n, obs_len, PAD_DIM), dtype=np.float32)
            obj_mask = np.zeros((obj_n, obs_len), dtype=bool)
            for row_i, (obj, t) in enumerate(crop_index):
                obj_obs[obj, t] = encoded[row_i]
                obj_mask[obj, t] = True
            point_obs = np.repeat(obj_obs[:, None, :, :], per_obj_m, axis=1).reshape(point_n, obs_len, PAD_DIM)
            point_mask = np.repeat(obj_mask[:, None, :], per_obj_m, axis=1).reshape(point_n, obs_len)
            point_conf = conf[:, :, :obs_len].reshape(point_n, obs_len).astype(np.float32)
            inst_emb = point_obs.sum(axis=1) / np.maximum(point_mask.sum(axis=1, keepdims=True), 1.0)
            inst_emb = inst_emb / np.maximum(np.linalg.norm(inst_emb, axis=1, keepdims=True), 1e-6)
            point_inst = np.repeat(np.asarray(z["object_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(point_n)
            semantic_id = np.repeat(np.asarray(z["semantic_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(point_n)
            point_id = np.asarray(z["point_id"], dtype=np.int64).reshape(point_n)
            np.savez_compressed(
                out_dir / p.name,
                sample_uid=str(scalar(z["item_key"])).replace("::", "__"),
                item_key=str(scalar(z["item_key"])),
                split=split,
                dataset=str(scalar(z["dataset"])),
                point_id=point_id,
                point_to_instance_id=point_inst,
                semantic_id=semantic_id,
                obs_points=tracks[:, :, :obs_len].reshape(point_n, obs_len, 2).astype(np.float32),
                obs_vis=vis[:, :, :obs_len].reshape(point_n, obs_len).astype(bool),
                obs_conf=conf[:, :, :obs_len].reshape(point_n, obs_len).astype(np.float32),
                obs_semantic_measurements=point_obs,
                obs_semantic_measurement_mask=point_mask,
                obs_measurement_teacher_name="openai_clip_vit_b32_local_padded768",
                obs_measurement_confidence=point_conf * point_mask.astype(np.float32),
                instance_observed_semantic_measurement=inst_emb.astype(np.float32),
                teacher_agreement_score=point_mask.astype(np.float32),
                frame_paths=np.asarray(z["frame_paths"], dtype=object),
                raw_frame_paths_available=True,
                future_teacher_embeddings_supervision_only=False,
                future_teacher_embeddings_input_allowed=False,
                leakage_safe=True,
                trace_source_npz=str(p.relative_to(ROOT)),
            )
            rows.append(
                {
                    "cache_path": str(p.relative_to(ROOT)),
                    "output_path": str((out_dir / p.name).relative_to(ROOT)),
                    "split": split,
                    "dataset": str(scalar(z["dataset"])),
                    "object_count": int(obj_n),
                    "point_count": int(point_n),
                    "obs_measurement_coverage": float(point_mask.mean()),
                    "obs_visibility_mean": float(vis[:, :, :obs_len].mean()),
                    "raw_frame_paths_available": True,
                }
            )
        except Exception as exc:
            blockers.append(f"{p}: {type(exc).__name__}: {exc}")
    coverage = [r["obs_measurement_coverage"] for r in rows]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_observed_semantic_measurement_cache_built": bool(rows),
        "cache_root": str(cache_root.relative_to(ROOT)),
        "out_root": str(out_root.relative_to(ROOT)),
        "teacher_name": "openai_clip_vit_b32_local_padded768",
        "frozen_teacher_measurement_only": True,
        "future_teacher_embeddings_input_allowed": False,
        "sample_count": len(rows),
        "point_count": int(sum(r["point_count"] for r in rows)),
        "measurement_coverage_mean": float(np.mean(coverage)) if coverage else 0.0,
        "rows": rows,
        "exact_blockers": blockers[:20],
        "leakage_safe": True,
        "recommended_next_step": "rerun_v35_video_input_closure_with_real_measurements",
        "中文结论": "已从 CoTracker M128/H32 video-derived trace cache 的 observed frames 构建 frozen CLIP crop measurement cache；future frames/teacher embeddings 未作为输入。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.10 Video Observed Semantic Measurement Cache\n\n"
        f"- video_observed_semantic_measurement_cache_built: {report['video_observed_semantic_measurement_cache_built']}\n"
        f"- sample_count: {len(rows)}\n"
        f"- point_count: {report['point_count']}\n"
        f"- measurement_coverage_mean: {report['measurement_coverage_mean']}\n"
        f"- frozen_teacher_measurement_only: true\n"
        f"- future_teacher_embeddings_input_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"sample_count": len(rows), "recommended_next_step": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
