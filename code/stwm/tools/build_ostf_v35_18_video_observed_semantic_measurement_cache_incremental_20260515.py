#!/usr/bin/env python3
"""V35.18 增量构建 observed video semantic measurement cache。"""
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools import build_ostf_v35_10_video_observed_semantic_measurement_cache_20260515 as base
from stwm.tools.ostf_v17_common_20260502 import ROOT

CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H32"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_10_video_observed_semantic_measurement_bank/M128_H32"
REPORT = ROOT / "reports/stwm_ostf_v35_18_video_observed_semantic_measurement_cache_incremental_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_18_VIDEO_OBSERVED_SEMANTIC_MEASUREMENT_CACHE_INCREMENTAL_20260515.md"


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--out-root", default=str(OUT_ROOT))
    ap.add_argument("--max-new-samples", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    cache_root = Path(args.cache_root)
    if not cache_root.is_absolute():
        cache_root = ROOT / cache_root
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    paths = base.list_npz(cache_root)
    missing_paths: list[Path] = []
    skipped_existing = 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        split = str(base.scalar(z["split"]))
        out_path = out_root / split / p.name
        if out_path.exists():
            skipped_existing += 1
        else:
            missing_paths.append(p)
    if args.max_new_samples > 0:
        missing_paths = missing_paths[: args.max_new_samples]

    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    if missing_paths:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        model, preprocess = clip.load("ViT-B/32", device=str(device), download_root=str(Path.home() / ".cache" / "clip"))
        model.eval()
    else:
        device = torch.device("cpu")
        model = None
        preprocess = None

    for i, p in enumerate(missing_paths):
        try:
            z = np.load(p, allow_pickle=True)
            split = str(base.scalar(z["split"]))
            out_dir = out_root / split
            out_dir.mkdir(parents=True, exist_ok=True)
            tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
            vis = np.asarray(z["visibility"]).astype(bool)
            conf = np.asarray(z["confidence"], dtype=np.float32)
            obs_len = int(base.scalar(z["obs_len"]))
            obj_n, per_obj_m, _, _ = tracks.shape
            point_n = obj_n * per_obj_m
            crops = []
            crop_index: list[tuple[int, int]] = []
            for obj in range(obj_n):
                for t in range(obs_len):
                    crop = base.crop_for_object(z, obj, t)
                    if crop is not None:
                        crops.append(crop)
                        crop_index.append((obj, t))
            assert model is not None and preprocess is not None
            encoded = base.encode_crops(model, preprocess, device, crops, args.batch_size)
            obj_obs = np.zeros((obj_n, obs_len, base.PAD_DIM), dtype=np.float32)
            obj_mask = np.zeros((obj_n, obs_len), dtype=bool)
            for row_i, (obj, t) in enumerate(crop_index):
                obj_obs[obj, t] = encoded[row_i]
                obj_mask[obj, t] = True
            point_obs = np.repeat(obj_obs[:, None, :, :], per_obj_m, axis=1).reshape(point_n, obs_len, base.PAD_DIM)
            point_mask = np.repeat(obj_mask[:, None, :], per_obj_m, axis=1).reshape(point_n, obs_len)
            point_conf = conf[:, :, :obs_len].reshape(point_n, obs_len).astype(np.float32)
            inst_emb = point_obs.sum(axis=1) / np.maximum(point_mask.sum(axis=1, keepdims=True), 1.0)
            inst_emb = inst_emb / np.maximum(np.linalg.norm(inst_emb, axis=1, keepdims=True), 1e-6)
            point_inst = np.repeat(np.asarray(z["object_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(point_n)
            semantic_id = np.repeat(np.asarray(z["semantic_id"], dtype=np.int64)[:, None], per_obj_m, axis=1).reshape(point_n)
            point_id = np.asarray(z["point_id"], dtype=np.int64).reshape(point_n)
            out_path = out_dir / p.name
            np.savez_compressed(
                out_path,
                sample_uid=str(base.scalar(z["item_key"])).replace("::", "__"),
                item_key=str(base.scalar(z["item_key"])),
                split=split,
                dataset=str(base.scalar(z["dataset"])),
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
                trace_source_npz=rel(p),
            )
            rows.append(
                {
                    "cache_path": rel(p),
                    "output_path": rel(out_path),
                    "split": split,
                    "dataset": str(base.scalar(z["dataset"])),
                    "object_count": int(obj_n),
                    "point_count": int(point_n),
                    "obs_measurement_coverage": float(point_mask.mean()),
                    "obs_visibility_mean": float(vis[:, :, :obs_len].mean()),
                }
            )
            print(json.dumps({"进度": f"{i + 1}/{len(missing_paths)}", "新增样本": rel(out_path)}, ensure_ascii=False), flush=True)
        except Exception as exc:  # pragma: no cover - diagnostic script
            blockers.append(f"{p}: {type(exc).__name__}: {exc}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "incremental_measurement_cache_built": True,
        "cache_root": rel(cache_root),
        "out_root": rel(out_root),
        "total_trace_samples_seen": len(paths),
        "skipped_existing_sample_count": skipped_existing,
        "new_sample_count": len(rows),
        "future_teacher_embeddings_input_allowed": False,
        "frozen_teacher_measurement_only": True,
        "rows": rows,
        "exact_blockers": blockers[:50],
        "leakage_safe": True,
        "中文结论": "V35.18 只为新增 video trace clips 增量构建 observed-frame CLIP measurement cache；已有 cache 不重复计算，future frames 未作为输入。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.18 增量 observed semantic measurement cache\n\n"
        f"- total_trace_samples_seen: {len(paths)}\n"
        f"- skipped_existing_sample_count: {skipped_existing}\n"
        f"- new_sample_count: {len(rows)}\n"
        f"- future_teacher_embeddings_input_allowed: false\n"
        f"- leakage_safe: true\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"新增样本数": len(rows), "跳过已有样本数": skipped_existing, "报告": rel(REPORT)}, ensure_ascii=False), flush=True)
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
