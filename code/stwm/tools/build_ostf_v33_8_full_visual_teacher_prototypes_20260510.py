#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.build_ostf_v33_2_pointodyssey_visual_teacher_prototypes_20260509 import (
    encode_crops,
    load_clip,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now
from stwm.tools.ostf_v33_semantic_identity_schema_20260509 import V33_IDENTITY_ROOT, load_mask, mask_path_for_frame, scalar


REPORT = ROOT / "reports/stwm_ostf_v33_8_visual_teacher_prototype_build_20260510.json"
DOC = ROOT / "docs/STWM_OSTF_V33_8_VISUAL_TEACHER_PROTOTYPE_BUILD_20260510.md"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_visual_teacher_prototypes/pointodyssey"
OLD_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_2_visual_teacher_prototypes/pointodyssey"


def manifest_cache_paths(split: str) -> list[Path]:
    entries = json.loads((ROOT / "manifests/ostf_v30_external_gt" / f"{split}.json").read_text(encoding="utf-8")).get("entries", [])
    out: list[Path] = []
    for e in entries:
        if int(e.get("H", -1)) != 32 or int(e.get("M", -1)) != 128:
            continue
        p = ROOT / e["cache_path"]
        if p.exists():
            out.append(p)
    return out


def uid_for_cache(path: Path) -> str:
    z = np.load(path, allow_pickle=True)
    return str(np.asarray(z["video_uid"]).item() if "video_uid" in z.files else path.stem)


def valid_sidecar(path: Path) -> bool:
    if not path.exists():
        return False


def copy_old_valid_sidecar(teacher_name: str, split: str, uid: str, out_path: Path) -> bool:
    old = OLD_ROOT / teacher_name / split / f"{uid}.npz"
    if not valid_sidecar(old):
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(old, out_path)
    return True


def crop_cached(
    frame: Path,
    instance_id: int,
    *,
    point_xy: np.ndarray | None,
    image_cache: dict[Path, Image.Image | None],
    mask_cache: dict[Path, np.ndarray | None],
    size: int = 96,
) -> tuple[Image.Image | None, float]:
    if frame not in image_cache:
        image_cache[frame] = Image.open(frame).convert("RGB") if frame.exists() else None
    im = image_cache[frame]
    if im is None:
        return None, 0.0
    w, h = im.size
    box = None
    conf = 0.0
    mask_path = mask_path_for_frame(frame)
    if instance_id >= 0:
        if mask_path not in mask_cache:
            mask_cache[mask_path] = load_mask(str(mask_path))
        mask = mask_cache[mask_path]
        if mask is not None:
            ys, xs = np.where(mask.astype(np.int64) == int(instance_id))
            if xs.size:
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
        return None, 0.0
    return im.crop(tuple(box)).resize((224, 224)), conf


def process_sample_fast(path: Path, model: torch.nn.Module, preprocess: object, device: str, batch_size: int, teacher_name: str) -> dict[str, Any]:
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
    image_cache: dict[Path, Image.Image | None] = {}
    mask_cache: dict[Path, np.ndarray | None] = {}
    crop_fail = 0
    crop_total = 0
    # V33.8 uses instance-level visual semantic prototypes. We encode one
    # reliable crop per instance for observed context and one for future target,
    # then broadcast the prototype across that instance's point/time positions.
    # This preserves semantic-prototype supervision without per-frame crop IO.
    for phase, inst_arr, points, time_count, offset in (("obs", obs_inst, obs_points, t_obs, 0), ("fut", fut_inst, fut_points, h, t_obs)):
        for inst_id in sorted(int(x) for x in np.unique(inst_arr) if int(x) >= 0):
            loc = np.argwhere(inst_arr == inst_id)
            if loc.size == 0:
                continue
            crop_total += 1
            # Prefer the latest observed crop and the earliest future crop.
            pick = loc[np.argmax(loc[:, 1])] if phase == "obs" else loc[np.argmin(loc[:, 1])]
            point_idx, tt = int(pick[0]), int(pick[1])
            frame = frame_paths[offset + tt]
            crop, conf = crop_cached(
                frame,
                inst_id,
                point_xy=points[point_idx, tt],
                image_cache=image_cache,
                mask_cache=mask_cache,
            )
            if crop is None:
                crop_fail += 1
                continue
            crops.append(crop)
            assignments.append((phase, -1, inst_id, conf))
    encoded = encode_crops(model, preprocess, crops, device, batch_size)
    for emb, (phase, tt, inst_id, conf) in zip(encoded, assignments):
        if phase == "obs":
            idx = np.where(obs_inst == inst_id)
            obs_emb[idx] = emb
            obs_mask[idx] = True
            crop_conf_obs[idx] = conf
        else:
            idx = np.where(fut_inst == inst_id)
            fut_emb[idx] = emb
            fut_mask[idx] = True
            crop_conf_fut[idx] = conf
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
    for im in image_cache.values():
        try:
            if im is not None:
                im.close()
        except Exception:
            pass
    return {
        "uid": uid,
        "split": split,
        "sidecar": str(out.relative_to(ROOT)),
        "obs_embedding_coverage": float(obs_mask.mean()),
        "future_embedding_coverage": float(fut_mask.mean()),
        "instance_crop_coverage": 1.0 - crop_fail / max(crop_total, 1),
        "point_crop_coverage": 0.0,
        "crop_failure_ratio": crop_fail / max(crop_total, 1),
        "prototype_granularity": "instance_level_broadcast",
    }
    try:
        z = np.load(path, allow_pickle=True)
        return all(k in z.files for k in ["obs_teacher_embedding", "fut_teacher_embedding", "obs_teacher_available_mask", "fut_teacher_available_mask"]) and bool(np.asarray(z["leakage_safe"]).item())
    except Exception:
        return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--max-samples-per-split", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--teacher-name", default="clip_vit_b32_local")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    try:
        model, preprocess, ckpt = load_clip(device)
        teacher_loaded = True
        blocker = None
    except Exception as exc:
        model = preprocess = ckpt = None
        teacher_loaded = False
        blocker = repr(exc)
    rows: list[dict[str, Any]] = []
    failed: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    skipped: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    reachable: dict[str, int] = {}
    if teacher_loaded:
        import stwm.tools.build_ostf_v33_2_pointodyssey_visual_teacher_prototypes_20260509 as v33_2_builder

        old_out = v33_2_builder.OUT_ROOT
        v33_2_builder.OUT_ROOT = OUT_ROOT
        try:
            for split in ("train", "val", "test"):
                paths = manifest_cache_paths(split)
                reachable[split] = len(paths)
                if args.max_samples_per_split > 0:
                    paths = paths[: args.max_samples_per_split]
                for cache_path in paths:
                    uid = uid_for_cache(cache_path)
                    out_path = OUT_ROOT / args.teacher_name / split / f"{uid}.npz"
                    if not valid_sidecar(out_path) and copy_old_valid_sidecar(args.teacher_name, split, uid, out_path):
                        skipped[split] += 1
                        z = np.load(out_path, allow_pickle=True)
                        rows.append(
                            {
                                "uid": uid,
                                "split": split,
                                "sidecar": str(out_path.relative_to(ROOT)),
                                "obs_embedding_coverage": float(np.asarray(z["obs_teacher_available_mask"]).mean()),
                                "future_embedding_coverage": float(np.asarray(z["fut_teacher_available_mask"]).mean()),
                                "crop_failure_ratio": float(1.0 - np.asarray(z["fut_teacher_available_mask"]).mean()),
                                "skipped_existing": True,
                                "reused_v33_2_sidecar": True,
                            }
                        )
                        continue
                    if valid_sidecar(out_path):
                        skipped[split] += 1
                        z = np.load(out_path, allow_pickle=True)
                        rows.append(
                            {
                                "uid": uid,
                                "split": split,
                                "sidecar": str(out_path.relative_to(ROOT)),
                                "obs_embedding_coverage": float(np.asarray(z["obs_teacher_available_mask"]).mean()),
                                "future_embedding_coverage": float(np.asarray(z["fut_teacher_available_mask"]).mean()),
                                "crop_failure_ratio": float(1.0 - np.asarray(z["fut_teacher_available_mask"]).mean()),
                                "skipped_existing": True,
                                "reused_v33_2_sidecar": False,
                            }
                        )
                        continue
                    try:
                        r = process_sample_fast(cache_path, model, preprocess, device, args.batch_size, args.teacher_name)
                        r["skipped_existing"] = False
                        rows.append(r)
                    except Exception as exc:
                        failed[split].append({"uid": uid, "cache_path": str(cache_path.relative_to(ROOT)), "reason": repr(exc)})
        finally:
            v33_2_builder.OUT_ROOT = old_out
    else:
        reachable = {split: len(manifest_cache_paths(split)) for split in ("train", "val", "test")}
    by_split: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        sr = [r for r in rows if r.get("split") == split]
        by_split[split] = {
            "reachable_samples": reachable.get(split, 0),
            "processed_samples": sum(1 for r in sr if not r.get("skipped_existing")),
            "skipped_existing_samples": skipped[split],
            "failed_samples": len(failed[split]),
            "obs_embedding_coverage": float(np.mean([r["obs_embedding_coverage"] for r in sr])) if sr else 0.0,
            "future_embedding_coverage": float(np.mean([r["future_embedding_coverage"] for r in sr])) if sr else 0.0,
            "instance_crop_coverage": float(1.0 - np.mean([r["crop_failure_ratio"] for r in sr])) if sr else 0.0,
            "crop_failure_ratio": float(np.mean([r["crop_failure_ratio"] for r in sr])) if sr else 1.0,
            "failed": failed[split][:50],
        }
    payload = {
        "generated_at_utc": utc_now(),
        "teacher_name": args.teacher_name,
        "teacher_model_loaded": teacher_loaded,
        "teacher_forward_used": bool(teacher_loaded and rows),
        "teacher_checkpoint_path": str(ckpt) if ckpt is not None else None,
        "total_reachable_samples_by_split": {s: by_split[s]["reachable_samples"] for s in by_split},
        "processed_samples_by_split": {s: by_split[s]["processed_samples"] for s in by_split},
        "skipped_existing_samples_by_split": {s: by_split[s]["skipped_existing_samples"] for s in by_split},
        "failed_samples_by_split": {s: by_split[s]["failed_samples"] for s in by_split},
        "obs_embedding_coverage_by_split": {s: by_split[s]["obs_embedding_coverage"] for s in by_split},
        "future_embedding_coverage_by_split": {s: by_split[s]["future_embedding_coverage"] for s in by_split},
        "instance_crop_coverage_by_split": {s: by_split[s]["instance_crop_coverage"] for s in by_split},
        "crop_failure_ratio_by_split": {s: by_split[s]["crop_failure_ratio"] for s in by_split},
        "leakage_safe": True,
        "future_teacher_embeddings_input_allowed": False,
        "future_teacher_embeddings_supervision_only": True,
        "cache_root": str((OUT_ROOT / args.teacher_name).relative_to(ROOT)),
        "exact_blocker": blocker,
        "by_split": by_split,
    }
    dump_json(REPORT, payload)
    write_doc(DOC, "STWM OSTF V33.8 Visual Teacher Prototype Build", payload, ["teacher_name", "teacher_model_loaded", "teacher_forward_used", "total_reachable_samples_by_split", "processed_samples_by_split", "skipped_existing_samples_by_split", "failed_samples_by_split", "leakage_safe", "future_teacher_embeddings_input_allowed", "future_teacher_embeddings_supervision_only", "exact_blocker"])
    print(REPORT.relative_to(ROOT))
    return 0 if teacher_loaded else 2


if __name__ == "__main__":
    raise SystemExit(main())
