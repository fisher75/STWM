#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
import torch
from PIL import Image, ImageDraw

setproctitle.setproctitle("python")


ROOT = Path(__file__).resolve().parents[3]
PREDECODE_ROOT = ROOT / "data/processed/stage2_tusb_v3_predecode_cache_20260418"
OUT_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v15c"


def _apply_process_title() -> None:
    try:
        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM CoTracker Object-Dense Teacher V15C", ""]
    for key in [
        "teacher_run_success",
        "teacher_source",
        "processed_clip_count",
        "failed_clip_count",
        "object_count",
        "point_count",
        "valid_point_ratio",
        "official_repo_path",
        "checkpoint_path",
        "gpu_id",
        "runtime_seconds",
        "next_step_if_failed",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _repo_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _gpu_id() -> str | None:
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _norm_key(path: Path) -> str:
    parts = path.stem.split("__", 2)
    if len(parts) == 3:
        ds, _split, clip = parts
        ds_norm = "VIPSEG" if ds.lower() == "vipseg" else ds.upper()
        return f"{ds_norm}::{clip}"
    return path.stem


def _mixed_split_map() -> dict[str, str]:
    report = _load_json(ROOT / "reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json")
    splits = report.get("splits", {})
    out: dict[str, str] = {}
    if isinstance(splits, dict):
        for split, keys in splits.items():
            for key in keys:
                out[str(key)] = str(split)
    return out


def _scalar_obj(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


def _numeric_stem(path: Path) -> int | None:
    try:
        return int(path.stem)
    except Exception:
        return None


def _frame_sequence(anchor: Path, total: int = 16, preferred_query_frame: int = 7) -> tuple[list[Path] | None, int | None, str | None]:
    if not anchor.exists():
        return None, None, "anchor_frame_missing"
    exts = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        [
            p
            for p in anchor.parent.iterdir()
            if p.suffix.lower() in exts and not p.name.startswith("._") and not p.name.startswith(".")
        ],
        key=lambda p: (_numeric_stem(p) is None, _numeric_stem(p) or 0, p.name),
    )
    try:
        idx = frames.index(anchor)
    except ValueError:
        return None, None, "anchor_not_in_parent_frame_list"
    start = idx - preferred_query_frame
    end = start + total
    if start >= 0 and end <= len(frames):
        return frames[start:end], preferred_query_frame, None
    # If the anchor is at the beginning of the available clip, use it as the first visible observed frame.
    start = idx
    end = start + total
    if end <= len(frames):
        return frames[start:end], 0, None
    return None, None, f"insufficient_contiguous_frames_for_last_or_first_visible_query_idx_{idx}_frame_count_{len(frames)}"


def _load_video(frames: list[Path], max_side: int) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int], tuple[float, float]]:
    imgs = [Image.open(p).convert("RGB") for p in frames]
    raw_w, raw_h = imgs[0].size
    scale = min(float(max_side) / max(raw_w, raw_h), 1.0)
    new_w = max(16, int(round(raw_w * scale)))
    new_h = max(16, int(round(raw_h * scale)))
    arrs = []
    for img in imgs:
        if img.size != (raw_w, raw_h):
            img = img.resize((raw_w, raw_h), Image.BILINEAR)
        img_r = img.resize((new_w, new_h), Image.BILINEAR)
        arrs.append(np.asarray(img_r, dtype=np.float32))
    video = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)[None].contiguous()
    return video, (raw_w, raw_h), (new_w, new_h), (new_w / raw_w, new_h / raw_h)


def _sample_points(mask: np.ndarray, m: int, seed: int) -> np.ndarray:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rng = np.random.default_rng(seed)
    # Use boundary+interior mixture to avoid all-center tracks while staying inside the query mask.
    fg = mask.astype(bool)
    padded = np.pad(fg, 1, constant_values=False)
    neigh = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    boundary = fg & (neigh < 4)
    by, bx = np.where(boundary)
    iy, ix = np.where(fg & ~boundary)
    pts = []
    nb = min(max(m // 4, 1), len(bx)) if len(bx) else 0
    if nb:
        idx = rng.choice(len(bx), size=nb, replace=len(bx) < nb)
        pts.append(np.stack([bx[idx], by[idx]], axis=-1))
    ni = m - nb
    pool_x, pool_y = (ix, iy) if len(ix) else (xs, ys)
    idx = rng.choice(len(pool_x), size=ni, replace=len(pool_x) < ni)
    pts.append(np.stack([pool_x[idx], pool_y[idx]], axis=-1))
    out = np.concatenate(pts, axis=0).astype(np.float32)
    rng.shuffle(out)
    return out


def _query_points(pre: np.lib.npyio.NpzFile, m: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, str | None]:
    if "semantic_instance_id_map" not in pre.files:
        return np.zeros((0, m, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), "missing_instance_id_map"
    inst = np.asarray(pre["semantic_instance_id_map"])
    sem_ids = np.asarray(pre.get("semantic_entity_dominant_instance_id", np.asarray([], dtype=np.int64)), dtype=np.int64).reshape(-1)
    if (len(sem_ids) == 0 or not any(bool((inst == int(x)).any()) for x in sem_ids.tolist())) and "semantic_source_summary_json" in pre.files:
        try:
            summary = _scalar_obj(pre["semantic_source_summary_json"])
            if isinstance(summary, dict) and "target_instance_ids" in summary:
                sem_ids = np.asarray(summary["target_instance_ids"], dtype=np.int64).reshape(-1)
        except Exception:
            pass
    object_points: list[np.ndarray] = []
    object_ids: list[int] = []
    semantic_ids: list[int] = []
    for obj_idx, sem_id in enumerate(sem_ids.tolist()):
        mask = inst == int(sem_id)
        pts = _sample_points(mask, m, seed + obj_idx * 1009)
        if pts.shape[0] == m:
            object_points.append(pts)
            object_ids.append(obj_idx)
            semantic_ids.append(int(sem_id))
    if not object_points:
        return np.zeros((0, m, 2), dtype=np.float32), np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64), "no_object_mask_pixels_for_target_ids"
    return np.stack(object_points), np.asarray(object_ids, dtype=np.int64), np.asarray(semantic_ids, dtype=np.int64), None


def _select_items(max_train: int, max_val: int, max_test: int, split_map: dict[str, str]) -> list[Path]:
    quotas = {"train": max_train, "val": max_val, "test": max_test}
    selected: list[Path] = []
    counts = Counter()
    paths = sorted(PREDECODE_ROOT.glob("*/*.npz"))
    # Interleave datasets and splits deterministically.
    buckets: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in paths:
        split = split_map.get(_norm_key(path), path.parent.name)
        ds = _norm_key(path).split("::", 1)[0]
        if split in quotas:
            buckets[(split, ds)].append(path)
    rng = random.Random(20260502)
    for key in buckets:
        rng.shuffle(buckets[key])
    for split, quota in quotas.items():
        if quota <= 0:
            continue
        candidate_quota = max(quota * 8, quota)
        split_candidates: list[Path] = []
        for ds in ["VSPW", "VIPSEG"]:
            split_candidates.extend(buckets.get((split, ds), []))
        # Round-robin preserving some dataset balance.
        ds_lists = [buckets.get((split, ds), []) for ds in ["VSPW", "VIPSEG"]]
        while counts[split] < candidate_quota and any(ds_lists):
            progressed = False
            for rows in ds_lists:
                if rows and counts[split] < candidate_quota:
                    selected.append(rows.pop())
                    counts[split] += 1
                    progressed = True
            if not progressed:
                break
        if counts[split] < candidate_quota:
            for p in split_candidates:
                if p not in selected:
                    selected.append(p)
                    counts[split] += 1
                    if counts[split] >= candidate_quota:
                        break
    return selected


def _run_clip(model: torch.nn.Module, pre_path: Path, args: argparse.Namespace, device: torch.device, split_map: dict[str, str]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    item_key = _norm_key(pre_path)
    split = split_map.get(item_key, pre_path.parent.name)
    dataset = item_key.split("::", 1)[0]
    try:
        pre = np.load(pre_path, allow_pickle=True)
        anchor = Path(str(_scalar_obj(pre["semantic_frame_path"])))
        frames, query_frame, frame_err = _frame_sequence(anchor, total=args.obs_len + args.horizon, preferred_query_frame=args.obs_len - 1)
        if frame_err or frames is None:
            return None, {"item_key": item_key, "split": split, "dataset": dataset, "reason": frame_err}
        q_points_raw, object_ids, semantic_ids, q_err = _query_points(pre, args.m, int(hashlib.md5(item_key.encode()).hexdigest()[:8], 16))
        if q_err:
            return None, {"item_key": item_key, "split": split, "dataset": dataset, "reason": q_err}
        video, raw_size, resized_size, scale_xy = _load_video(frames, args.max_side)
        sx, sy = scale_xy
        q_scaled = q_points_raw.copy()
        q_scaled[..., 0] *= sx
        q_scaled[..., 1] *= sy
        obj_count = int(q_scaled.shape[0])
        queries = np.zeros((obj_count * args.m, 3), dtype=np.float32)
        queries[:, 0] = float(query_frame)
        queries[:, 1:] = q_scaled.reshape(-1, 2)
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
            tracks, visibility = model(
                video.to(device),
                queries=torch.from_numpy(queries)[None].to(device),
                backward_tracking=True,
            )
        peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        tr = tracks[0].detach().float().cpu().numpy()  # [T, N, 2] at resized scale
        vis = visibility[0].detach().cpu().numpy().astype(bool)
        tr[..., 0] /= sx
        tr[..., 1] /= sy
        tr = tr.reshape(args.obs_len + args.horizon, obj_count, args.m, 2).transpose(1, 2, 0, 3)
        vis = vis.reshape(args.obs_len + args.horizon, obj_count, args.m).transpose(1, 2, 0)
        point_ids = np.arange(obj_count * args.m, dtype=np.int64).reshape(obj_count, args.m)
        out_dir = OUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_out = out_dir / f"{item_key.replace('::', '__')}.npz"
        np.savez_compressed(
            clip_out,
            item_key=np.asarray(item_key, dtype=object),
            split=np.asarray(split, dtype=object),
            dataset=np.asarray(dataset, dtype=object),
            frame_paths=np.asarray([str(p) for p in frames], dtype=object),
            query_frame=np.asarray(query_frame, dtype=np.int32),
            obs_len=np.asarray(args.obs_len, dtype=np.int32),
            horizon=np.asarray(args.horizon, dtype=np.int32),
            M=np.asarray(args.m, dtype=np.int32),
            object_id=object_ids,
            semantic_id=semantic_ids,
            point_id=point_ids,
            query_points_xy=q_points_raw.astype(np.float32),
            tracks_xy=tr.astype(np.float32),
            visibility=vis,
            confidence=vis.astype(np.float32),
            raw_size=np.asarray(raw_size, dtype=np.int32),
            resized_size=np.asarray(resized_size, dtype=np.int32),
            teacher_source=np.asarray("cotracker_official", dtype=object),
            official_repo_path=np.asarray(str(args.repo_path), dtype=object),
            checkpoint_path=np.asarray(str(args.checkpoint), dtype=object),
            no_future_box_projection=np.asarray(True),
            teacher_uses_full_obs_future_clip_as_target=np.asarray(True),
            stwm_input_restricted_to_observed=np.asarray(True),
            predecode_path=np.asarray(str(pre_path), dtype=object),
        )
        return {
            "item_key": item_key,
            "split": split,
            "dataset": dataset,
            "cache_path": str(clip_out.relative_to(ROOT)),
            "object_count": obj_count,
            "point_count": int(obj_count * args.m),
            "valid_point_ratio": float(vis.mean()) if vis.size else 0.0,
            "query_frame": int(query_frame),
            "raw_frame_paths_available": True,
            "raw_size": raw_size,
            "resized_size": resized_size,
            "peak_gpu_memory_bytes": peak_mem,
        }, None
    except Exception as exc:
        return None, {"item_key": item_key, "split": split, "dataset": dataset, "reason": repr(exc)}


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default="baselines/repos/co-tracker")
    parser.add_argument("--checkpoint", default="baselines/checkpoints/cotracker/scaled_offline.pth")
    parser.add_argument("--max-train", type=int, default=20)
    parser.add_argument("--max-val", type=int, default=10)
    parser.add_argument("--max-test", type=int, default=10)
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.repo_path = str(ROOT / args.repo_path)
    args.checkpoint = str(ROOT / args.checkpoint)
    repo = Path(args.repo_path)
    ckpt = Path(args.checkpoint)
    t0 = time.time()
    failures: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if not repo.exists() or not ckpt.exists():
        payload = {
            "teacher_run_success": False,
            "teacher_source": "missing",
            "official_repo_path": str(repo),
            "checkpoint_path": str(ckpt),
            "blocker": "cotracker_repo_or_checkpoint_missing",
            "next_step_if_failed": "fix_cotracker_integration",
        }
        _dump(ROOT / "reports/stwm_cotracker_object_dense_teacher_v15c_20260502.json", payload)
        _write_doc(ROOT / "docs/STWM_COTRACKER_OBJECT_DENSE_TEACHER_V15C_20260502.md", payload)
        return 1
    sys.path.insert(0, str(repo))
    from cotracker.predictor import CoTrackerPredictor  # type: ignore

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = CoTrackerPredictor(checkpoint=str(ckpt), offline=True, window_len=60).to(device).eval()
    split_map = _mixed_split_map()
    selected = _select_items(args.max_train, args.max_val, args.max_test, split_map)
    target_quotas = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    success_by_split: Counter[str] = Counter()
    for idx, pre_path in enumerate(selected):
        split_name = split_map.get(_norm_key(pre_path), pre_path.parent.name)
        if success_by_split[split_name] >= target_quotas.get(split_name, 0):
            continue
        if all(success_by_split[s] >= q for s, q in target_quotas.items()):
            break
        row, fail = _run_clip(model, pre_path, args, device, split_map)
        if row:
            rows.append(row)
            success_by_split[row["split"]] += 1
        if fail:
            failures.append(fail)
        print(f"[{idx+1}/{len(selected)}] ok={len(rows)} failed={len(failures)} {pre_path.name}", flush=True)
    split_counts = Counter(row["split"] for row in rows)
    dataset_counts = Counter(row["dataset"] for row in rows)
    point_count = int(sum(row["point_count"] for row in rows))
    object_count = int(sum(row["object_count"] for row in rows))
    payload = {
        "teacher_run_success": bool(rows),
        "teacher_source": "cotracker_official" if rows else "none",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repo_path": str(repo),
        "official_commit_hash": _repo_commit(repo),
        "checkpoint_path": str(ckpt),
        "checkpoint_exists": ckpt.exists(),
        "exact_command": " ".join(sys.argv),
        "gpu_id": _gpu_id(),
        "device": str(device),
        "runtime_seconds": float(time.time() - t0),
        "processed_clip_count": len(rows),
        "failed_clip_count": len(failures),
        "requested_split_counts": {"train": args.max_train, "val": args.max_val, "test": args.max_test},
        "split_source": "reports/stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json item_key map when available",
        "processed_split_counts": dict(split_counts),
        "processed_dataset_counts": dict(dataset_counts),
        "M": args.m,
        "obs_len": args.obs_len,
        "horizon": args.horizon,
        "object_count": object_count,
        "point_count": point_count,
        "valid_point_ratio": float(np.mean([row["valid_point_ratio"] for row in rows])) if rows else 0.0,
        "average_points_per_object": float(point_count / max(object_count, 1)),
        "average_points_per_scene": float(point_count / max(len(rows), 1)),
        "peak_memory_bytes_max": int(max([row["peak_gpu_memory_bytes"] for row in rows], default=0)),
        "cache_root": str(OUT_ROOT.relative_to(ROOT)),
        "clip_caches": rows,
        "failed_clips": failures,
        "failure_reason_top": dict(Counter(str(f["reason"]) for f in failures).most_common(10)),
        "no_future_box_projection": True,
        "teacher_uses_full_obs_future_clip_as_target": True,
        "stwm_input_restricted_to_observed": True,
        "next_step_if_failed": "fix_cotracker_integration" if not rows else None,
    }
    _dump(ROOT / "reports/stwm_cotracker_object_dense_teacher_v15c_20260502.json", payload)
    _write_doc(ROOT / "docs/STWM_COTRACKER_OBJECT_DENSE_TEACHER_V15C_20260502.md", payload)
    print("reports/stwm_cotracker_object_dense_teacher_v15c_20260502.json")
    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
