#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as tvf
from omegaconf import OmegaConf
from PIL import Image

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.run_cotracker_object_dense_teacher_v15c_20260502 import (
    _frame_sequence,
    _mixed_split_map,
    _norm_key,
    _query_points,
    _select_items,
)


OUT_ROOT = ROOT / "outputs/cache/stwm_traceanything_object_dense_v2"
DEFAULT_REPORT_PATH = ROOT / "reports/stwm_traceanything_teacher_pilot_v2_20260502.json"
DEFAULT_DOC_PATH = ROOT / "docs/STWM_TRACEANYTHING_TEACHER_PILOT_V2_20260502.md"
V15C_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v15c"
V16_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16"


def _apply_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _scalar(x: Any) -> Any:
    arr = np.asarray(x)
    return arr.item() if arr.shape == () else arr.reshape(-1)[0]


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


def _repo_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _gpu_id() -> str | None:
    return os.environ.get("CUDA_VISIBLE_DEVICES")


def _to_plain_dict(x: Any) -> dict[str, Any]:
    return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x


def _load_traceanything_model(repo: Path, ckpt: Path, device: torch.device) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    sys.path.insert(0, str(repo))
    from trace_anything.trace_anything import TraceAnything, evaluate_bspline_conf  # type: ignore

    cfg = OmegaConf.load(repo / "configs/eval.yaml")
    net_cfg = cfg.get("model", {}).get("net", None) or cfg.get("net", None)
    if net_cfg is None:
        raise KeyError("TraceAnything eval.yaml missing model.net / net block")
    model = TraceAnything(
        encoder_args=_to_plain_dict(net_cfg["encoder_args"]),
        decoder_args=_to_plain_dict(net_cfg["decoder_args"]),
        head_args=_to_plain_dict(net_cfg["head_args"]),
        targeting_mechanism=net_cfg.get("targeting_mechanism", "bspline_conf"),
        poly_degree=net_cfg.get("poly_degree", 10),
        whether_local=False,
    )
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    load_status = {
        "missing_key_count": int(len(missing)),
        "unexpected_key_count": int(len(unexpected)),
        "missing_keys_head": list(missing[:20]),
        "unexpected_keys_head": list(unexpected[:20]),
        "load_consistent_with_official_infer": bool(
            len(unexpected) == 0 and all(k.startswith(("ds_head_local.", "local_head.")) for k in missing)
        ),
    }
    return model, evaluate_bspline_conf, load_status


def _load_views(frames: list[Path], device: torch.device, max_side: int) -> tuple[list[dict[str, Any]], tuple[int, int], tuple[int, int], tuple[float, float]]:
    tfm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5,) * 3, (0.5,) * 3)])
    imgs = [Image.open(p).convert("RGB") for p in frames]
    raw_w, raw_h = imgs[0].size
    if raw_h > raw_w:
        raise RuntimeError(f"portrait_frame_not_supported_by_current_traceanything_adapter raw_size={raw_w}x{raw_h}")
    scale = min(float(max_side) / max(raw_w, raw_h), 1.0)
    resized_w = max(16, int(round(raw_w * scale)))
    resized_h = max(16, int(round(raw_h * scale)))
    resized_w -= resized_w % 16
    resized_h -= resized_h % 16
    resized_w = max(resized_w, 16)
    resized_h = max(resized_h, 16)
    sx = resized_w / float(raw_w)
    sy = resized_h / float(raw_h)
    views: list[dict[str, Any]] = []
    for i, img in enumerate(imgs):
        if img.size != (raw_w, raw_h):
            img = img.resize((raw_w, raw_h), Image.BILINEAR)
        img = img.resize((resized_w, resized_h), Image.BILINEAR)
        tensor = tfm(img).unsqueeze(0).to(device)
        time_step = i / max(1, len(imgs) - 1)
        views.append({"img": tensor, "time_step": time_step})
    return views, (raw_w, raw_h), (resized_w, resized_h), (sx, sy)


def _scaled_box(box: np.ndarray, sx: float, sy: float, width: int, height: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in box.tolist()]
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        return 0, 0, width, height
    xi0 = max(0, min(width - 1, int(math.floor(x0 * sx))))
    yi0 = max(0, min(height - 1, int(math.floor(y0 * sy))))
    xi1 = max(xi0 + 1, min(width, int(math.ceil(x1 * sx))))
    yi1 = max(yi0 + 1, min(height, int(math.ceil(y1 * sy))))
    return xi0, yi0, xi1, yi1


def _visibility_series(pre: np.lib.npyio.NpzFile, object_index: int, total_steps: int) -> np.ndarray:
    if "obs_valid" in pre.files and "fut_valid" in pre.files:
        obs = np.asarray(pre["obs_valid"]).astype(bool)
        fut = np.asarray(pre["fut_valid"]).astype(bool)
        if obs.ndim == 2 and fut.ndim == 2 and object_index < obs.shape[1] and object_index < fut.shape[1]:
            series = np.concatenate([obs[:, object_index], fut[:, object_index]], axis=0)
            if series.shape[0] == total_steps:
                return series
    return np.ones((total_steps,), dtype=bool)


def _extract_tracks_for_m(
    pre: np.lib.npyio.NpzFile,
    preds: list[dict[str, Any]],
    evaluate_bspline_conf: Any,
    query_frame: int,
    q_points_raw: np.ndarray,
    object_ids: np.ndarray,
    semantic_ids: np.ndarray,
    raw_size: tuple[int, int],
    resized_size: tuple[int, int],
    scale_xy: tuple[float, float],
    obs_len: int,
    horizon: int,
) -> dict[str, Any]:
    total_steps = obs_len + horizon
    device = preds[0]["ctrl_pts3d"].device
    sx, sy = scale_xy
    width, height = resized_size
    boxes = np.asarray(pre["entity_boxes_over_time"], dtype=np.float32) if "entity_boxes_over_time" in pre.files else np.zeros((total_steps, q_points_raw.shape[0], 4), dtype=np.float32)
    time_values = torch.tensor([float(_scalar(pred["time"].detach().cpu().numpy())) for pred in preds], device=device)
    query_pred = preds[query_frame]
    query_pts3d_t, query_conf_t = evaluate_bspline_conf(query_pred["ctrl_pts3d"], query_pred["ctrl_conf"], time_values)
    query_pts3d_t = query_pts3d_t.detach().cpu().numpy()
    query_conf_t = query_conf_t.detach().cpu().numpy()

    target_maps: list[np.ndarray] = []
    target_confs: list[np.ndarray] = []
    for t in range(total_steps):
        target_time = time_values[t : t + 1]
        pts_map, conf_map = evaluate_bspline_conf(preds[t]["ctrl_pts3d"], preds[t]["ctrl_conf"], target_time)
        target_maps.append(pts_map[0].detach().cpu().numpy())
        target_confs.append(conf_map[0].detach().cpu().numpy())

    object_count, m = q_points_raw.shape[:2]
    q_scaled = q_points_raw.copy().astype(np.float32)
    q_scaled[..., 0] *= sx
    q_scaled[..., 1] *= sy
    qx = np.clip(np.rint(q_scaled[..., 0]).astype(np.int64), 0, width - 1)
    qy = np.clip(np.rint(q_scaled[..., 1]).astype(np.int64), 0, height - 1)

    tracks_xy = np.zeros((object_count, m, total_steps, 2), dtype=np.float32)
    visibility = np.zeros((object_count, m, total_steps), dtype=bool)
    confidence = np.zeros((object_count, m, total_steps), dtype=np.float32)
    nn_distance = np.zeros((object_count, m, total_steps), dtype=np.float32)
    same_pixel_fraction_per_object: list[float] = []

    for obj_idx in range(object_count):
        source_obj = int(object_ids[obj_idx]) if obj_idx < len(object_ids) else obj_idx
        obj_valid = _visibility_series(pre, source_obj, total_steps)
        for t in range(total_steps):
            q3d = query_pts3d_t[t, qy[obj_idx], qx[obj_idx], :]  # [M,3]
            field = target_maps[t]
            conf_field = target_confs[t]
            if source_obj < boxes.shape[1]:
                x0, y0, x1, y1 = _scaled_box(boxes[t, source_obj], sx, sy, width, height)
            else:
                x0, y0, x1, y1 = 0, 0, width, height
            crop_pts = field[y0:y1, x0:x1].reshape(-1, 3)
            crop_conf = conf_field[y0:y1, x0:x1].reshape(-1)
            crop_w = x1 - x0
            if crop_pts.size == 0 or crop_w <= 0:
                crop_pts = field.reshape(-1, 3)
                crop_conf = conf_field.reshape(-1)
                crop_w = width
                x0, y0 = 0, 0
            dists = ((q3d[:, None, :] - crop_pts[None, :, :]) ** 2).sum(axis=-1)
            arg = dists.argmin(axis=1)
            min_d = dists[np.arange(m), arg]
            yy, xx = np.divmod(arg, crop_w)
            xy = np.stack([x0 + xx, y0 + yy], axis=-1).astype(np.float32)
            conf = crop_conf[arg].astype(np.float32)
            conf_thr = float(np.quantile(crop_conf, 0.25)) if crop_conf.size else 0.0
            vis = np.logical_and(obj_valid[t], conf >= conf_thr)
            tracks_xy[obj_idx, :, t, 0] = xy[:, 0] / max(sx, 1e-6)
            tracks_xy[obj_idx, :, t, 1] = xy[:, 1] / max(sy, 1e-6)
            visibility[obj_idx, :, t] = vis
            confidence[obj_idx, :, t] = conf
            nn_distance[obj_idx, :, t] = min_d
        unique_tracks = np.unique(np.round(tracks_xy[obj_idx].reshape(m, -1), decimals=3), axis=0).shape[0]
        same_pixel_fraction_per_object.append(1.0 - float(unique_tracks) / max(1, m))

    point_id = np.arange(object_count * m, dtype=np.int64).reshape(object_count, m)
    return {
        "object_id": object_ids.astype(np.int64),
        "semantic_id": semantic_ids.astype(np.int64),
        "point_id": point_id,
        "query_points_xy": q_points_raw.astype(np.float32),
        "tracks_xy": tracks_xy,
        "visibility": visibility,
        "confidence": confidence,
        "nn_distance": nn_distance,
        "trajectory_field_query_confidence": query_conf_t[:, qy, qx].transpose(1, 2, 0).astype(np.float32),
        "same_trajectory_fraction": float(np.mean(same_pixel_fraction_per_object)) if same_pixel_fraction_per_object else 1.0,
        "valid_point_ratio": float(visibility.mean()) if visibility.size else 0.0,
        "native_visibility_available": False,
        "visibility_source": "estimated_from_predecode_object_valid_plus_traceanything_confidence",
        "trajectory_field_adapter": "query_frame_bspline_3d_to_target_frame_nearest_pixel_within_target_object_box",
    }


def _save_cache(
    cache_dir: Path,
    *,
    item_key: str,
    split: str,
    dataset: str,
    frame_paths: list[Path],
    query_frame: int,
    obs_len: int,
    horizon: int,
    m: int,
    raw_size: tuple[int, int],
    resized_size: tuple[int, int],
    repo: Path,
    repo_commit: str | None,
    ckpt: Path,
    pre_path: Path,
    extract: dict[str, Any],
) -> Path:
    out_dir = cache_dir / f"M{m}_H{horizon}" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{item_key.replace('::', '__')}.npz"
    np.savez_compressed(
        out_path,
        item_key=np.asarray(item_key, dtype=object),
        split=np.asarray(split, dtype=object),
        dataset=np.asarray(dataset, dtype=object),
        frame_paths=np.asarray([str(p) for p in frame_paths], dtype=object),
        query_frame=np.asarray(query_frame, dtype=np.int32),
        obs_len=np.asarray(obs_len, dtype=np.int32),
        horizon=np.asarray(horizon, dtype=np.int32),
        M=np.asarray(m, dtype=np.int32),
        object_id=extract["object_id"],
        semantic_id=extract["semantic_id"],
        point_id=extract["point_id"],
        query_points_xy=extract["query_points_xy"],
        tracks_xy=extract["tracks_xy"].astype(np.float32),
        visibility=extract["visibility"],
        confidence=extract["confidence"].astype(np.float32),
        nn_distance=extract["nn_distance"].astype(np.float32),
        raw_size=np.asarray(raw_size, dtype=np.int32),
        resized_size=np.asarray(resized_size, dtype=np.int32),
        teacher_source=np.asarray("traceanything_official_trajectory_field", dtype=object),
        official_repo_path=np.asarray(str(repo), dtype=object),
        official_commit_hash=np.asarray(repo_commit or "", dtype=object),
        checkpoint_path=np.asarray(str(ckpt), dtype=object),
        native_visibility_available=np.asarray(False),
        visibility_source=np.asarray(extract["visibility_source"], dtype=object),
        trajectory_field_adapter=np.asarray(extract["trajectory_field_adapter"], dtype=object),
        no_future_box_projection=np.asarray(True),
        target_side_object_box_search_used=np.asarray(True),
        teacher_uses_full_obs_future_clip_as_target=np.asarray(True),
        stwm_input_restricted_to_observed=np.asarray(True),
        predecode_path=np.asarray(str(pre_path), dtype=object),
    )
    return out_path


def _locate_cotracker_cache(item_key: str, split: str, m: int, horizon: int) -> Path | None:
    filename = f"{item_key.replace('::', '__')}.npz"
    candidates = []
    if m == 128 and horizon == 8:
        candidates.append(V15C_ROOT / split / filename)
    candidates.append(V16_ROOT / f"M{m}_H{horizon}" / split / filename)
    for path in candidates:
        if path.exists():
            return path
    return None


def _compare_to_cotracker(trace_path: Path, cot_path: Path) -> dict[str, Any]:
    trace = np.load(trace_path, allow_pickle=True)
    cot = np.load(cot_path, allow_pickle=True)
    query_match = bool(
        trace["query_points_xy"].shape == cot["query_points_xy"].shape
        and np.allclose(trace["query_points_xy"], cot["query_points_xy"], atol=1e-3)
    )
    tracks_a = np.asarray(trace["tracks_xy"], dtype=np.float32)
    tracks_b = np.asarray(cot["tracks_xy"], dtype=np.float32)
    vis_a = np.asarray(trace["visibility"]).astype(bool)
    vis_b = np.asarray(cot["visibility"]).astype(bool)
    if tracks_a.shape != tracks_b.shape:
        return {
            "comparison_available": False,
            "reason": f"shape_mismatch trace={tracks_a.shape} cot={tracks_b.shape}",
            "query_points_match": query_match,
        }
    both_vis = np.logical_and(vis_a, vis_b)
    point_dist = np.linalg.norm(tracks_a - tracks_b, axis=-1)
    endpoint_dist = np.linalg.norm(tracks_a[:, :, -1] - tracks_b[:, :, -1], axis=-1)
    var_a = float(np.var(tracks_a, axis=2).mean())
    var_b = float(np.var(tracks_b, axis=2).mean())
    return {
        "comparison_available": True,
        "query_points_match": query_match,
        "point_consistency_l2_px": float(point_dist[both_vis].mean()) if np.any(both_vis) else float(point_dist.mean()),
        "endpoint_consistency_l2_px": float(endpoint_dist.mean()),
        "visibility_agreement": float((vis_a == vis_b).mean()),
        "traceanything_trajectory_variance": var_a,
        "cotracker_trajectory_variance": var_b,
        "traceanything_visible_ratio": float(vis_a.mean()) if vis_a.size else 0.0,
        "cotracker_visible_ratio": float(vis_b.mean()) if vis_b.size else 0.0,
    }


def _aggregate_comparisons(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [row["comparison_by_m"][key] for row in rows if key in row["comparison_by_m"] and row["comparison_by_m"][key].get("comparison_available")]
    if not vals:
        return {"matched_clip_count": 0}
    def mean(name: str) -> float | None:
        arr = [float(v[name]) for v in vals if v.get(name) is not None]
        return float(np.mean(arr)) if arr else None
    return {
        "matched_clip_count": len(vals),
        "mean_point_consistency_l2_px": mean("point_consistency_l2_px"),
        "mean_endpoint_consistency_l2_px": mean("endpoint_consistency_l2_px"),
        "mean_visibility_agreement": mean("visibility_agreement"),
        "mean_traceanything_trajectory_variance": mean("traceanything_trajectory_variance"),
        "mean_cotracker_trajectory_variance": mean("cotracker_trajectory_variance"),
    }


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", default="third_party/TraceAnything")
    parser.add_argument("--checkpoint", default="models/checkpoints/traceanything/traceanything_pretrained.pt")
    parser.add_argument("--max-train", type=int, default=20)
    parser.add_argument("--max-val", type=int, default=10)
    parser.add_argument("--max-test", type=int, default=10)
    parser.add_argument("--m-values", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allowed-splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--report-path", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--doc-path", default=str(DEFAULT_DOC_PATH))
    args = parser.parse_args()
    report_path = Path(args.report_path)
    doc_path = Path(args.doc_path)

    repo = ROOT / args.repo_path
    ckpt = ROOT / args.checkpoint
    start = time.time()
    if not repo.exists() or not ckpt.exists():
        payload = {
            "traceanything_teacher_runnable": False,
            "traceanything_object_tracks_valid": False,
            "official_repo_path": str(repo),
            "checkpoint_path": str(ckpt),
            "failure_reason": "traceanything_repo_or_checkpoint_missing",
            "recommended_next_step": "fix_traceanything_adapter",
        }
        dump_json(report_path, payload)
        write_doc(
            doc_path,
            "STWM TraceAnything Teacher Pilot V2",
            payload,
            ["traceanything_teacher_runnable", "failure_reason", "recommended_next_step"],
        )
        return 1

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, evaluate_bspline_conf, load_status = _load_traceanything_model(repo, ckpt, device)
    split_map = _mixed_split_map()
    selected = _select_items(args.max_train, args.max_val, args.max_test, split_map)
    target_quotas = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    success_by_split: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    repo_commit = _repo_commit(repo)

    for idx, pre_path in enumerate(selected):
        split = split_map.get(_norm_key(pre_path), pre_path.parent.name)
        if split not in set(args.allowed_splits):
            continue
        if success_by_split[split] >= target_quotas.get(split, 0):
            continue
        if all(success_by_split[s] >= q for s, q in target_quotas.items()):
            break
        item_key = _norm_key(pre_path)
        dataset = item_key.split("::", 1)[0]
        try:
            pre = np.load(pre_path, allow_pickle=True)
            anchor = Path(str(_scalar(pre["semantic_frame_path"])))
            frames, query_frame, frame_err = _frame_sequence(anchor, total=args.obs_len + args.horizon, preferred_query_frame=args.obs_len - 1)
            if frame_err or frames is None:
                failures.append({"item_key": item_key, "split": split, "dataset": dataset, "reason": frame_err})
                continue
            views, raw_size, resized_size, scale_xy = _load_views(frames, device, args.max_side)
            clip_start = time.time()
            with torch.no_grad():
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                preds = model.forward(views)
                peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            forward_seconds = float(time.time() - clip_start)
            clip_row = {
                "item_key": item_key,
                "split": split,
                "dataset": dataset,
                "query_frame": int(query_frame),
                "frame_count": len(frames),
                "raw_size": raw_size,
                "resized_size": resized_size,
                "forward_seconds": forward_seconds,
                "peak_gpu_memory_bytes": peak_mem,
                "cache_paths_by_m": {},
                "point_count_by_m": {},
                "valid_point_ratio_by_m": {},
                "same_trajectory_fraction_by_m": {},
                "comparison_by_m": {},
            }
            for m in args.m_values:
                seed = int(hashlib.md5(item_key.encode()).hexdigest()[:8], 16)
                q_points_raw, object_ids, semantic_ids, q_err = _query_points(pre, m, seed)
                if q_err:
                    clip_row["comparison_by_m"][f"M{m}"] = {"comparison_available": False, "reason": q_err}
                    continue
                extract = _extract_tracks_for_m(
                    pre,
                    preds,
                    evaluate_bspline_conf,
                    int(query_frame),
                    q_points_raw,
                    object_ids,
                    semantic_ids,
                    raw_size,
                    resized_size,
                    scale_xy,
                    args.obs_len,
                    args.horizon,
                )
                cache_path = _save_cache(
                    OUT_ROOT,
                    item_key=item_key,
                    split=split,
                    dataset=dataset,
                    frame_paths=frames,
                    query_frame=int(query_frame),
                    obs_len=args.obs_len,
                    horizon=args.horizon,
                    m=m,
                    raw_size=raw_size,
                    resized_size=resized_size,
                    repo=repo,
                    repo_commit=repo_commit,
                    ckpt=ckpt,
                    pre_path=pre_path,
                    extract=extract,
                )
                key = f"M{m}"
                clip_row["cache_paths_by_m"][key] = str(cache_path.relative_to(ROOT))
                clip_row["point_count_by_m"][key] = int(np.asarray(extract["point_id"]).size)
                clip_row["valid_point_ratio_by_m"][key] = float(extract["valid_point_ratio"])
                clip_row["same_trajectory_fraction_by_m"][key] = float(extract["same_trajectory_fraction"])
                cot_path = _locate_cotracker_cache(item_key, split, m, args.horizon)
                if cot_path is not None:
                    clip_row["comparison_by_m"][key] = _compare_to_cotracker(cache_path, cot_path)
                    clip_row["comparison_by_m"][key]["cotracker_cache_path"] = str(cot_path.relative_to(ROOT))
                else:
                    clip_row["comparison_by_m"][key] = {"comparison_available": False, "reason": "matching_cotracker_cache_missing"}
            rows.append(clip_row)
            success_by_split[split] += 1
        except Exception as exc:
            failures.append({"item_key": item_key, "split": split, "dataset": dataset, "reason": repr(exc)})
        print(f"[{idx+1}/{len(selected)}] ok={len(rows)} failed={len(failures)} {pre_path.name}", flush=True)

    comparison_summary = {f"M{m}": _aggregate_comparisons(rows, f"M{m}") for m in args.m_values}
    valid_ratios = [row["valid_point_ratio_by_m"].get(f"M{m}") for row in rows for m in args.m_values if row["valid_point_ratio_by_m"].get(f"M{m}") is not None]
    same_fracs = [row["same_trajectory_fraction_by_m"].get(f"M{m}") for row in rows for m in args.m_values if row["same_trajectory_fraction_by_m"].get(f"M{m}") is not None]
    payload = {
        "audit_name": "stwm_traceanything_teacher_pilot_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repo_path": str(repo),
        "official_commit_hash": repo_commit,
        "checkpoint_path": str(ckpt),
        "gpu_id": _gpu_id(),
        "device": str(device),
        "exact_command": " ".join(sys.argv),
        "model_load_status": load_status,
        "teacher_source": "traceanything_official_trajectory_field",
        "teacher_uses_full_obs_future_clip_as_target": True,
        "stwm_input_restricted_to_observed": True,
        "target_side_object_box_search_used": True,
        "native_visibility_available": False,
        "processed_clip_count": len(rows),
        "failed_clip_count": len(failures),
        "processed_split_counts": dict(Counter(row["split"] for row in rows)),
        "processed_dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "requested_split_counts": {"train": args.max_train, "val": args.max_val, "test": args.max_test},
        "m_values": list(args.m_values),
        "obs_len": args.obs_len,
        "horizon": args.horizon,
        "runtime_seconds": float(time.time() - start),
        "peak_gpu_memory_bytes_max": max([int(row["peak_gpu_memory_bytes"]) for row in rows], default=0),
        "mean_forward_seconds_per_clip": float(np.mean([row["forward_seconds"] for row in rows])) if rows else None,
        "mean_valid_point_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "mean_same_trajectory_fraction": float(np.mean(same_fracs)) if same_fracs else 1.0,
        "comparison_to_cotracker_same_clips": comparison_summary,
        "traceanything_teacher_runnable": bool(rows),
        "traceanything_object_tracks_valid": bool(rows and (np.mean(valid_ratios) if valid_ratios else 0.0) > 0.50 and (np.mean(same_fracs) if same_fracs else 1.0) < 0.95),
        "rows": rows,
        "failures": failures[:200],
        "recommended_next_step": "train_ostf_v2_on_traceanything_hardbench" if rows else "fix_traceanything_adapter",
    }
    dump_json(report_path, payload)
    write_doc(
        doc_path,
        "STWM TraceAnything Teacher Pilot V2",
        payload,
        [
            "traceanything_teacher_runnable",
            "traceanything_object_tracks_valid",
            "processed_clip_count",
            "failed_clip_count",
            "processed_split_counts",
            "processed_dataset_counts",
            "mean_valid_point_ratio",
            "mean_same_trajectory_fraction",
            "comparison_to_cotracker_same_clips",
            "recommended_next_step",
        ],
    )
    print(report_path.relative_to(ROOT) if report_path.is_relative_to(ROOT) else report_path)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
