#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import cKDTree

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.run_cotracker_object_dense_teacher_v15c_20260502 import (
    PREDECODE_ROOT,
    _frame_sequence,
    _norm_key,
    _query_points,
)
from stwm.tools.run_traceanything_object_trajectory_teacher_v2_20260502 import (
    _aggregate_comparisons,
    _apply_process_title,
    _compare_to_cotracker,
    _gpu_id,
    _load_traceanything_model,
    _load_views,
    _locate_cotracker_cache,
    _repo_commit,
    _scalar,
)


DEFAULT_BENCHMARK_PATH = ROOT / "reports/stwm_ostf_hard_benchmark_v2_20260502.json"
DEFAULT_OUT_ROOT = ROOT / "outputs/cache/stwm_traceanything_hardbench_v25"


def _predecode_index() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(PREDECODE_ROOT.glob("*/*.npz")):
        out[_norm_key(path)] = path
    return out


def _json_load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_reason_tags(tags: list[str]) -> int:
    score = 0
    tags = set(tags)
    if "top20_cv_hard" in tags:
        score += 100
    if "top30_cv_hard" in tags:
        score += 50
    if "occlusion_hard" in tags:
        score += 30
    if "nonlinear_hard" in tags:
        score += 20
    if "interaction_hard" in tags:
        score += 10
    return score


def _balanced_select(items: list[dict[str, Any]], max_clips: int) -> list[dict[str, Any]]:
    ds_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        ds_buckets[row["dataset"]].append(row)
    for rows in ds_buckets.values():
        rows.sort(key=lambda r: (-int(r["score"]), r["item_key"]))
    selected: list[dict[str, Any]] = []
    datasets = sorted(ds_buckets.keys())
    while len(selected) < max_clips and any(ds_buckets.values()):
        progressed = False
        for ds in datasets:
            rows = ds_buckets[ds]
            if rows and len(selected) < max_clips:
                selected.append(rows.pop(0))
                progressed = True
        if not progressed:
            break
    return selected


def _allowed_item_keys(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    payload = _json_load(path)
    return {str(x) for x in payload.get("item_keys", [])}


def _candidate_feasibility_reason(raw_frame_path: str | None, *, obs_len: int, horizon: int) -> str | None:
    if not raw_frame_path:
        return "raw_frame_path_missing"
    frames, _, err = _frame_sequence(Path(raw_frame_path), total=obs_len + horizon, preferred_query_frame=obs_len - 1)
    if err or frames is None:
        return err or "frame_sequence_unavailable"
    return None


def _select_candidates(
    benchmark: dict[str, Any],
    predecode_index: dict[str, Path],
    *,
    obs_len: int,
    selection_horizon: int,
    max_clips: int,
    allowed_item_keys: set[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_item = benchmark.get("per_item", {})
    candidates: list[dict[str, Any]] = []
    blocker_counts: Counter[str] = Counter()
    dataset_blocked: Counter[str] = Counter()
    for item_key, row in per_item.items():
        if allowed_item_keys is not None and item_key not in allowed_item_keys:
            blocker_counts["not_in_allowed_item_keys"] += 1
            continue
        horizon_key = f"H{selection_horizon}"
        horizon_status = row.get("horizon_status", {}).get(horizon_key, {})
        if not horizon_status.get("feasible", False):
            blocker_counts[f"{horizon_key}_benchmark_infeasible"] += 1
            continue
        if not row.get("raw_frame_available", False):
            blocker_counts["raw_frame_unavailable"] += 1
            continue
        if not row.get("semantic_instance_available", False):
            blocker_counts["semantic_instance_unavailable"] += 1
            continue
        if item_key not in predecode_index:
            blocker_counts["predecode_missing"] += 1
            continue
        tags = list(row.get("reason_tags", []))
        if not tags:
            blocker_counts["reason_tags_missing"] += 1
            continue
        frame_reason = _candidate_feasibility_reason(
            row.get("raw_frame_path"),
            obs_len=obs_len,
            horizon=selection_horizon,
        )
        if frame_reason is not None:
            blocker_counts[frame_reason] += 1
            dataset_blocked[str(row.get("dataset", "unknown"))] += 1
            continue
        candidates.append(
            {
                "item_key": item_key,
                "dataset": row["dataset"],
                "split": row["split"],
                "reason_tags": tags,
                "score": _score_reason_tags(tags),
                "predecode_path": predecode_index[item_key],
                "raw_frame_path": row.get("raw_frame_path"),
            }
        )
    selected = _balanced_select(candidates, max_clips)
    stats = {
        "candidate_count_after_actual_frame_feasibility": len(candidates),
        "selected_candidate_count": len(selected),
        "candidate_blocker_counts": dict(blocker_counts),
        "dataset_blocked_by_actual_frames": dict(dataset_blocked),
        "per_dataset_candidate_counts": dict(Counter(row["dataset"] for row in candidates)),
        "per_split_candidate_counts": dict(Counter(row["split"] for row in candidates)),
    }
    return selected, stats


def _target_cache_path(out_root: Path, *, item_key: str, split: str, m: int, horizon: int) -> Path:
    return out_root / f"M{m}_H{horizon}" / split / f"{item_key.replace('::', '__')}.npz"


def _resume_cache_ok(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        z = np.load(path, allow_pickle=True)
        return (
            "tracks_xy" in z.files
            and "visibility" in z.files
            and "point_id" in z.files
            and str(_scalar(z["teacher_source"])) == "traceanything_official_trajectory_field"
        )
    except Exception:
        return False


def _save_cache_v25(
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
    target_side_box_search_used: bool,
) -> Path:
    out_path = _target_cache_path(cache_dir, item_key=item_key, split=split, m=m, horizon=horizon)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        trajectory_field_query_confidence=extract["trajectory_field_query_confidence"].astype(np.float32),
        same_trajectory_fraction=np.asarray(float(extract["same_trajectory_fraction"]), dtype=np.float32),
        valid_point_ratio=np.asarray(float(extract["valid_point_ratio"]), dtype=np.float32),
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
        target_side_object_box_search_used=np.asarray(bool(target_side_box_search_used)),
        teacher_target_uses_full_clip=np.asarray(True),
        model_input_observed_only=np.asarray(True),
        predecode_path=np.asarray(str(pre_path), dtype=object),
    )
    return out_path


def _is_oom(exc: BaseException) -> bool:
    msg = repr(exc).lower()
    return "out of memory" in msg or "cudaerrormemoryallocation" in msg


def _extract_tracks_for_m_mode(
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
    *,
    use_target_box: bool,
) -> dict[str, Any]:
    total_steps = obs_len + horizon
    device = preds[0]["ctrl_pts3d"].device
    sx, sy = scale_xy
    width, height = resized_size
    boxes = (
        np.asarray(pre["entity_boxes_over_time"], dtype=np.float32)
        if "entity_boxes_over_time" in pre.files
        else np.zeros((total_steps, q_points_raw.shape[0], 4), dtype=np.float32)
    )
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
    same_trajectory_fraction_per_object: list[float] = []

    for obj_idx in range(object_count):
        source_obj = int(object_ids[obj_idx]) if obj_idx < len(object_ids) else obj_idx
        obj_valid = np.ones((total_steps,), dtype=bool)
        if "obs_valid" in pre.files and "fut_valid" in pre.files:
            obs = np.asarray(pre["obs_valid"]).astype(bool)
            fut = np.asarray(pre["fut_valid"]).astype(bool)
            if obs.ndim == 2 and fut.ndim == 2 and source_obj < obs.shape[1] and source_obj < fut.shape[1]:
                known_valid = np.concatenate([obs[:, source_obj], fut[:, source_obj]], axis=0)
                obj_valid[: min(total_steps, known_valid.shape[0])] = known_valid[: min(total_steps, known_valid.shape[0])]
        prev_xy = np.stack([qx[obj_idx], qy[obj_idx]], axis=-1).astype(np.float32)
        for t in range(total_steps):
            q3d = query_pts3d_t[t, qy[obj_idx], qx[obj_idx], :]
            field = target_maps[t]
            conf_field = target_confs[t]
            prev_min = np.floor(prev_xy.min(axis=0)).astype(np.int64)
            prev_max = np.ceil(prev_xy.max(axis=0)).astype(np.int64)
            obj_span = np.maximum(prev_max - prev_min, 1)
            margin = int(max(16, min(64, np.max(obj_span) * 0.25 + 8)))
            wx0 = max(0, int(prev_min[0]) - margin)
            wy0 = max(0, int(prev_min[1]) - margin)
            wx1 = min(width, int(prev_max[0]) + margin + 1)
            wy1 = min(height, int(prev_max[1]) + margin + 1)

            if use_target_box and t < boxes.shape[0] and source_obj < boxes.shape[1]:
                x0, y0, x1, y1 = [float(v) for v in boxes[t, source_obj].tolist()]
                if np.isfinite([x0, y0, x1, y1]).all() and x1 > x0 and y1 > y0:
                    bx0 = max(0, min(width - 1, int(np.floor(x0 * sx))))
                    by0 = max(0, min(height - 1, int(np.floor(y0 * sy))))
                    bx1 = max(bx0 + 1, min(width, int(np.ceil(x1 * sx))))
                    by1 = max(by0 + 1, min(height, int(np.ceil(y1 * sy))))
                    xi0 = max(0, min(wx0, bx0))
                    yi0 = max(0, min(wy0, by0))
                    xi1 = min(width, max(wx1, bx1))
                    yi1 = min(height, max(wy1, by1))
                else:
                    xi0, yi0, xi1, yi1 = wx0, wy0, wx1, wy1
            else:
                xi0, yi0, xi1, yi1 = wx0, wy0, wx1, wy1

            crop_pts = field[yi0:yi1, xi0:xi1].reshape(-1, 3)
            crop_conf = conf_field[yi0:yi1, xi0:xi1].reshape(-1)
            crop_w = xi1 - xi0
            if crop_pts.size == 0 or crop_w <= 0:
                crop_pts = field.reshape(-1, 3)
                crop_conf = conf_field.reshape(-1)
                crop_w = width
                xi0, yi0 = 0, 0
            tree = cKDTree(crop_pts)
            min_d, arg = tree.query(q3d, k=1, workers=-1)
            min_d = np.asarray(min_d, dtype=np.float32)
            arg = np.asarray(arg, dtype=np.int64)
            yy, xx = np.divmod(arg, crop_w)
            xy = np.stack([xi0 + xx, yi0 + yy], axis=-1).astype(np.float32)
            conf = crop_conf[arg].astype(np.float32)
            conf_thr = float(np.quantile(crop_conf, 0.25)) if crop_conf.size else 0.0
            consistency_thr = float(np.quantile(min_d, 0.75)) if min_d.size else float("inf")
            vis = np.logical_and(obj_valid[t], np.logical_and(conf >= conf_thr, min_d <= consistency_thr))
            tracks_xy[obj_idx, :, t, 0] = xy[:, 0] / max(sx, 1e-6)
            tracks_xy[obj_idx, :, t, 1] = xy[:, 1] / max(sy, 1e-6)
            visibility[obj_idx, :, t] = vis
            confidence[obj_idx, :, t] = conf
            nn_distance[obj_idx, :, t] = min_d
            prev_xy = xy
        unique_tracks = np.unique(np.round(tracks_xy[obj_idx].reshape(m, -1), decimals=3), axis=0).shape[0]
        same_trajectory_fraction_per_object.append(1.0 - float(unique_tracks) / max(1, m))

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
        "same_trajectory_fraction": float(np.mean(same_trajectory_fraction_per_object)) if same_trajectory_fraction_per_object else 1.0,
        "valid_point_ratio": float(visibility.mean()) if visibility.size else 0.0,
        "native_visibility_available": False,
        "visibility_source": "estimated_from_traceanything_confidence_and_trajectory_consistency_teacher_only",
        "trajectory_field_adapter": (
            "query_frame_bspline_3d_to_target_frame_nearest_pixel_within_target_object_box"
            if use_target_box
            else "query_frame_bspline_3d_to_target_frame_nearest_pixel_global"
        ),
    }


def _compare_extraction_modes(primary: dict[str, Any], alt: dict[str, Any]) -> dict[str, Any]:
    tracks_a = np.asarray(primary["tracks_xy"], dtype=np.float32)
    tracks_b = np.asarray(alt["tracks_xy"], dtype=np.float32)
    vis_a = np.asarray(primary["visibility"]).astype(bool)
    vis_b = np.asarray(alt["visibility"]).astype(bool)
    both = np.logical_and(vis_a, vis_b)
    point_dist = np.linalg.norm(tracks_a - tracks_b, axis=-1)
    return {
        "mean_point_disagreement_px": float(point_dist[both].mean()) if np.any(both) else float(point_dist.mean()),
        "visibility_agreement": float((vis_a == vis_b).mean()) if vis_a.size else 0.0,
        "primary_valid_point_ratio": float(primary["valid_point_ratio"]),
        "alt_valid_point_ratio": float(alt["valid_point_ratio"]),
        "primary_same_trajectory_fraction": float(primary["same_trajectory_fraction"]),
        "alt_same_trajectory_fraction": float(alt["same_trajectory_fraction"]),
    }


def _load_views_and_forward(
    model: torch.nn.Module,
    frames: list[Path],
    device: torch.device,
    *,
    max_side: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[int, int], tuple[int, int], tuple[float, float], int]:
    views, raw_size, resized_size, scale_xy = _load_views(frames, device, max_side)
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        preds = model.forward(views)
        peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    return views, preds, raw_size, resized_size, scale_xy, peak_mem


def main() -> int:
    _apply_process_title()
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-report", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--repo-path", default="third_party/TraceAnything")
    parser.add_argument("--checkpoint", default="models/checkpoints/traceanything/traceanything_pretrained.pt")
    parser.add_argument("--obs-len", type=int, default=8)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--selection-horizon", type=int, default=None)
    parser.add_argument("--max-clips", type=int, default=300)
    parser.add_argument("--m-values", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--fallback-max-side", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--allowed-item-keys-json", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-path", required=True)
    parser.add_argument("--doc-path", required=True)
    args = parser.parse_args()

    report_path = Path(args.report_path)
    doc_path = Path(args.doc_path)
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = (ROOT / out_root).resolve()
    selection_horizon = int(args.selection_horizon or args.horizon)
    start = time.time()
    repo = ROOT / args.repo_path
    ckpt = ROOT / args.checkpoint
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    if not repo.exists() or not ckpt.exists():
        payload = {
            "audit_name": "stwm_traceanything_hardbench_cache_v25_shard",
            "traceanything_teacher_runnable": False,
            "failure_reason": "traceanything_repo_or_checkpoint_missing",
            "processed_clip_count": 0,
            "failed_clip_count": 0,
            "shard_terminal_status": "failed",
        }
        dump_json(report_path, payload)
        write_doc(doc_path, "STWM TraceAnything Hardbench Cache V25 Shard", payload, list(payload.keys()))
        return 1

    benchmark = _json_load(Path(args.benchmark_report))
    predecode_index = _predecode_index()
    allowed_item_keys = _allowed_item_keys(Path(args.allowed_item_keys_json)) if args.allowed_item_keys_json else None
    selected, selection_stats = _select_candidates(
        benchmark,
        predecode_index,
        obs_len=args.obs_len,
        selection_horizon=selection_horizon,
        max_clips=args.max_clips,
        allowed_item_keys=allowed_item_keys,
    )
    sharded = selected[args.shard_index :: args.num_shards]

    model, evaluate_bspline_conf, load_status = _load_traceanything_model(repo, ckpt, device)
    repo_commit = _repo_commit(repo)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    retry_count = 0
    for idx, meta in enumerate(sharded):
        item_key = meta["item_key"]
        split = meta["split"]
        dataset = meta["dataset"]
        pre_path = meta["predecode_path"]
        wanted_paths = [
            _target_cache_path(out_root, item_key=item_key, split=split, m=m, horizon=args.horizon)
            for m in args.m_values
        ]
        if args.resume and all(_resume_cache_ok(path) for path in wanted_paths):
            skipped.append(
                {
                    "item_key": item_key,
                    "split": split,
                    "dataset": dataset,
                    "reason": "resume_existing_successful_cache",
                    "cache_paths_by_m": {
                        f"M{m}": str(path.relative_to(ROOT))
                        for m, path in zip(args.m_values, wanted_paths)
                    },
                }
            )
            print(
                f"[H{args.horizon} shard {args.shard_index}] {idx+1}/{len(sharded)} "
                f"ok={len(rows)} skipped={len(skipped)} failed={len(failures)} {item_key} resume",
                flush=True,
            )
            continue
        try:
            pre = np.load(pre_path, allow_pickle=True)
            anchor = Path(str(_scalar(pre["semantic_frame_path"])))
            frames, query_frame, frame_err = _frame_sequence(anchor, total=args.obs_len + args.horizon, preferred_query_frame=args.obs_len - 1)
            if frame_err or frames is None:
                failures.append({"item_key": item_key, "split": split, "dataset": dataset, "reason": frame_err})
                continue

            clip_start = time.time()
            max_side_used = int(args.max_side)
            try:
                _, preds, raw_size, resized_size, scale_xy, peak_mem = _load_views_and_forward(
                    model,
                    frames,
                    device,
                    max_side=args.max_side,
                )
            except Exception as exc:
                if not _is_oom(exc) or int(args.fallback_max_side) >= int(args.max_side):
                    raise
                retry_count += 1
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                max_side_used = int(args.fallback_max_side)
                _, preds, raw_size, resized_size, scale_xy, peak_mem = _load_views_and_forward(
                    model,
                    frames,
                    device,
                    max_side=args.fallback_max_side,
                )
            clip_row = {
                "item_key": item_key,
                "split": split,
                "dataset": dataset,
                "reason_tags": meta["reason_tags"],
                "query_frame": int(query_frame),
                "frame_count": len(frames),
                "forward_seconds": float(time.time() - clip_start),
                "peak_gpu_memory_bytes": peak_mem,
                "max_side_used": max_side_used,
                "cache_paths_by_m": {},
                "point_count_by_m": {},
                "valid_point_ratio_by_m": {},
                "same_trajectory_fraction_by_m": {},
                "visibility_coverage_by_m": {},
                "alternative_extraction_comparison_by_m": {},
                "comparison_by_m": {},
                "comparison_to_cotracker_by_m": {},
                "cache_checksums_by_m": {},
                "cache_sizes_by_m": {},
            }
            for m in args.m_values:
                seed = int(hashlib.md5(item_key.encode()).hexdigest()[:8], 16)
                q_points_raw, object_ids, semantic_ids, q_err = _query_points(pre, m, seed)
                key = f"M{m}"
                if q_err:
                    clip_row["comparison_to_cotracker_by_m"][key] = {"comparison_available": False, "reason": q_err}
                    clip_row["comparison_by_m"][key] = clip_row["comparison_to_cotracker_by_m"][key]
                    continue
                extract_box = _extract_tracks_for_m_mode(
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
                    use_target_box=True,
                )
                extract_global = _extract_tracks_for_m_mode(
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
                    use_target_box=False,
                )
                cache_path = _save_cache_v25(
                    out_root,
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
                    extract=extract_box,
                    target_side_box_search_used=True,
                )
                clip_row["cache_paths_by_m"][key] = str(cache_path.relative_to(ROOT))
                clip_row["point_count_by_m"][key] = int(np.asarray(extract_box["point_id"]).size)
                clip_row["valid_point_ratio_by_m"][key] = float(extract_box["valid_point_ratio"])
                clip_row["same_trajectory_fraction_by_m"][key] = float(extract_box["same_trajectory_fraction"])
                clip_row["visibility_coverage_by_m"][key] = float(np.asarray(extract_box["visibility"]).mean()) if np.asarray(extract_box["visibility"]).size else 0.0
                clip_row["alternative_extraction_comparison_by_m"][key] = _compare_extraction_modes(extract_box, extract_global)
                cot_path = _locate_cotracker_cache(item_key, split, m, 16)
                if cot_path is not None:
                    clip_row["comparison_to_cotracker_by_m"][key] = _compare_to_cotracker(cache_path, cot_path)
                    clip_row["comparison_to_cotracker_by_m"][key]["cotracker_cache_path"] = str(cot_path.relative_to(ROOT))
                else:
                    clip_row["comparison_to_cotracker_by_m"][key] = {"comparison_available": False, "reason": "matching_cotracker_cache_missing_for_H16_prefix_compare"}
                clip_row["comparison_by_m"][key] = clip_row["comparison_to_cotracker_by_m"][key]
                stat = cache_path.stat()
                clip_row["cache_sizes_by_m"][key] = int(stat.st_size)
                clip_row["cache_checksums_by_m"][key] = hashlib.md5(cache_path.read_bytes()).hexdigest()
            rows.append(clip_row)
        except Exception as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            reason = repr(exc)
            failures.append({"item_key": item_key, "split": split, "dataset": dataset, "reason": reason})
            print(f"[H{args.horizon} shard {args.shard_index}] failure {item_key}: {reason}", flush=True)
        print(
            f"[H{args.horizon} shard {args.shard_index}] {idx+1}/{len(sharded)} "
            f"ok={len(rows)} skipped={len(skipped)} failed={len(failures)} {item_key}",
            flush=True,
        )

    comparison_summary = {f"M{m}": _aggregate_comparisons(rows, f"M{m}") for m in args.m_values}
    point_counts = [int(v) for row in rows for v in row["point_count_by_m"].values()]
    valid_ratios = [float(v) for row in rows for v in row["valid_point_ratio_by_m"].values()]
    same_fracs = [float(v) for row in rows for v in row["same_trajectory_fraction_by_m"].values()]
    vis_cov = [float(v) for row in rows for v in row["visibility_coverage_by_m"].values()]
    subset_counts = Counter(tag for row in rows for tag in row["reason_tags"])
    failure_reasons = Counter(f["reason"] for f in failures)
    if rows or skipped:
        terminal_status = "completed"
    elif failures:
        terminal_status = "failed"
    else:
        terminal_status = "skipped_with_reason"
    payload = {
        "audit_name": "stwm_traceanything_hardbench_cache_v25_shard",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_repo_path": str(repo),
        "official_commit_hash": repo_commit,
        "checkpoint_path": str(ckpt),
        "gpu_id": _gpu_id(),
        "device": str(device),
        "model_load_status": load_status,
        "teacher_source": "traceanything_official_trajectory_field",
        "teacher_target_uses_full_clip": True,
        "teacher_target_uses_full_clip_for_target_extraction_only": True,
        "model_input_observed_only": True,
        "target_side_object_box_search_used_teacher_only": True,
        "native_visibility_available": False,
        "estimated_visibility_quality_audit": "confidence_plus_trajectory_consistency_teacher_only",
        "horizon": int(args.horizon),
        "selection_horizon": int(selection_horizon),
        "m_values": list(args.m_values),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "selected_clip_target_count": int(args.max_clips),
        "selected_clip_count_total": int(selection_stats["selected_candidate_count"]),
        "selected_clip_count_for_this_shard": len(sharded),
        "selection_stats": selection_stats,
        "processed_clip_count": len(rows),
        "skipped_clip_count": len(skipped),
        "failed_clip_count": len(failures),
        "processed_split_counts": dict(Counter(row["split"] for row in rows)),
        "processed_dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "per_subset_counts": dict(subset_counts),
        "point_count": int(sum(point_counts)),
        "mean_valid_point_ratio": float(np.mean(valid_ratios)) if valid_ratios else 0.0,
        "mean_same_trajectory_fraction": float(np.mean(same_fracs)) if same_fracs else 1.0,
        "mean_visibility_coverage": float(np.mean(vis_cov)) if vis_cov else 0.0,
        "comparison_to_cotracker_same_clips": comparison_summary,
        "failed_clip_reasons": dict(failure_reasons),
        "rows": rows,
        "skipped": skipped[:200],
        "failures": failures[:200],
        "runtime_seconds": float(time.time() - start),
        "peak_gpu_memory_bytes_max": max([int(row["peak_gpu_memory_bytes"]) for row in rows], default=0),
        "retry_count": retry_count,
        "traceanything_teacher_runnable": bool(rows or skipped),
        "traceanything_object_tracks_valid": bool(rows and (np.mean(valid_ratios) if valid_ratios else 0.0) > 0.40 and (np.mean(same_fracs) if same_fracs else 1.0) < 0.95),
        "shard_terminal_status": terminal_status,
    }
    dump_json(report_path, payload)
    write_doc(
        doc_path,
        "STWM TraceAnything Hardbench Cache V25 Shard",
        payload,
        [
            "horizon",
            "selected_clip_count_total",
            "selected_clip_count_for_this_shard",
            "processed_clip_count",
            "skipped_clip_count",
            "failed_clip_count",
            "processed_split_counts",
            "processed_dataset_counts",
            "per_subset_counts",
            "point_count",
            "mean_valid_point_ratio",
            "mean_same_trajectory_fraction",
            "mean_visibility_coverage",
            "retry_count",
            "failed_clip_reasons",
            "shard_terminal_status",
        ],
    )
    try:
        print(report_path.relative_to(ROOT))
    except ValueError:
        print(report_path)
    return 0 if rows or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
