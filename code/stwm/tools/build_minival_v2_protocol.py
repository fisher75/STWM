from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import numpy as np
from PIL import Image


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build week2 mini-val v2 hard protocol artifacts")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week1_mini.json")
    parser.add_argument("--output-manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2.json")
    parser.add_argument(
        "--output-val-ids",
        default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_week2_minival_v2_val_clip_ids.json",
    )
    parser.add_argument(
        "--output-report",
        default="/home/chen034/workspace/stwm/reports/week2_minival_v2_hard_selection.json",
    )
    parser.add_argument("--obs-steps", type=int, default=8)
    parser.add_argument("--pred-steps", type=int, default=8)
    parser.add_argument("--val-clips", type=int, default=18)
    parser.add_argument("--max-frames", type=int, default=64)
    return parser


def _read_labels(path: str) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def _compute_label_stats(mask_paths: list[str], label_id: int, obs_steps: int, pred_steps: int, max_frames: int) -> dict[str, Any]:
    horizon = min(len(mask_paths), max_frames)
    areas: list[float] = []
    vis: list[int] = []
    centroids: list[tuple[float, float] | None] = []

    for path in mask_paths[:horizon]:
        labels = _read_labels(path)
        fg = labels == int(label_id)
        area = float(fg.mean())
        areas.append(area)
        visible = 1 if area > 1e-6 else 0
        vis.append(visible)
        if visible:
            ys, xs = np.nonzero(fg)
            h, w = labels.shape[:2]
            cx = float(xs.mean() / max(1, w - 1))
            cy = float(ys.mean() / max(1, h - 1))
            centroids.append((cx, cy))
        else:
            centroids.append(None)

    if not areas:
        return {
            "label_id": int(label_id),
            "valid": False,
            "score": -1e9,
        }

    start = min(obs_steps, len(vis))
    end = min(len(vis), obs_steps + pred_steps)
    future_vis = vis[start:end]

    reappear = 0
    disappear = 0
    for i in range(1, len(future_vis)):
        if future_vis[i - 1] == 0 and future_vis[i] == 1:
            reappear += 1
        if future_vis[i - 1] == 1 and future_vis[i] == 0:
            disappear += 1

    motions = []
    for i in range(1, len(centroids)):
        if centroids[i - 1] is None or centroids[i] is None:
            continue
        dx = centroids[i][0] - centroids[i - 1][0]
        dy = centroids[i][1] - centroids[i - 1][1]
        motions.append(float((dx * dx + dy * dy) ** 0.5))

    area_mean = float(np.mean(areas))
    area_std = float(np.std(areas))
    area_cv = float(area_std / (area_mean + 1e-8))
    motion_mean = float(np.mean(motions)) if motions else 0.0
    present_ratio = float(np.mean(vis))

    # Favor labels that are visible in observation but challenging in prediction.
    obs_present = float(np.mean(vis[:start])) if start > 0 else 0.0
    future_present = float(np.mean(future_vis)) if future_vis else 0.0

    score = (
        4.0 * float(reappear)
        + 2.0 * float(disappear)
        + 3.0 * motion_mean
        + 1.5 * area_cv
        + 1.5 * abs(obs_present - future_present)
    )

    # Exclude near-empty or overwhelming labels.
    valid = area_mean > 5e-4 and area_mean < 0.5 and obs_present > 0.2
    if not valid:
        score -= 100.0

    return {
        "label_id": int(label_id),
        "valid": bool(valid),
        "score": float(score),
        "reappear_events": int(reappear),
        "disappear_events": int(disappear),
        "motion_mean": float(motion_mean),
        "area_mean": float(area_mean),
        "area_cv": float(area_cv),
        "obs_present_ratio": float(obs_present),
        "future_present_ratio": float(future_present),
        "present_ratio": float(present_ratio),
    }


def _choose_target_label(mask_paths: list[str], obs_steps: int, pred_steps: int, max_frames: int) -> dict[str, Any]:
    horizon = min(len(mask_paths), max_frames)
    label_ids: set[int] = set()
    frame_multi_label_counts: list[int] = []
    for path in mask_paths[:horizon]:
        labels = _read_labels(path)
        ids = [int(x) for x in np.unique(labels) if int(x) != 0]
        label_ids.update(ids)
        frame_multi_label_counts.append(len(ids))

    if not label_ids:
        return {
            "target_label_id": None,
            "target_score": -1e9,
            "target_stats": {},
            "multi_instance_density": 0.0,
            "num_candidate_labels": 0,
        }

    stats = [_compute_label_stats(mask_paths, lid, obs_steps, pred_steps, max_frames) for lid in sorted(label_ids)]
    best = max(stats, key=lambda item: item.get("score", -1e9))

    return {
        "target_label_id": int(best["label_id"]) if best.get("valid", False) else None,
        "target_score": float(best.get("score", -1e9)),
        "target_stats": best,
        "multi_instance_density": float(np.mean([c >= 2 for c in frame_multi_label_counts])) if frame_multi_label_counts else 0.0,
        "num_candidate_labels": len(label_ids),
        "label_stats": stats,
    }


def main() -> None:
    args = build_parser().parse_args()

    manifest_path = Path(args.manifest)
    items = json.loads(manifest_path.read_text())

    needed = int(args.obs_steps) + int(args.pred_steps)
    enriched: list[dict[str, Any]] = []
    vspw_rank: list[dict[str, Any]] = []

    for item in items:
        clip_id = str(item.get("clip_id", ""))
        metadata = dict(item.get("metadata", {}))
        dataset = str(metadata.get("dataset", "")).lower()
        mask_paths = metadata.get("mask_paths", []) if isinstance(metadata.get("mask_paths"), list) else []
        frame_paths = item.get("frame_paths", []) if isinstance(item.get("frame_paths"), list) else []

        if dataset == "vspw" and len(frame_paths) >= needed and len(mask_paths) >= needed:
            target_info = _choose_target_label(
                mask_paths,
                obs_steps=int(args.obs_steps),
                pred_steps=int(args.pred_steps),
                max_frames=int(args.max_frames),
            )
            metadata["target_label_id"] = target_info["target_label_id"]
            metadata["minival_v2"] = {
                "target_score": target_info["target_score"],
                "multi_instance_density": target_info["multi_instance_density"],
                "num_candidate_labels": target_info["num_candidate_labels"],
                "target_stats": target_info["target_stats"],
            }
            vspw_rank.append(
                {
                    "clip_id": clip_id,
                    "target_label_id": target_info["target_label_id"],
                    "target_score": float(target_info["target_score"]),
                    "multi_instance_density": float(target_info["multi_instance_density"]),
                    "num_candidate_labels": int(target_info["num_candidate_labels"]),
                    "target_stats": target_info["target_stats"],
                }
            )

        enriched_item = dict(item)
        enriched_item["metadata"] = metadata
        enriched.append(enriched_item)

    vspw_rank = sorted(vspw_rank, key=lambda x: (x["target_score"], x["multi_instance_density"]), reverse=True)
    selected = [x for x in vspw_rank if x.get("target_label_id") is not None]
    selected = selected[: max(1, int(args.val_clips))]
    val_clip_ids = [x["clip_id"] for x in selected]

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(enriched, indent=2))

    output_val_ids = Path(args.output_val_ids)
    output_val_ids.parent.mkdir(parents=True, exist_ok=True)
    output_val_ids.write_text(json.dumps(val_clip_ids, indent=2))

    report = {
        "source_manifest": str(manifest_path),
        "output_manifest": str(output_manifest),
        "output_val_ids": str(output_val_ids),
        "obs_steps": int(args.obs_steps),
        "pred_steps": int(args.pred_steps),
        "val_clips": int(args.val_clips),
        "selected_count": len(val_clip_ids),
        "selected": selected,
        "top_ranked": vspw_rank[: max(20, int(args.val_clips))],
    }
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2))

    print(json.dumps({
        "selected_count": len(val_clip_ids),
        "output_manifest": str(output_manifest),
        "output_val_ids": str(output_val_ids),
        "output_report": str(output_report),
    }, indent=2))


if __name__ == "__main__":
    main()
