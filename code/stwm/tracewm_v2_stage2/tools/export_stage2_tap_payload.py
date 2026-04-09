#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict
import json

import numpy as np


def _read_npz(path: str | Path) -> Dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"npz not found: {p}")
    arr = np.load(p, allow_pickle=False)
    return {str(k): arr[k] for k in arr.files}


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def export_proxy_payload_to_tapvid(
    *,
    proxy_payload_npz: str | Path,
    output_npz: str | Path,
    output_report_json: str | Path | None = None,
    raster_resolution: int = 256,
    pred_occlusion_mode: str = "all_visible",
    query_time_index: int = 0,
) -> Dict[str, Any]:
    raw = _read_npz(proxy_payload_npz)

    required = {
        "predicted_tracks_2d",
        "gt_tracks_2d",
        "visibility_mask",
        "query_points_2d",
    }
    missing = sorted(required - set(raw.keys()))
    if missing:
        raise RuntimeError(f"proxy payload missing required arrays: {missing}")

    pred_tracks_2d = np.asarray(raw["predicted_tracks_2d"], dtype=np.float32)
    gt_tracks_2d = np.asarray(raw["gt_tracks_2d"], dtype=np.float32)
    visibility_mask = np.asarray(raw["visibility_mask"], dtype=np.bool_)
    query_points_2d = np.asarray(raw["query_points_2d"], dtype=np.float32)

    if pred_tracks_2d.shape != gt_tracks_2d.shape:
        raise RuntimeError("predicted_tracks_2d and gt_tracks_2d must have the same shape")
    if pred_tracks_2d.ndim != 4 or pred_tracks_2d.shape[-1] != 2:
        raise RuntimeError(f"expected [batch, time, token, 2], got {list(pred_tracks_2d.shape)}")
    if visibility_mask.shape != pred_tracks_2d.shape[:-1]:
        raise RuntimeError("visibility_mask shape must match track tensors without the last coord dim")
    if query_points_2d.shape != pred_tracks_2d[:, 0].shape:
        raise RuntimeError("query_points_2d must match [batch, token, 2]")

    pred_tracks = np.transpose(pred_tracks_2d, (0, 2, 1, 3)).astype(np.float32)
    gt_tracks = np.transpose(gt_tracks_2d, (0, 2, 1, 3)).astype(np.float32)
    gt_occluded = np.logical_not(np.transpose(visibility_mask, (0, 2, 1)))

    scale = float(raster_resolution)
    pred_tracks[..., 0] *= scale
    pred_tracks[..., 1] *= scale
    gt_tracks[..., 0] *= scale
    gt_tracks[..., 1] *= scale

    if pred_occlusion_mode == "all_visible":
        pred_occluded = np.zeros_like(gt_occluded, dtype=np.bool_)
    else:
        raise RuntimeError(f"unsupported pred_occlusion_mode: {pred_occlusion_mode}")

    query_points = np.stack(
        [
            np.full(query_points_2d.shape[:2], float(query_time_index), dtype=np.float32),
            query_points_2d[..., 1] * scale,
            query_points_2d[..., 0] * scale,
        ],
        axis=-1,
    ).astype(np.float32)

    out_npz = Path(output_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        query_points=query_points,
        gt_occluded=gt_occluded.astype(np.bool_),
        gt_tracks=gt_tracks.astype(np.float32),
        pred_occluded=pred_occluded.astype(np.bool_),
        pred_tracks=pred_tracks.astype(np.float32),
    )

    report = {
        "adapter_version": "stage2_tapvid_adapter_v1",
        "source_proxy_payload_npz": str(Path(proxy_payload_npz)),
        "output_tap_payload_npz": str(out_npz),
        "coord_space_input": "normalized_stage2_xy",
        "coord_space_output": f"raster_xy_{int(raster_resolution)}",
        "query_mode_expected_for_eval": "first",
        "query_time_index": int(query_time_index),
        "pred_occlusion_mode": str(pred_occlusion_mode),
        "predicted_visibility_is_model_output": False,
        "benchmark_native_full_tap_episode": False,
        "query_time_matches_official_task": False,
        "adapter_limitations": [
            "stage2 current rollout exports future-only 2D tracks rather than full official TAP-Vid benchmark episodes",
            "query point is anchored to the first predicted frame because the frozen stage2 bridge does not export benchmark-native TAP query times",
            "pred_occluded is synthesized as all-visible because current frozen stage2 mainline does not expose an occlusion prediction head",
        ],
        "shapes": {
            "query_points": list(query_points.shape),
            "gt_occluded": list(gt_occluded.shape),
            "gt_tracks": list(gt_tracks.shape),
            "pred_occluded": list(pred_occluded.shape),
            "pred_tracks": list(pred_tracks.shape),
        },
        "counts": {
            "batch": int(pred_tracks.shape[0]),
            "queries_per_sample": int(pred_tracks.shape[1]),
            "frames_per_track": int(pred_tracks.shape[2]),
            "gt_occluded_points": int(gt_occluded.sum()),
            "total_points": int(gt_occluded.size),
        },
    }

    if output_report_json:
        _write_json(output_report_json, report)
    return report


def parse_args() -> Any:
    p = ArgumentParser(description="Export a Stage2 TAP-style proxy payload into official TAP-Vid metric format")
    p.add_argument("--proxy-payload-npz", required=True)
    p.add_argument("--output-npz", required=True)
    p.add_argument("--output-report-json", default="")
    p.add_argument("--raster-resolution", type=int, default=256)
    p.add_argument("--pred-occlusion-mode", default="all_visible")
    p.add_argument("--query-time-index", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = export_proxy_payload_to_tapvid(
        proxy_payload_npz=args.proxy_payload_npz,
        output_npz=args.output_npz,
        output_report_json=args.output_report_json or None,
        raster_resolution=int(args.raster_resolution),
        pred_occlusion_mode=str(args.pred_occlusion_mode),
        query_time_index=int(args.query_time_index),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
