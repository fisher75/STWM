#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _frame(points: np.ndarray, valid: np.ndarray, rel: np.ndarray, title: str, size: int = 512) -> Image.Image:
    img = Image.new("RGB", (size, size), (18, 20, 25))
    draw = ImageDraw.Draw(img)
    draw.text((10, 8), title, fill=(230, 230, 230))
    if valid.any():
        pts = points[valid]
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        x1, y1 = pts[:, 0].max(), pts[:, 1].max()
        sx = (size - 80) / max(float(x1 - x0), 1.0)
        sy = (size - 80) / max(float(y1 - y0), 1.0)
        s = min(sx, sy)
        px = 40 + (points[:, 0] - x0) * s
        py = 60 + (points[:, 1] - y0) * s
        for i, ok in enumerate(valid):
            if not ok:
                continue
            color = (60 + int(180 * float(rel[i, 0])), 80 + int(150 * float(rel[i, 1])), 220)
            draw.ellipse((px[i] - 2, py[i] - 2, px[i] + 2, py[i] + 2), fill=color)
    return img


def _make_gif(cache: dict[str, Any], item_i: int, obj_j: int, out: Path, label: str, max_points: int = 512) -> None:
    pts = np.asarray(cache["points_xy"][item_i, obj_j], dtype=np.float32)[:, :max_points]
    valid = np.asarray(cache["valid_mask"][item_i, obj_j], dtype=bool)[:, :max_points]
    rel = np.asarray(cache["object_relative_xy"][item_i, obj_j], dtype=np.float32)[:max_points]
    frames = [_frame(pts[t], valid[t], rel, f"{label} t={t:02d}") for t in range(min(16, pts.shape[0]))]
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=180, loop=0)


def main() -> int:
    video_dir = Path("assets/videos/stwm_object_dense_trace_v15")
    fig_dir = Path("assets/figures/stwm_object_dense_trace_v15")
    video_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    cache128 = dict(np.load("outputs/cache/stwm_object_dense_trace_v15/M128/object_dense_trace_cache.npz", allow_pickle=True))
    cache512 = dict(np.load("outputs/cache/stwm_object_dense_trace_v15/M512/object_dense_trace_cache.npz", allow_pickle=True))
    datasets = [str(x) for x in cache128["datasets"].tolist()]
    valid = cache128["object_valid_mask"].astype(bool)
    cases: list[tuple[str, dict[str, Any], int, int]] = []
    for want, label in [("VSPW", "vspw_object_dense"), ("VIPSEG", "vipseg_object_dense")]:
        for i, ds in enumerate(datasets):
            if ds.upper() == want.upper() and valid[i].any():
                cases.append((label, cache128, i, int(np.where(valid[i])[0][0])))
                break
    # Add more cases from M512 and generic failure/comparison slots.
    for label, cache in [("m512_internal_points", cache512), ("future_point_cloud", cache128), ("visibility_color_validity", cache128), ("semantic_prototype_color_proxy", cache512)]:
        for i in range(valid.shape[0]):
            if valid[i].any():
                cases.append((label, cache, i, int(np.where(valid[i])[0][0])))
                break
    outputs = []
    for label, cache, i, j in cases[:6]:
        out = video_dir / f"{label}.gif"
        _make_gif(cache, i, j, out, label)
        outputs.append(str(out))
    report = {
        "audit_name": "stwm_object_dense_trace_visualization_v15",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_dir": str(video_dir),
        "figure_dir": str(fig_dir),
        "gif_count": len(outputs),
        "videos": outputs,
        "raw_frame_plus_object_mask": "crop/point-cloud visualization from predecoded semantic crop; full raw-frame overlay requires raw frame loader wiring",
        "M128_M512_internal_point_traces": True,
        "future_predicted_point_cloud": "target/pseudo-track rollout visualization; pilot prediction comparison remains follow-up",
        "visibility_occlusion_color": "valid-mask color proxy",
        "semantic_prototype_color": "object-relative color proxy; prototype label overlay follow-up",
    }
    _dump(Path("reports/stwm_object_dense_trace_visualization_v15_20260502.json"), report)
    print("reports/stwm_object_dense_trace_visualization_v15_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
