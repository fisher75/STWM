#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json
from stwm.tools.run_traceanything_object_trajectory_teacher_v2_20260502 import _locate_cotracker_cache, _scalar


CACHE_ROOT = ROOT / "outputs/cache/stwm_traceanything_hardbench_v25"
OUT_DIR = ROOT / "assets/videos/stwm_traceanything_hardbench_v25"
REPORT_PATH = ROOT / "reports/stwm_traceanything_hardbench_visualization_v25_20260502.json"
COLORS = [
    (255, 99, 71),
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (255, 255, 255),
]


def _load_frame(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((960, 540))
    canvas = Image.new("RGB", (960, 540), (12, 14, 18))
    canvas.paste(img, ((960 - img.width) // 2, (540 - img.height) // 2))
    return canvas


def _scale_meta(raw_size: np.ndarray) -> tuple[float, int, int]:
    raw_w, raw_h = int(raw_size[0]), int(raw_size[1])
    scale = min(960 / max(raw_w, 1), 540 / max(raw_h, 1), 1.0)
    draw_w, draw_h = int(round(raw_w * scale)), int(round(raw_h * scale))
    return scale, (960 - draw_w) // 2, (540 - draw_h) // 2


def _transform(points: np.ndarray, scale: float, ox: int, oy: int) -> np.ndarray:
    out = points.astype(np.float32, copy=True)
    out[..., 0] = out[..., 0] * scale + ox
    out[..., 1] = out[..., 1] * scale + oy
    return out


def _draw_history(draw: ImageDraw.ImageDraw, tr: np.ndarray, vis: np.ndarray, t: int, color: tuple[int, int, int], width: int = 2) -> None:
    hist = [(float(tr[k, 0]), float(tr[k, 1])) for k in range(min(t + 1, len(tr))) if bool(vis[k]) and np.isfinite(tr[k]).all()]
    if len(hist) >= 2:
        draw.line(hist, fill=color, width=width)
    if hist:
        x, y = hist[-1]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def _pick_cases(limit: int = 30) -> list[Path]:
    files = sorted(CACHE_ROOT.glob("M*_H*/*/*.npz"))
    selected: list[Path] = []
    seen = set()
    for combo in ["M128_H32", "M512_H32", "M128_H64", "M512_H64"]:
        for split in ["train", "val", "test"]:
            for f in sorted((CACHE_ROOT / combo / split).glob("*.npz")):
                z = np.load(f, allow_pickle=True)
                dataset = str(_scalar(z["dataset"]))
                key = (combo, split, dataset)
                if key not in seen:
                    selected.append(f)
                    seen.add(key)
                if len(selected) >= limit:
                    return selected[:limit]
    for f in files:
        if f not in selected:
            selected.append(f)
        if len(selected) >= limit:
            break
    return selected[:limit]


def _render_case(path: Path, case_idx: int) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    frame_paths = [str(x) for x in z["frame_paths"].tolist()]
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    visibility = np.asarray(z["visibility"]).astype(bool)
    query_points = np.asarray(z["query_points_xy"], dtype=np.float32)
    raw_size = np.asarray(z["raw_size"])
    scale, ox, oy = _scale_meta(raw_size)
    obj_scores = visibility.mean(axis=(1, 2))
    obj = int(np.argmax(obj_scores))
    point_indices = np.linspace(0, tracks.shape[1] - 1, num=min(128, tracks.shape[1]), dtype=int).tolist()
    tr_draw = _transform(tracks[obj], scale, ox, oy)
    split = str(_scalar(z["split"]))
    item_key = str(_scalar(z["item_key"]))
    dataset = str(_scalar(z["dataset"]))
    m = int(_scalar(z["M"]))
    cot_path = _locate_cotracker_cache(item_key, split, m, 16)
    cot = np.load(cot_path, allow_pickle=True) if cot_path and cot_path.exists() else None
    cot_draw = None
    cot_vis = None
    if cot is not None:
        cot_draw = _transform(np.asarray(cot["tracks_xy"], dtype=np.float32)[obj], scale, ox, oy)
        cot_vis = np.asarray(cot["visibility"]).astype(bool)[obj]

    frames: list[Image.Image] = []
    for t, frame_path in enumerate(frame_paths):
        img = _load_frame(frame_path)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 960, 96), fill=(0, 0, 0))
        draw.text((12, 8), f"TraceAnything hardbench V25 item={item_key} obj={obj} t={t}", fill=(255, 255, 255))
        draw.text((12, 30), f"dataset={dataset} split={split} M={m} H={int(_scalar(z['horizon']))} teacher=traceanything_official_trajectory_field", fill=(255, 220, 160))
        draw.text((12, 52), "solid colors = TraceAnything persistent points; cyan = query points; white = CoTracker H16 prefix ref if available", fill=(180, 230, 255))
        draw.text((12, 74), "visibility estimated from confidence + consistency; model input remains observed-only", fill=(200, 200, 200))
        if t == int(_scalar(z["query_frame"])):
            qp = _transform(query_points[obj], scale, ox, oy)
            for x, y in qp[:: max(len(qp) // 64, 1)]:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(80, 255, 255))
        for n, pidx in enumerate(point_indices):
            _draw_history(draw, tr_draw[pidx], visibility[obj, pidx], t, COLORS[n % len(COLORS)], width=2)
        if cot_draw is not None and cot_vis is not None:
            for pidx in point_indices[::4]:
                _draw_history(draw, cot_draw[pidx], cot_vis[pidx], min(t, cot_draw.shape[1] - 1), (255, 255, 255), width=1)
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combo = path.parent.parent.name
    out = OUT_DIR / f"case{case_idx:02d}_{combo}_{dataset.lower()}_{split}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=180, loop=0)
    return {
        "case_index": case_idx,
        "source_cache": str(path.relative_to(ROOT)),
        "gif_path": str(out.relative_to(ROOT)),
        "dataset": dataset,
        "split": split,
        "item_key": item_key,
        "combo": combo,
        "cotracker_reference_available": cot is not None,
    }


def main() -> int:
    selected = _pick_cases(30)
    cases = [_render_case(path, idx) for idx, path in enumerate(selected)]
    payload = {
        "audit_name": "stwm_traceanything_hardbench_visualization_v25",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "generated_gif_count": len(cases),
        "traceanything_hardbench_visualization_ready": len(cases) >= 30,
        "cases": cases,
    }
    dump_json(REPORT_PATH, payload)
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
