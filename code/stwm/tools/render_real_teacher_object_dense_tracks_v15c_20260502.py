#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v15c"
OUT_DIR = ROOT / "assets/videos/stwm_real_teacher_object_dense_v15c"


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


def _scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


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


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _draw_mask_proxy(draw: ImageDraw.ImageDraw, query_points: np.ndarray, scale: float, ox: int, oy: int) -> None:
    pts = _transform(query_points, scale, ox, oy)
    for x, y in pts[:: max(len(pts) // 64, 1)]:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(80, 180, 255))


def _draw_history(draw: ImageDraw.ImageDraw, tr: np.ndarray, vis: np.ndarray, t: int, color: tuple[int, int, int]) -> None:
    hist = [(float(tr[k, 0]), float(tr[k, 1])) for k in range(t + 1) if bool(vis[k]) and np.isfinite(tr[k]).all()]
    if len(hist) >= 2:
        draw.line(hist, fill=color, width=2)
    if hist:
        x, y = hist[-1]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    elif t < len(tr):
        x, y = tr[t]
        draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=2)
        draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=2)


def _render(path: Path, case_idx: int) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    frame_paths = [str(x) for x in z["frame_paths"].tolist()]
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    visibility = np.asarray(z["visibility"]).astype(bool)
    query_points = np.asarray(z["query_points_xy"], dtype=np.float32)
    raw_size = np.asarray(z["raw_size"])
    scale, ox, oy = _scale_meta(raw_size)
    # Pick object with most visible points and draw up to 128 colored point identities.
    obj_scores = visibility.mean(axis=(1, 2))
    obj = int(np.argmax(obj_scores))
    point_indices = np.linspace(0, tracks.shape[1] - 1, num=min(128, tracks.shape[1]), dtype=int).tolist()
    tr_draw = _transform(tracks[obj], scale, ox, oy)  # [M,T,2]
    frames: list[Image.Image] = []
    for t, frame_path in enumerate(frame_paths):
        img = _load_frame(frame_path)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 960, 86), fill=(0, 0, 0))
        draw.text((12, 8), f"CoTracker official teacher V15C item={_scalar(z['item_key'])} obj={obj} t={t}", fill=(255, 255, 255))
        draw.text((12, 30), "raw temporal frame + query mask points + 128 persistent point tracks", fill=(255, 220, 160))
        draw.text((12, 52), "visibility: missing/invisible points are red x markers; teacher_source=cotracker_official", fill=(180, 230, 255))
        if t == int(_scalar(z["query_frame"])):
            _draw_mask_proxy(draw, query_points[obj], scale, ox, oy)
        for n, pidx in enumerate(point_indices):
            _draw_history(draw, tr_draw[pidx], visibility[obj, pidx], t, COLORS[n % len(COLORS)])
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"case{case_idx:02d}_{str(_scalar(z['dataset'])).lower()}_{str(_scalar(z['split']))}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=180, loop=0)
    return {
        "case_index": case_idx,
        "source_cache": str(path.relative_to(ROOT)),
        "gif_path": str(out.relative_to(ROOT)),
        "item_key": str(_scalar(z["item_key"])),
        "dataset": str(_scalar(z["dataset"])),
        "split": str(_scalar(z["split"])),
        "object_index": obj,
        "raw_temporal_frames": True,
        "object_mask_query_points": True,
        "point_id_colored_tracks": True,
        "visibility_markers": True,
        "teacher_source": str(_scalar(z["teacher_source"])),
    }


def main() -> int:
    paths = sorted(CACHE_ROOT.glob("*/*.npz"))
    selected: list[Path] = []
    for split in ["train", "val", "test"]:
        for ds in ["VSPW", "VIPSEG"]:
            for p in paths:
                z = np.load(p, allow_pickle=True)
                if str(_scalar(z["split"])) == split and str(_scalar(z["dataset"])) == ds and p not in selected:
                    selected.append(p)
                    break
    for p in paths:
        if len(selected) >= 8:
            break
        if p not in selected:
            selected.append(p)
    cases = [_render(p, idx) for idx, p in enumerate(selected[:8])]
    payload = {
        "audit_name": "stwm_real_teacher_object_dense_visualization_v15c",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "visualization_ready": len(cases) >= 8,
        "gif_count": len(cases),
        "cases": cases,
        "raw_temporal_frames": True,
        "teacher_source_overlay": "cotracker_official",
        "pseudo_bbox_relative_label_used": False,
    }
    _dump(ROOT / "reports/stwm_real_teacher_object_dense_visualization_v15c_20260502.json", payload)
    print("reports/stwm_real_teacher_object_dense_visualization_v15c_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
