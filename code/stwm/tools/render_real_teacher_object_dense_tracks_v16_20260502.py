#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
CACHE_BASE = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16"
OUT_DIR = ROOT / "assets/videos/stwm_real_teacher_object_dense_v16"
COLORS = [(255, 99, 71), (46, 204, 113), (52, 152, 219), (241, 196, 15), (155, 89, 182), (26, 188, 156), (230, 126, 34), (255, 255, 255)]


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


def _scalar(arr: np.ndarray) -> Any:
    a = np.asarray(arr)
    return a.item() if a.shape == () else a.reshape(-1)[0]


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


def _draw_query_points(draw: ImageDraw.ImageDraw, pts: np.ndarray, scale: float, ox: int, oy: int) -> None:
    q = _transform(pts, scale, ox, oy)
    step = max(len(q) // 96, 1)
    for x, y in q[::step]:
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(80, 180, 255))


def _render(path: Path, case_idx: int) -> dict[str, Any]:
    z = np.load(path, allow_pickle=True)
    frames = [str(x) for x in z["frame_paths"].tolist()]
    tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
    visibility = np.asarray(z["visibility"]).astype(bool)
    query = np.asarray(z["query_points_xy"], dtype=np.float32)
    scale, ox, oy = _scale_meta(np.asarray(z["raw_size"]))
    obj = int(np.argmax(visibility.mean(axis=(1, 2))))
    point_indices = np.linspace(0, tracks.shape[1] - 1, num=min(128, tracks.shape[1]), dtype=int).tolist()
    tr_draw = _transform(tracks[obj], scale, ox, oy)
    out_frames = []
    for t, frame in enumerate(frames):
        img = _load_frame(frame)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 960, 88), fill=(0, 0, 0))
        draw.text((12, 8), f"V16 CoTracker official {path.parent.parent.name} item={_scalar(z['item_key'])} obj={obj} t={t}", fill=(255, 255, 255))
        draw.text((12, 30), "raw temporal frames + object mask/query points + persistent colored point tracks", fill=(255, 220, 160))
        draw.text((12, 52), "teacher_source=cotracker_official; red x=invisible/occluded marker", fill=(180, 230, 255))
        if t == int(_scalar(z["query_frame"])):
            _draw_query_points(draw, query[obj], scale, ox, oy)
        for n, pidx in enumerate(point_indices):
            _draw_history(draw, tr_draw[pidx], visibility[obj, pidx], t, COLORS[n % len(COLORS)])
        out_frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"case{case_idx:02d}_{path.parent.parent.name}_{str(_scalar(z['dataset'])).lower()}_{str(_scalar(z['split']))}.gif"
    out_frames[0].save(out, save_all=True, append_images=out_frames[1:], duration=180, loop=0)
    return {
        "case_index": case_idx,
        "source_cache": str(path.relative_to(ROOT)),
        "gif_path": str(out.relative_to(ROOT)),
        "combo": path.parent.parent.name,
        "dataset": str(_scalar(z["dataset"])),
        "split": str(_scalar(z["split"])),
        "teacher_source": str(_scalar(z["teacher_source"])),
        "raw_temporal_frames": True,
        "object_mask_query_points": True,
        "colored_persistent_point_tracks": True,
        "visibility_markers": True,
    }


def main() -> int:
    all_paths = sorted(CACHE_BASE.glob("M*_H*/*/*.npz"))
    selected: list[Path] = []
    for combo in ["M128_H8", "M512_H8", "M128_H16", "M512_H16"]:
        for split in ["train", "val", "test"]:
            for ds in ["VSPW", "VIPSEG"]:
                for p in all_paths:
                    if p.parent.parent.name != combo or p in selected:
                        continue
                    z = np.load(p, allow_pickle=True)
                    if str(_scalar(z["split"])) == split and str(_scalar(z["dataset"])) == ds:
                        selected.append(p)
                        break
                if len(selected) >= 20:
                    break
            if len(selected) >= 20:
                break
        if len(selected) >= 20:
            break
    for p in all_paths:
        if len(selected) >= 20:
            break
        if p not in selected:
            selected.append(p)
    cases = [_render(p, idx) for idx, p in enumerate(selected[:20])]
    payload = {
        "audit_name": "stwm_real_teacher_object_dense_visualization_v16",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "visualization_ready": len(cases) >= 20,
        "gif_count": len(cases),
        "cases": cases,
        "teacher_source_overlay": "cotracker_official",
        "pseudo_bbox_relative_label_used": False,
    }
    _dump(ROOT / "reports/stwm_real_teacher_object_dense_visualization_v16_20260502.json", payload)
    print("reports/stwm_real_teacher_object_dense_visualization_v16_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
