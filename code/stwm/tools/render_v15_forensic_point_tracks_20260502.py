#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent))
from stwm_v15b_forensic_utils_20260502 import (  # noqa: E402
    ROOT,
    dump_json,
    load_cache,
    now_utc,
    predecode_for_item,
    predecode_index,
    trajectory_same_delta_ratio,
)

sys.path.append(str(ROOT / "code"))
from stwm.modules.dense_to_semantic_trace_unit_v15 import OSTFMultiTracePilot  # noqa: E402


COLORS = [
    (231, 76, 60),
    (46, 204, 113),
    (52, 152, 219),
    (241, 196, 15),
    (155, 89, 182),
    (26, 188, 156),
    (230, 126, 34),
    (236, 240, 241),
]


def _load_model(m: int) -> tuple[OSTFMultiTracePilot | None, float]:
    ckpt_path = ROOT / f"outputs/checkpoints/stwm_ostf_multitrace_pilot_v15/M{m}_seed42.pt"
    if not ckpt_path.exists():
        return None, 4096.0
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = OSTFMultiTracePilot(obs_len=8, horizon=8, unit_dim=128)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, float(ckpt.get("scale", 4096.0))


def _predict(model: OSTFMultiTracePilot | None, scale: float, points: np.ndarray, valid: np.ndarray, rel: np.ndarray) -> np.ndarray | None:
    if model is None:
        return None
    obs = torch.tensor(points[:8].astype(np.float32) / scale, dtype=torch.float32).permute(1, 0, 2)[None]
    ov = torch.tensor(valid[:8].astype(bool), dtype=torch.bool).permute(1, 0)[None]
    rr = torch.tensor(rel.astype(np.float32), dtype=torch.float32)[None]
    with torch.no_grad():
        pred = model(obs, ov, rr)[0].numpy() * scale
    return pred  # [H, M, 2]


def _background(pre_path: Path | None) -> tuple[Image.Image, dict[str, Any]]:
    meta: dict[str, Any] = {"raw_frame_path": None, "raw_frame_exists": False, "scale": 1.0, "offset_x": 0, "offset_y": 0}
    if pre_path is None:
        return Image.new("RGB", (960, 540), (20, 20, 20)), meta
    z = np.load(pre_path, allow_pickle=True)
    frame_path = None
    if "semantic_frame_path" in z.files:
        arr = np.asarray(z["semantic_frame_path"])
        frame_path = str(arr.item() if arr.shape == () else arr.reshape(-1)[0])
    meta["raw_frame_path"] = frame_path
    if frame_path and Path(frame_path).exists():
        img = Image.open(frame_path).convert("RGB")
        meta["raw_frame_exists"] = True
    else:
        crop = np.asarray(z["semantic_rgb_crop"][0]).transpose(1, 2, 0)
        img = Image.fromarray(np.clip(crop * 255, 0, 255).astype(np.uint8)).convert("RGB").resize((512, 512))
        meta["raw_frame_exists"] = False
    original_w, original_h = img.width, img.height
    img.thumbnail((960, 540))
    scale = min(img.width / max(original_w, 1), img.height / max(original_h, 1))
    canvas = Image.new("RGB", (960, 540), (16, 18, 22))
    ox, oy = (960 - img.width) // 2, (540 - img.height) // 2
    meta.update({"scale": scale, "offset_x": ox, "offset_y": oy, "drawn_width": img.width, "drawn_height": img.height})
    canvas.paste(img, (ox, oy))
    return canvas, meta


def _select_cases(m: int, max_cases: int = 12) -> list[tuple[int, int, str]]:
    z = load_cache(m)
    valid = z["object_valid_mask"].astype(bool)
    datasets = [str(x) for x in z["datasets"].tolist()]
    points = z["points_xy"]
    cases: list[tuple[int, int, str, float]] = []
    for i, j in zip(*np.where(valid)):
        p = np.asarray(points[i, j], dtype=np.float32)
        motion = float(np.nanmean(np.linalg.norm(p[-1] - p[0], axis=-1)))
        cases.append((int(i), int(j), datasets[int(i)], motion))
    vspw = sorted([x for x in cases if x[2] == "VSPW"], key=lambda x: x[3], reverse=True)[:4]
    vipseg = sorted([x for x in cases if x[2] == "VIPSEG"], key=lambda x: x[3], reverse=True)[:4]
    high = sorted(cases, key=lambda x: x[3], reverse=True)[:4]
    merged: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for rows in [vspw, vipseg, high, cases]:
        for i, j, ds, _motion in rows:
            if (i, j) in seen:
                continue
            merged.append((i, j, ds))
            seen.add((i, j))
            if len(merged) >= max_cases:
                return merged
    return merged


def _draw_track(draw: ImageDraw.ImageDraw, pts: np.ndarray, upto_t: int, color: tuple[int, int, int], radius: int = 2) -> None:
    upto_t = min(upto_t, pts.shape[0] - 1)
    history = [(float(pts[t, 0]), float(pts[t, 1])) for t in range(upto_t + 1) if np.isfinite(pts[t]).all()]
    if len(history) >= 2:
        draw.line(history, fill=color, width=2)
    if history:
        x, y = history[-1]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)


def _transform_xy(xy: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    out = xy.astype(np.float32, copy=True)
    out[..., 0] = out[..., 0] * float(meta.get("scale", 1.0)) + float(meta.get("offset_x", 0))
    out[..., 1] = out[..., 1] * float(meta.get("scale", 1.0)) + float(meta.get("offset_y", 0))
    return out


def _render_case(m: int, case_idx: int, i: int, j: int, ds: str, out_dir: Path, index: dict[str, Path]) -> dict[str, Any]:
    z = load_cache(m)
    item_key = str(z["item_keys"][i])
    pre_path = predecode_for_item(item_key, index)
    base, bg_meta = _background(pre_path)
    points = np.asarray(z["points_xy"][i, j], dtype=np.float32)
    valid = np.asarray(z["valid_mask"][i, j], dtype=bool)
    rel = np.asarray(z["object_relative_xy"][i, j], dtype=np.float32)
    model, scale = _load_model(m)
    pred = _predict(model, scale, points, valid, rel)
    points_draw = _transform_xy(points, bg_meta)
    pred_draw = _transform_xy(pred, bg_meta) if pred is not None else None
    same_delta = trajectory_same_delta_ratio(points)
    point_indices = np.linspace(0, m - 1, num=min(32, m), dtype=int).tolist()
    frames: list[Image.Image] = []
    for t in range(16):
        img = base.copy()
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 960, 92), fill=(0, 0, 0))
        draw.text((12, 10), f"V15B forensic {ds} item={item_key} obj={j} M={m} t={t}", fill=(255, 255, 255))
        draw.text((12, 32), "teacher target: mask_bbox_relative_pseudo_track; colored lines=point IDs", fill=(255, 220, 160))
        draw.text((12, 54), f"same-delta-ratio={same_delta:.3f}; raw frame exists={bg_meta['raw_frame_exists']}", fill=(180, 220, 255))
        if pre_path is not None:
            try:
                pre = np.load(pre_path, allow_pickle=True)
                boxes = np.asarray(pre["entity_boxes_over_time"], dtype=np.float32)
                if j < boxes.shape[1] and t < boxes.shape[0]:
                    x0, y0, x1, y1 = boxes[t, j].tolist()
                    x0 = x0 * float(bg_meta["scale"]) + float(bg_meta["offset_x"])
                    x1 = x1 * float(bg_meta["scale"]) + float(bg_meta["offset_x"])
                    y0 = y0 * float(bg_meta["scale"]) + float(bg_meta["offset_y"])
                    y1 = y1 * float(bg_meta["scale"]) + float(bg_meta["offset_y"])
                    draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255), width=2)
            except Exception:
                pass
        for n, pidx in enumerate(point_indices):
            color = COLORS[n % len(COLORS)]
            _draw_track(draw, points_draw[:, pidx], t, color, radius=2)
            if pred_draw is not None and t >= 8:
                pred_track = pred_draw[: t - 7, pidx]
                if pred_track.shape[0] >= 2:
                    draw.line([(float(x), float(y)) for x, y in pred_track], fill=(255, 255, 255), width=1)
                if pred_track.shape[0]:
                    x, y = pred_track[-1]
                    draw.rectangle((x - 2, y - 2, x + 2, y + 2), outline=(255, 255, 255))
            if not valid[t, pidx]:
                x, y = points_draw[t, pidx]
                draw.line((x - 4, y - 4, x + 4, y + 4), fill=(255, 0, 0), width=2)
                draw.line((x - 4, y + 4, x + 4, y - 4), fill=(255, 0, 0), width=2)
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / f"case{case_idx:02d}_{ds.lower()}_M{m}.gif"
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=180, loop=0)
    return {
        "case_index": case_idx,
        "dataset": ds,
        "M": m,
        "item_key": item_key,
        "object_index": j,
        "gif_path": str(gif_path.relative_to(ROOT)),
        "raw_frame_path": bg_meta["raw_frame_path"],
        "raw_frame_exists": bg_meta["raw_frame_exists"],
        "point_id_colored_tracks": True,
        "future_teacher_tracks": True,
        "model_predicted_tracks": pred is not None,
        "visibility_markers": True,
        "same_delta_ratio": same_delta,
        "points_share_identical_trajectories_flag": bool(same_delta > 0.95),
    }


def main() -> int:
    out_dir = ROOT / "assets/videos/stwm_v15_forensic_tracks"
    index = predecode_index()
    rows: list[dict[str, Any]] = []
    # Six M128 and six M512 cases: covers both requested densities and enough datasets/failures.
    for m, offset in [(128, 0), (512, 6)]:
        for local_idx, (i, j, ds) in enumerate(_select_cases(m, max_cases=6)):
            rows.append(_render_case(m, offset + local_idx, i, j, ds, out_dir, index))
    payload: dict[str, Any] = {
        "audit_name": "stwm_v15_forensic_visualization",
        "generated_at_utc": now_utc(),
        "output_dir": str(out_dir.relative_to(ROOT)),
        "gif_count": len(rows),
        "cases": rows,
        "raw_observed_frames": "single semantic_frame_path background when available; not full raw temporal video sequence",
        "object_mask_display": "bbox and point overlays; full per-frame mask overlay not reliably available for all cases",
        "visualization_not_trajectory": False,
        "teacher_is_physical_trajectory": False,
        "teacher_source": "mask_bbox_relative_pseudo_track",
        "claim_note": "These forensic GIFs verify cached point paths and identity colors; they do not establish physical dense tracking teacher quality.",
    }
    out = ROOT / "reports/stwm_v15_forensic_visualization_20260502.json"
    dump_json(out, payload)
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
