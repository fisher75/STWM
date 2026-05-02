#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.modules.ostf_multitrace_world_model_v17 import OSTFMultiTraceWorldModel, OSTFMultiTraceWorldModelConfig
from stwm.tools.ostf_v17_common_20260502 import OSTFObjectSample, ROOT, batch_from_samples, load_json, load_v16_samples


OUT_DIR = ROOT / "assets/videos/stwm_ostf_v17"
REPORT_PATH = ROOT / "reports/stwm_ostf_v17_visualization_20260502.json"
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
    canvas = Image.new("RGB", (1280, 720), (10, 12, 18))
    ox = 20 + (960 - img.width) // 2
    oy = 90 + (540 - img.height) // 2
    canvas.paste(img, (ox, oy))
    return canvas


def _scale_meta(raw_size: np.ndarray) -> tuple[float, int, int]:
    raw_w, raw_h = int(raw_size[0]), int(raw_size[1])
    scale = min(960 / max(raw_w, 1), 540 / max(raw_h, 1), 1.0)
    draw_w, draw_h = int(round(raw_w * scale)), int(round(raw_h * scale))
    return scale, 20 + (960 - draw_w) // 2, 90 + (540 - draw_h) // 2


def _transform(points: np.ndarray, scale: float, ox: int, oy: int) -> np.ndarray:
    out = points.astype(np.float32, copy=True)
    out[..., 0] = out[..., 0] * scale + ox
    out[..., 1] = out[..., 1] * scale + oy
    return out


def _load_model(checkpoint_rel: str) -> tuple[OSTFMultiTraceWorldModel, np.ndarray]:
    ckpt = torch.load(ROOT / checkpoint_rel, map_location="cpu", weights_only=False)
    cfg = OSTFMultiTraceWorldModelConfig()
    model = OSTFMultiTraceWorldModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, np.asarray(ckpt["proto_centers"], dtype=np.float32)


def _predict(model: OSTFMultiTraceWorldModel, sample) -> dict[str, np.ndarray]:
    batch = batch_from_samples([sample], torch.device("cpu"))
    with torch.no_grad():
        out = model(
            obs_points=batch["obs_points"],
            obs_vis=batch["obs_vis"],
            rel_xy=batch["rel_xy"],
            anchor_obs=batch["anchor_obs"],
            anchor_obs_vel=batch["anchor_obs_vel"],
            semantic_feat=batch["semantic_feat"],
        )
    return {
        "point_pred": out["point_pred"][0].cpu().numpy(),
        "visibility_logits": out["visibility_logits"][0].cpu().numpy(),
        "semantic_logits": out["semantic_logits"][0].cpu().numpy(),
    }


def _collapse_single_to_m1(sample: OSTFObjectSample) -> OSTFObjectSample:
    return OSTFObjectSample(
        item_key=sample.item_key,
        dataset=sample.dataset,
        split=sample.split,
        source_cache_path=sample.source_cache_path,
        object_index=sample.object_index,
        object_id=sample.object_id,
        m=1,
        h=sample.h,
        obs_points=sample.anchor_obs[None].copy(),
        fut_points=sample.anchor_fut[None].copy(),
        obs_vis=np.ones((1, sample.anchor_obs.shape[0]), dtype=bool),
        fut_vis=np.ones((1, sample.anchor_fut.shape[0]), dtype=bool),
        rel_xy=np.asarray([[0.5, 0.5]], dtype=np.float32),
        anchor_obs=sample.anchor_obs.copy(),
        anchor_fut=sample.anchor_fut.copy(),
        anchor_obs_vel=sample.anchor_obs_vel.copy(),
        semantic_feat=sample.semantic_feat.copy(),
        semantic_valid=sample.semantic_valid,
        semantic_id=sample.semantic_id,
        proto_target=sample.proto_target,
    )


def _bbox_xyxy(pre: np.lib.npyio.NpzFile, object_id: int, t: int) -> np.ndarray | None:
    if "entity_boxes_over_time" not in pre.files:
        return None
    boxes = np.asarray(pre["entity_boxes_over_time"], dtype=np.float32)
    if t >= boxes.shape[0] or object_id >= boxes.shape[1]:
        return None
    return boxes[t, object_id]


def _draw_polyline(draw: ImageDraw.ImageDraw, pts: np.ndarray, vis: np.ndarray, upto: int, color: tuple[int, int, int], width: int) -> None:
    hist = [(float(pts[k, 0]), float(pts[k, 1])) for k in range(min(upto + 1, len(pts))) if bool(vis[k]) and np.isfinite(pts[k]).all()]
    if len(hist) >= 2:
        draw.line(hist, fill=color, width=width)
    if hist:
        x, y = hist[-1]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def _render_case(case_index: int, label: str, sample, dense_pred: dict[str, np.ndarray], anchor_pred: dict[str, np.ndarray]) -> dict[str, Any]:
    cache_npz = ROOT / sample.source_cache_path
    z = np.load(cache_npz, allow_pickle=True)
    pre = np.load(str(z["predecode_path"].item()), allow_pickle=True)
    frame_paths = [str(x) for x in z["frame_paths"].tolist()]
    raw_size = np.asarray(z["raw_size"], dtype=np.float32)
    scale, ox, oy = _scale_meta(raw_size)
    object_id = int(sample.object_id)
    tracks_full = np.asarray(z["tracks_xy"], dtype=np.float32)[object_id]
    vis_full = np.asarray(z["visibility"]).astype(bool)[object_id]
    teacher_tr = _transform(tracks_full, scale, ox, oy)
    pred_dense_full = np.concatenate([sample.obs_points, dense_pred["point_pred"]], axis=1) * float(max(raw_size.tolist()))
    pred_dense_full = _transform(pred_dense_full, scale, ox, oy)
    pred_dense_vis = np.concatenate([sample.obs_vis, dense_pred["visibility_logits"] > 0], axis=1)
    pred_anchor_full = np.concatenate([sample.anchor_obs[None], anchor_pred["point_pred"]], axis=1)[0] * float(max(raw_size.tolist()))
    pred_anchor_vis = np.concatenate([np.ones((1, sample.anchor_obs.shape[0]), dtype=bool), anchor_pred["visibility_logits"] > 0], axis=1)[0]
    pred_anchor_full = _transform(pred_anchor_full, scale, ox, oy)
    query = _transform(np.asarray(z["query_points_xy"], dtype=np.float32)[object_id], scale, ox, oy)
    semantic_crop = None
    if "semantic_rgb_crop" in pre.files and object_id < pre["semantic_rgb_crop"].shape[0]:
        crop = np.asarray(pre["semantic_rgb_crop"][object_id]).transpose(1, 2, 0)
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        semantic_crop = Image.fromarray(crop).resize((160, 160))
    mask_crop = None
    if "semantic_mask_crop" in pre.files and object_id < pre["semantic_mask_crop"].shape[0]:
        mask = np.asarray(pre["semantic_mask_crop"][object_id, 0])
        mask = (255.0 * np.clip(mask, 0.0, 1.0)).astype(np.uint8)
        mask_crop = Image.fromarray(mask, mode="L").resize((160, 160))
    point_indices = np.linspace(0, tracks_full.shape[0] - 1, num=min(64, tracks_full.shape[0]), dtype=int)
    frames: list[Image.Image] = []
    obs_len = int(z["obs_len"].item())
    total_t = tracks_full.shape[1]
    pred_proto = int(np.asarray(dense_pred["semantic_logits"]).mean(axis=0).argmax())
    for t, frame_path in enumerate(frame_paths[:total_t]):
        img = _load_frame(frame_path)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 1280, 78), fill=(0, 0, 0))
        draw.text((16, 8), f"OSTF V17 {label} case={case_index:02d} dataset={sample.dataset} item={sample.item_key} obj={sample.object_id} t={t}", fill=(255, 255, 255))
        draw.text((16, 30), "white=teacher observed, red=teacher future, green=OSTF dense pred, cyan=matched M1 anchor", fill=(255, 220, 160))
        draw.text((16, 52), f"teacher_source=cotracker_official pred_proto={pred_proto}", fill=(180, 230, 255))
        bbox = _bbox_xyxy(pre, object_id, t)
        if bbox is not None and np.isfinite(bbox).all():
            bb = _transform(np.asarray(bbox[None], dtype=np.float32), scale, ox, oy)[0]
            x0, x1 = sorted([float(bb[0]), float(bb[2])])
            y0, y1 = sorted([float(bb[1]), float(bb[3])])
            draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=2)
        if t == int(z["query_frame"].item()):
            for x, y in query[:: max(len(query) // 96, 1)]:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(80, 180, 255))
        for n, pidx in enumerate(point_indices):
            teacher_color = (255, 255, 255) if t < obs_len else (255, 96, 96)
            _draw_polyline(draw, teacher_tr[pidx], vis_full[pidx], t, teacher_color, 2)
            if t >= obs_len:
                _draw_polyline(draw, pred_dense_full[pidx], pred_dense_vis[pidx], t, COLORS[n % len(COLORS)], 2)
        _draw_polyline(draw, pred_anchor_full, pred_anchor_vis, t, (80, 255, 255), 3)
        if semantic_crop is not None:
            img.paste(semantic_crop, (1085, 100))
            draw.rectangle((1085, 100, 1245, 260), outline=(255, 255, 255), width=1)
            draw.text((1085, 264), "semantic crop", fill=(255, 255, 255))
        if mask_crop is not None:
            mask_rgb = Image.new("RGB", (160, 160), (0, 0, 0))
            mask_rgb.paste(Image.merge("RGB", (mask_crop, mask_crop, mask_crop)))
            img.paste(mask_rgb, (1085, 320))
            draw.rectangle((1085, 320, 1245, 480), outline=(255, 255, 255), width=1)
            draw.text((1085, 484), "mask crop", fill=(255, 255, 255))
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"case{case_index:02d}_{label}_{sample.dataset.lower()}_{sample.item_key.replace('::', '_')}_obj{sample.object_id}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=180, loop=0)
    return {
        "case_index": case_index,
        "label": label,
        "dataset": sample.dataset,
        "item_key": sample.item_key,
        "object_id": sample.object_id,
        "gif_path": str(out.relative_to(ROOT)),
        "teacher_source": "cotracker_official",
        "has_raw_frames": True,
        "has_box_overlay": True,
        "has_mask_crop": mask_crop is not None,
        "has_dense_prediction": True,
        "has_teacher_future_tracks": True,
        "has_anchor_baseline_overlay": True,
    }


def _choose_cases(run_report: dict[str, Any], count: int) -> list[tuple[str, int]]:
    rows = sorted(run_report["item_scores"], key=lambda r: float(r["point_l1_px"]))
    keep: list[tuple[str, int]] = []
    seen = set()
    for block in [rows[: count // 2], rows[-(count - count // 2) :]]:
        for row in block:
            key = (str(row["item_key"]), int(row.get("object_id", row["object_index"])))
            if key not in seen:
                keep.append(key)
                seen.add(key)
    return keep[:count]


def main() -> int:
    eval_summary = load_json(ROOT / "reports/stwm_ostf_v17_eval_summary_20260502.json")
    if not eval_summary:
        raise SystemExit("missing V17 eval summary")
    run_m128 = load_json(ROOT / "reports/stwm_ostf_v17_runs/ostf_multitrace_m128_seed42_h8.json")
    run_m512 = load_json(ROOT / "reports/stwm_ostf_v17_runs/ostf_multitrace_m512_seed42_h8.json")
    rows_m128 = load_v16_samples("M128_H8")["test"]
    rows_m512 = load_v16_samples("M512_H8")["test"]
    row_map_m128 = {(s.item_key, s.object_id): s for s in rows_m128}
    row_map_m512 = {(s.item_key, s.object_id): s for s in rows_m512}
    model_m128, _ = _load_model(run_m128["checkpoint_path"])
    model_m512, _ = _load_model(run_m512["checkpoint_path"])
    model_m1_m128, _ = _load_model("outputs/checkpoints/stwm_ostf_v17/m1_anchor_stwm_m128_seed42_h8.pt")
    model_m1_m512, _ = _load_model("outputs/checkpoints/stwm_ostf_v17/m1_anchor_stwm_seed42_h8.pt")
    selected: list[tuple[str, str, int]] = []
    for item_key, object_id in _choose_cases(run_m128, 10):
        selected.append(("m128", item_key, object_id))
    for item_key, object_id in _choose_cases(run_m512, 10):
        selected.append(("m512", item_key, object_id))
    cases = []
    for idx, (label, item_key, object_id) in enumerate(selected):
        if label == "m128":
            sample = row_map_m128[(item_key, object_id)]
            dense_pred = _predict(model_m128, sample)
            anchor_pred = _predict(model_m1_m128, _collapse_single_to_m1(sample))
        else:
            sample = row_map_m512[(item_key, object_id)]
            dense_pred = _predict(model_m512, sample)
            anchor_pred = _predict(model_m1_m512, _collapse_single_to_m1(sample))
        cases.append(_render_case(idx, label, sample, dense_pred, anchor_pred))
    payload = {
        "audit_name": "stwm_ostf_v17_visualization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "visualization_ready": len(cases) >= 20,
        "gif_count": len(cases),
        "cases": cases,
        "teacher_source": "cotracker_official",
        "note": "Each GIF shows raw temporal frames, object box/mask crop, teacher future tracks, OSTF dense prediction, and matched M1 anchor baseline.",
    }
    _dump(REPORT_PATH, payload)
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
