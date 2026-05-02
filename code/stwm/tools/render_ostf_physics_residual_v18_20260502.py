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
from stwm.modules.ostf_physics_residual_world_model_v18 import OSTFPhysicsResidualConfig, OSTFPhysicsResidualWorldModel
from stwm.tools.ostf_v17_common_20260502 import ROOT, OSTFObjectSample, batch_from_samples, load_json, load_v16_samples
from stwm.tools.ostf_v18_common_20260502 import analytic_constant_velocity_predict


OUT_DIR = ROOT / "assets/videos/stwm_ostf_v18"
REPORT_PATH = ROOT / "reports/stwm_ostf_v18_visualization_20260502.json"

TEACHER_COLOR = (255, 255, 255)
CV_COLOR = (255, 196, 0)
V17_COLOR = (80, 220, 255)
V18_COLOR = (46, 204, 113)


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


def _load_v17_model(checkpoint_rel: str) -> OSTFMultiTraceWorldModel:
    ckpt = torch.load(ROOT / checkpoint_rel, map_location="cpu", weights_only=False)
    model = OSTFMultiTraceWorldModel(OSTFMultiTraceWorldModelConfig())
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_v18_model(checkpoint_rel: str, horizon: int) -> OSTFPhysicsResidualWorldModel:
    ckpt = torch.load(ROOT / checkpoint_rel, map_location="cpu", weights_only=False)
    model = OSTFPhysicsResidualWorldModel(
        OSTFPhysicsResidualConfig(
            horizon=horizon,
            hidden_dim=256,
            point_dim=128,
            num_layers=4,
            num_heads=8,
        )
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _predict_v17(model: OSTFMultiTraceWorldModel, sample: OSTFObjectSample) -> tuple[np.ndarray, np.ndarray]:
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
    return out["point_pred"][0].cpu().numpy(), out["visibility_logits"][0].cpu().numpy() > 0.0


def _predict_v18(model: OSTFPhysicsResidualWorldModel, sample: OSTFObjectSample) -> tuple[np.ndarray, np.ndarray]:
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
    return out["point_pred"][0].cpu().numpy(), out["visibility_logits"][0].cpu().numpy() > 0.0


def _cv_predict(sample: OSTFObjectSample) -> tuple[np.ndarray, np.ndarray]:
    pts, vis_logits, _ = analytic_constant_velocity_predict([sample], proto_count=32)
    return pts[0], vis_logits[0] > 0.0


def _bbox_xyxy(pre: np.lib.npyio.NpzFile, object_id: int, t: int) -> np.ndarray | None:
    if "entity_boxes_over_time" not in pre.files:
        return None
    boxes = np.asarray(pre["entity_boxes_over_time"], dtype=np.float32)
    if t >= boxes.shape[0] or object_id >= boxes.shape[1]:
        return None
    return boxes[t, object_id]


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    pts: np.ndarray,
    vis: np.ndarray,
    upto: int,
    color: tuple[int, int, int],
    width: int,
) -> None:
    hist = [(float(pts[k, 0]), float(pts[k, 1])) for k in range(min(upto + 1, len(pts))) if bool(vis[k]) and np.isfinite(pts[k]).all()]
    if len(hist) >= 2:
        draw.line(hist, fill=color, width=width)
    if hist:
        x, y = hist[-1]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def _choose_cases(scores: list[dict[str, Any]], count: int = 20) -> list[tuple[str, int]]:
    rows = sorted(scores, key=lambda r: float(r["extent_iou"]) - 0.002 * float(r["point_l1_px"]))
    best = rows[-10:]
    worst = rows[:10]
    keep: list[tuple[str, int]] = []
    seen = set()
    for group in [best[::-1], worst]:
        for row in group:
            key = (str(row["item_key"]), int(row["object_id"]))
            if key not in seen:
                keep.append(key)
                seen.add(key)
    return keep[:count]


def _render_case(
    case_index: int,
    sample: OSTFObjectSample,
    cv_pred: np.ndarray,
    cv_vis: np.ndarray,
    v17_pred: np.ndarray,
    v17_vis: np.ndarray,
    v18_pred: np.ndarray,
    v18_vis: np.ndarray,
) -> dict[str, Any]:
    cache_npz = ROOT / sample.source_cache_path
    z = np.load(cache_npz, allow_pickle=True)
    pre = np.load(str(z["predecode_path"].item()), allow_pickle=True)
    frame_paths = [str(x) for x in z["frame_paths"].tolist()]
    raw_size = np.asarray(z["raw_size"], dtype=np.float32)
    scale, ox, oy = _scale_meta(raw_size)
    object_id = int(sample.object_id)
    tracks_full = np.asarray(z["tracks_xy"], dtype=np.float32)[object_id]
    vis_full = np.asarray(z["visibility"]).astype(bool)[object_id]
    query = _transform(np.asarray(z["query_points_xy"], dtype=np.float32)[object_id], scale, ox, oy)
    teacher_tr = _transform(tracks_full, scale, ox, oy)
    cv_tr = _transform(np.concatenate([sample.obs_points, cv_pred], axis=1) * float(max(raw_size.tolist())), scale, ox, oy)
    v17_tr = _transform(np.concatenate([sample.obs_points, v17_pred], axis=1) * float(max(raw_size.tolist())), scale, ox, oy)
    v18_tr = _transform(np.concatenate([sample.obs_points, v18_pred], axis=1) * float(max(raw_size.tolist())), scale, ox, oy)
    cv_vis_full = np.concatenate([sample.obs_vis, cv_vis], axis=1)
    v17_vis_full = np.concatenate([sample.obs_vis, v17_vis], axis=1)
    v18_vis_full = np.concatenate([sample.obs_vis, v18_vis], axis=1)
    point_indices = np.linspace(0, teacher_tr.shape[0] - 1, num=min(64, teacher_tr.shape[0]), dtype=int)
    frames: list[Image.Image] = []
    obs_len = int(z["obs_len"].item())
    total_t = teacher_tr.shape[1]
    for t, frame_path in enumerate(frame_paths[:total_t]):
        img = _load_frame(frame_path)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 1280, 78), fill=(0, 0, 0))
        draw.text((16, 8), f"OSTF V18 case={case_index:02d} dataset={sample.dataset} item={sample.item_key} obj={sample.object_id} t={t}", fill=(255, 255, 255))
        draw.text((16, 30), "white=teacher  yellow=CV  cyan=V17  green=V18", fill=(255, 220, 160))
        draw.text((16, 52), "teacher_source=cotracker_official  observed-only input, future teacher only as target", fill=(180, 230, 255))
        bbox = _bbox_xyxy(pre, object_id, t)
        if bbox is not None and np.isfinite(bbox).all():
            bb = _transform(np.asarray(bbox[None], dtype=np.float32), scale, ox, oy)[0]
            x0, x1 = sorted([float(bb[0]), float(bb[2])])
            y0, y1 = sorted([float(bb[1]), float(bb[3])])
            draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 0), width=2)
        if t == int(z["query_frame"].item()):
            step = max(len(query) // 128, 1)
            for x, y in query[::step]:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 255))
        for pidx in point_indices:
            _draw_polyline(draw, teacher_tr[pidx], vis_full[pidx], t, TEACHER_COLOR, 2)
            if t >= obs_len:
                _draw_polyline(draw, cv_tr[pidx], cv_vis_full[pidx], t, CV_COLOR, 2)
                _draw_polyline(draw, v17_tr[pidx], v17_vis_full[pidx], t, V17_COLOR, 2)
                _draw_polyline(draw, v18_tr[pidx], v18_vis_full[pidx], t, V18_COLOR, 2)
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"case{case_index:02d}_{sample.dataset.lower()}_{sample.item_key.replace('::', '_')}_obj{sample.object_id}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=180, loop=0)
    return {
        "case_index": case_index,
        "dataset": sample.dataset,
        "item_key": sample.item_key,
        "object_id": sample.object_id,
        "gif_path": str(out.relative_to(ROOT)),
        "teacher_source": "cotracker_official",
        "has_raw_frames": True,
        "has_teacher_tracks": True,
        "has_constant_velocity": True,
        "has_v17_prediction": True,
        "has_v18_prediction": True,
        "has_visibility_markers": True,
    }


def main() -> int:
    v18_report = load_json(ROOT / "reports/stwm_ostf_v18_runs/v18_physics_residual_m512_seed42_h8.json")
    v17_report = load_json(ROOT / "reports/stwm_ostf_v17_runs/ostf_multitrace_m512_seed42_h8.json")
    v18_model = _load_v18_model(v18_report["best_checkpoint_path"], horizon=8)
    v17_ckpt = v17_report.get("best_checkpoint_path") or v17_report["checkpoint_path"]
    v17_model = _load_v17_model(v17_ckpt)
    samples = load_v16_samples("M512_H8")["test"]
    sample_map = {(s.item_key, s.object_id): s for s in samples}
    cases = []
    for case_index, key in enumerate(_choose_cases(v18_report["item_scores"], count=20)):
        sample = sample_map[key]
        cv_pred, cv_vis = _cv_predict(sample)
        v17_pred, v17_vis = _predict_v17(v17_model, sample)
        v18_pred, v18_vis = _predict_v18(v18_model, sample)
        cases.append(_render_case(case_index, sample, cv_pred, cv_vis, v17_pred, v17_vis, v18_pred, v18_vis))
    payload = {
        "audit_name": "stwm_ostf_v18_visualization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "visualization_ready": len(cases) >= 20,
        "gif_count": len(cases),
        "cases": cases,
        "teacher_source": "cotracker_official",
        "note": "Each GIF shows raw temporal frames, teacher dense tracks, constant-velocity prior, V17 prediction, and V18 physics-prior residual prediction.",
    }
    _dump(REPORT_PATH, payload)
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
