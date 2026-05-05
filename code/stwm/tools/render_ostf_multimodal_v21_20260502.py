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

from stwm.modules.ostf_multimodal_world_model_v21 import OSTFMultimodalWorldModel
from stwm.tools.ostf_v17_common_20260502 import ROOT, OSTFObjectSample, batch_from_samples, load_json, load_v16_samples
from stwm.tools.ostf_v18_common_20260502 import analytic_constant_velocity_predict
from stwm.tools.ostf_v20_common_20260502 import load_context_cache, sample_key
from stwm.tools.train_ostf_context_residual_v20_20260502 import _build_model as _build_v20_model
from stwm.tools.train_ostf_multimodal_v21_20260502 import _build_model as _build_v21_model


OUT_DIR = ROOT / "assets/videos/stwm_ostf_v21"
REPORT_PATH = ROOT / "reports/stwm_ostf_v21_visualization_20260502.json"
TEACHER_COLOR = (255, 255, 255)
CV_COLOR = (255, 196, 0)
V20_COLOR = (80, 220, 255)
V21_WEIGHTED_COLOR = (46, 204, 113)
V21_MODE_A = (255, 99, 132)
V21_MODE_B = (155, 89, 182)
V21_MODE_C = (52, 152, 219)


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


def _context_batch(sample: OSTFObjectSample, ctx_map: dict[tuple[str, int], dict[str, Any]]) -> dict[str, torch.Tensor]:
    ctx = ctx_map[sample_key(sample)]
    return {
        "crop_feat": torch.tensor(ctx["crop_feat"][None], dtype=torch.float32),
        "box_feat": torch.tensor(ctx["box_feat"][None], dtype=torch.float32),
        "neighbor_feat": torch.tensor(ctx["neighbor_feat"][None], dtype=torch.float32),
        "global_feat": torch.tensor(ctx["global_feat"][None], dtype=torch.float32),
    }


def _load_v20_model(run_report: dict[str, Any]) -> torch.nn.Module:
    ckpt = torch.load(ROOT / run_report["best_checkpoint_path"], map_location="cpu", weights_only=False)
    model = _build_v20_model(run_report["model_kind"], int(run_report["horizon"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _load_v21_model(run_report: dict[str, Any]) -> OSTFMultimodalWorldModel:
    ckpt = torch.load(ROOT / run_report["best_checkpoint_path"], map_location="cpu", weights_only=False)
    model = _build_v21_model(run_report["model_kind"], int(run_report["horizon"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _predict_v20(model: torch.nn.Module, sample: OSTFObjectSample, ctx_map: dict[tuple[str, int], dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    batch = batch_from_samples([sample], torch.device("cpu"))
    batch_ctx = _context_batch(sample, ctx_map)
    with torch.no_grad():
        out = model(
            obs_points=batch["obs_points"],
            obs_vis=batch["obs_vis"],
            rel_xy=batch["rel_xy"],
            anchor_obs=batch["anchor_obs"],
            anchor_obs_vel=batch["anchor_obs_vel"],
            semantic_feat=batch["semantic_feat"],
            crop_feat=batch_ctx["crop_feat"],
            box_feat=batch_ctx["box_feat"],
            neighbor_feat=batch_ctx["neighbor_feat"],
            global_feat=batch_ctx["global_feat"],
        )
    return out["point_pred"][0].cpu().numpy(), out["visibility_logits"][0].cpu().numpy() > 0.0


def _predict_v21(
    model: OSTFMultimodalWorldModel,
    sample: OSTFObjectSample,
    ctx_map: dict[tuple[str, int], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = batch_from_samples([sample], torch.device("cpu"))
    batch_ctx = _context_batch(sample, ctx_map)
    with torch.no_grad():
        out = model(
            obs_points=batch["obs_points"],
            obs_vis=batch["obs_vis"],
            rel_xy=batch["rel_xy"],
            anchor_obs=batch["anchor_obs"],
            anchor_obs_vel=batch["anchor_obs_vel"],
            semantic_feat=batch["semantic_feat"],
            crop_feat=batch_ctx["crop_feat"],
            box_feat=batch_ctx["box_feat"],
            neighbor_feat=batch_ctx["neighbor_feat"],
            global_feat=batch_ctx["global_feat"],
        )
    return (
        out["point_pred"][0].cpu().numpy(),
        out["point_hypotheses"][0].cpu().numpy(),
        out["hypothesis_logits"][0].cpu().numpy(),
        out["visibility_logits"][0].cpu().numpy() > 0.0,
    )


def _cv_predict(sample: OSTFObjectSample) -> tuple[np.ndarray, np.ndarray]:
    pts, vis_logits, _ = analytic_constant_velocity_predict([sample], proto_count=32, semantic_mode="observed_memory")
    return pts[0], vis_logits[0] > 0.0


def _draw_polyline(draw: ImageDraw.ImageDraw, pts: np.ndarray, vis: np.ndarray, upto: int, color: tuple[int, int, int], width: int) -> None:
    hist = [(float(pts[k, 0]), float(pts[k, 1])) for k in range(min(upto + 1, len(pts))) if bool(vis[k]) and np.isfinite(pts[k]).all()]
    if len(hist) >= 2:
        draw.line(hist, fill=color, width=width)
    if hist:
        x, y = hist[-1]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def _choose_cases(run_report: dict[str, Any], count: int = 20) -> list[tuple[str, int]]:
    rows = run_report["item_scores"]
    hard = [r for r in rows if r.get("top20_cv_hard")]
    easy = [r for r in rows if not r.get("top20_cv_hard")]
    occl = [r for r in rows if r.get("occlusion_hard")]
    inter = [r for r in rows if r.get("interaction_hard")]
    nonlinear = [r for r in rows if r.get("nonlinear_hard")]
    pool = (
        sorted(hard, key=lambda r: float(r["minFDE_K_px"]), reverse=True)[:6]
        + sorted(occl, key=lambda r: float(r["minFDE_K_px"]), reverse=True)[:4]
        + sorted(inter, key=lambda r: float(r["minFDE_K_px"]), reverse=True)[:4]
        + sorted(nonlinear, key=lambda r: float(r["minFDE_K_px"]), reverse=True)[:4]
        + sorted(easy, key=lambda r: float(r["weighted_endpoint_error_px"]))[:6]
    )
    keep: list[tuple[str, int]] = []
    seen = set()
    for row in pool:
        key = (str(row["item_key"]), int(row["object_id"]))
        if key not in seen:
            keep.append(key)
            seen.add(key)
        if len(keep) >= count:
            break
    return keep


def _render_case(
    case_index: int,
    sample: OSTFObjectSample,
    cv_pred: np.ndarray,
    cv_vis: np.ndarray,
    v20_pred: np.ndarray,
    v20_vis: np.ndarray,
    v21_pred: np.ndarray,
    v21_modes: np.ndarray,
    v21_mode_logits: np.ndarray,
    v21_vis: np.ndarray,
    *,
    best_variant_name: str,
) -> dict[str, Any]:
    cache_npz = ROOT / sample.source_cache_path
    z = np.load(cache_npz, allow_pickle=True)
    frame_paths = [str(x) for x in z["frame_paths"].tolist()]
    raw_size = np.asarray(z["raw_size"], dtype=np.float32)
    scale, ox, oy = _scale_meta(raw_size)
    object_id = int(sample.object_index)
    tracks_full = np.asarray(z["tracks_xy"], dtype=np.float32)[object_id]
    vis_full = np.asarray(z["visibility"]).astype(bool)[object_id]
    query = _transform(np.asarray(z["query_points_xy"], dtype=np.float32)[object_id], scale, ox, oy)
    teacher_tr = _transform(tracks_full, scale, ox, oy)
    scale_raw = float(max(raw_size.tolist()))
    cv_tr = _transform(np.concatenate([sample.obs_points, cv_pred], axis=1) * scale_raw, scale, ox, oy)
    v20_tr = _transform(np.concatenate([sample.obs_points, v20_pred], axis=1) * scale_raw, scale, ox, oy)
    v21_tr = _transform(np.concatenate([sample.obs_points, v21_pred], axis=1) * scale_raw, scale, ox, oy)
    mode_tr = [
        _transform(np.concatenate([sample.obs_points, v21_modes[:, :, idx]], axis=1) * scale_raw, scale, ox, oy)
        for idx in range(v21_modes.shape[2])
    ]
    mode_err = []
    for idx in range(v21_modes.shape[2]):
        err = np.abs(v21_modes[:, :, idx] - sample.fut_points).sum(axis=-1) * 1000.0
        valid = sample.fut_vis
        mode_err.append(float(err[valid].mean()) if np.any(valid) else 0.0)
    best_idx = int(np.argmin(mode_err))
    topk = np.argsort(v21_mode_logits)[-3:][::-1].tolist()
    cv_vis_full = np.concatenate([sample.obs_vis, cv_vis], axis=1)
    v20_vis_full = np.concatenate([sample.obs_vis, v20_vis], axis=1)
    v21_vis_full = np.concatenate([sample.obs_vis, v21_vis], axis=1)
    point_indices = np.linspace(0, teacher_tr.shape[0] - 1, num=min(64, teacher_tr.shape[0]), dtype=int)
    frames: list[Image.Image] = []
    obs_len = int(z["obs_len"].item())
    total_t = teacher_tr.shape[1]
    colors = [V21_MODE_A, V21_MODE_B, V21_MODE_C]
    for t, frame_path in enumerate(frame_paths[:total_t]):
        img = _load_frame(frame_path)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 1280, 86), fill=(0, 0, 0))
        draw.text((16, 6), f"OSTF V21 case={case_index:02d} dataset={sample.dataset} item={sample.item_key} obj={sample.object_id} t={t}", fill=(255, 255, 255))
        draw.text((16, 28), "white=teacher yellow=CV cyan=V20 green=V21-weighted magenta/purple/blue=top hypotheses", fill=(255, 220, 160))
        draw.text((16, 50), f"best_hypothesis={best_idx}  logits_top3={topk}  teacher_source=cotracker_official", fill=(180, 230, 255))
        if t == int(z["query_frame"].item()):
            step = max(len(query) // 128, 1)
            for x, y in query[::step]:
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 0, 255))
        for pidx in point_indices:
            _draw_polyline(draw, teacher_tr[pidx], vis_full[pidx], t, TEACHER_COLOR, 2)
            if t >= obs_len:
                _draw_polyline(draw, cv_tr[pidx], cv_vis_full[pidx], t, CV_COLOR, 2)
                _draw_polyline(draw, v20_tr[pidx], v20_vis_full[pidx], t, V20_COLOR, 2)
                _draw_polyline(draw, v21_tr[pidx], v21_vis_full[pidx], t, V21_WEIGHTED_COLOR, 2)
                for rank, mode_idx in enumerate(topk[:3]):
                    _draw_polyline(draw, mode_tr[mode_idx][pidx], v21_vis_full[pidx], t, colors[rank], 1)
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
        "best_hypothesis_idx": best_idx,
        "top_logit_modes": topk,
        "has_raw_frames": True,
        "has_teacher_tracks": True,
        "has_constant_velocity": True,
        "has_v20_prediction": True,
        "has_v21_weighted_prediction": True,
        "has_v21_multimodal_hypotheses": True,
        "has_visibility_markers": True,
    }


def main() -> int:
    decision = load_json(ROOT / "reports/stwm_ostf_v21_decision_20260502.json")
    best_name = str(decision["best_variant_name"])
    run_report = load_json(ROOT / f"reports/stwm_ostf_v21_runs/{best_name}.json")
    combo = str(run_report["source_combo"])
    horizon = int(run_report["horizon"])
    ctx_map = load_context_cache(ROOT / run_report["context_cache_path"])
    v20_name = "v20_context_residual_m128_seed42_h8" if "m128" in best_name else "v20_context_residual_m512_seed42_h8"
    v20_report = load_json(ROOT / f"reports/stwm_ostf_v20_runs/{v20_name}.json")
    v20_ctx = load_context_cache(ROOT / v20_report["context_cache_path"])
    v20_model = _load_v20_model(v20_report)
    v21_model = _load_v21_model(run_report)
    samples = load_v16_samples(combo)["test"]
    sample_map = {(s.item_key, s.object_id): s for s in samples}
    cases = []
    for case_index, key in enumerate(_choose_cases(run_report, count=20)):
        sample = sample_map[key]
        cv_pred, cv_vis = _cv_predict(sample)
        v20_pred, v20_vis = _predict_v20(v20_model, sample, v20_ctx)
        v21_pred, v21_modes, v21_logits, v21_vis = _predict_v21(v21_model, sample, ctx_map)
        cases.append(
            _render_case(
                case_index,
                sample,
                cv_pred,
                cv_vis,
                v20_pred,
                v20_vis,
                v21_pred,
                v21_modes,
                v21_logits,
                v21_vis,
                best_variant_name=best_name,
            )
        )
    payload = {
        "audit_name": "stwm_ostf_v21_visualization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
        "source_combo": combo,
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "visualization_ready": len(cases) >= 20,
        "gif_count": len(cases),
        "teacher_source": "cotracker_official",
        "cases": cases,
    }
    _dump(REPORT_PATH, payload)
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
