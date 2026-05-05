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

from stwm.tools.eval_ostf_calibrated_multimodal_v22_20260502 import _load_model
from stwm.tools.ostf_multimodal_metrics_v22 import multimodal_item_scores_v22
from stwm.tools.ostf_v17_common_20260502 import ROOT, OSTFObjectSample, load_json
from stwm.tools.ostf_v18_common_20260502 import analytic_constant_velocity_predict
from stwm.tools.ostf_v20_common_20260502 import load_context_cache, sample_key
from stwm.tools.train_ostf_calibrated_multimodal_v22_20260502 import (
    batch_from_samples_v22,
    context_batch,
    prepare_rows_for_model,
    subset_flags,
)
from stwm.tools.train_ostf_multimodal_v21_20260502 import _build_model as _build_v21_model


OUT_DIR = ROOT / "assets/videos/stwm_ostf_v22"
REPORT_PATH = ROOT / "reports/stwm_ostf_v22_visualization_20260502.json"
TEACHER_COLOR = (255, 255, 255)
CV_COLOR = (255, 196, 0)
V21_COLOR = (80, 220, 255)
V22_TOP1_COLOR = (46, 204, 113)
V22_ORACLE_COLOR = (255, 99, 132)


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


def _draw_polyline(draw: ImageDraw.ImageDraw, pts: np.ndarray, vis: np.ndarray, upto: int, color: tuple[int, int, int], width: int) -> None:
    hist = [(float(pts[k, 0]), float(pts[k, 1])) for k in range(min(upto + 1, len(pts))) if bool(vis[k]) and np.isfinite(pts[k]).all()]
    if len(hist) >= 2:
        draw.line(hist, fill=color, width=width)
    if hist:
        x, y = hist[-1]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def _load_v21_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    report = load_json(ROOT / "reports/stwm_ostf_v21_runs/v21_multimodal_m128_seed42_h8.json")
    ckpt = torch.load(ROOT / report["best_checkpoint_path"], map_location="cpu", weights_only=False)
    model = _build_v21_model(report["model_kind"], int(report["horizon"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, report


def _predict_v21(model: torch.nn.Module, sample: OSTFObjectSample, ctx_map: dict[tuple[str, int], dict[str, Any]]) -> np.ndarray:
    batch = batch_from_samples_v22([sample], torch.device("cpu"))
    batch_ctx = context_batch([sample], ctx_map, torch.device("cpu"))
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
    return out["point_pred"][0].cpu().numpy()


def _predict_v22(
    model: torch.nn.Module,
    sample: OSTFObjectSample,
    ctx_map: dict[tuple[str, int], dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = batch_from_samples_v22([sample], torch.device("cpu"))
    batch_ctx = context_batch([sample], ctx_map, torch.device("cpu"))
    with torch.no_grad():
        out = model(
            obs_points=batch["obs_points"],
            obs_vis=batch["obs_vis"],
            rel_xy=batch["rel_xy"],
            point_meta=batch["point_meta"],
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
        out["top1_point_pred"][0].cpu().numpy(),
        out["point_hypotheses"][0].cpu().numpy(),
        out["hypothesis_logits"][0].cpu().numpy(),
    )


def _choose_cases(best_report: dict[str, Any], count: int = 20) -> list[tuple[str, int]]:
    rows = best_report["item_scores"]
    hard = [r for r in rows if r.get("top20_cv_hard")]
    occl = [r for r in rows if r.get("occlusion_hard")]
    inter = [r for r in rows if r.get("interaction_hard")]
    nonlinear = [r for r in rows if r.get("nonlinear_hard")]
    easy = [r for r in rows if not r.get("top20_cv_hard")]
    pool = (
        sorted(hard, key=lambda r: float(r["minFDE_K_px"]), reverse=True)[:6]
        + sorted(occl, key=lambda r: float(r["minFDE_K_px"]), reverse=True)[:4]
        + sorted(inter, key=lambda r: float(r["minFDE_K_px"]), reverse=True)[:4]
        + sorted(nonlinear, key=lambda r: float(r["minFDE_K_px"]), reverse=True)[:4]
        + sorted(easy, key=lambda r: float(r["top1_endpoint_error_px"]))[:6]
    )
    keep = []
    seen = set()
    for row in pool:
        key = (str(row["item_key"]), int(row["object_id"]))
        if key not in seen:
            keep.append(key)
            seen.add(key)
        if len(keep) >= count:
            break
    if len(keep) < count:
        fallback = sorted(
            rows,
            key=lambda r: (
                not bool(r.get("top20_cv_hard")),
                -float(r.get("minFDE_K_px", 0.0)),
                -float(r.get("top1_endpoint_error_px", 0.0)),
            ),
        )
        for row in fallback:
            key = (str(row["item_key"]), int(row["object_id"]))
            if key not in seen:
                keep.append(key)
                seen.add(key)
            if len(keep) >= count:
                break
    return keep


def main() -> int:
    decision = load_json(ROOT / "reports/stwm_ostf_v22_decision_20260502.json")
    if not decision:
        raise SystemExit("Missing reports/stwm_ostf_v22_decision_20260502.json")
    best_name = str(decision["best_variant_name"])
    best_report = load_json(ROOT / "reports/stwm_ostf_v22_runs" / f"{best_name}.json")
    if not best_report:
        raise SystemExit(f"Missing run report for {best_name}")

    best_model = _load_model(best_report, torch.device("cpu"))
    v21_model, _ = _load_v21_model()

    rows, _, _, _, _ = prepare_rows_for_model(best_report["model_kind"], int(best_report["horizon"]), int(best_report["seed"]))
    samples = rows["test"]
    sample_map = {(s.item_key, int(s.object_id)): s for s in samples}
    ctx_map = load_context_cache(ROOT / best_report["context_cache_path"])
    cases = _choose_cases(best_report, count=20)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rendered = []
    for idx, key in enumerate(cases):
        sample = sample_map.get(key)
        if sample is None:
            continue
        cache_npz = ROOT / sample.source_cache_path
        z = np.load(cache_npz, allow_pickle=True)
        frame_paths = [str(x) for x in z["frame_paths"].tolist()]
        raw_size = np.asarray(z["raw_size"], dtype=np.float32)
        scale, ox, oy = _scale_meta(raw_size)
        object_id = int(sample.object_index)
        tracks_full = np.asarray(z["tracks_xy"], dtype=np.float32)[object_id]
        vis_full = np.asarray(z["visibility"]).astype(bool)[object_id]
        scale_raw = float(max(raw_size.tolist()))
        obs_len = int(z["obs_len"].item())
        total_t = tracks_full.shape[1]

        cv_pred, cv_vis, _ = analytic_constant_velocity_predict([sample], proto_count=32, semantic_mode="observed_memory")
        v21_pred = _predict_v21(v21_model, sample, ctx_map)
        v22_weighted, v22_top1, v22_hyp, v22_logits = _predict_v22(best_model, sample, ctx_map)
        flags = subset_flags([sample], ctx_map)
        item_row = multimodal_item_scores_v22(
            [sample],
            point_modes=v22_hyp[None],
            mode_logits=v22_logits[None],
            point_pred=v22_weighted[None],
            top1_pred=v22_top1[None],
            pred_vis_logits=None,
            pred_proto_logits=None,
            subset_flags=flags,
            cv_mode_index=0,
        )[0]
        best_idx = int(item_row["best_mode_idx_FDE"])
        oracle_pred = v22_hyp[:, :, best_idx]

        teacher_tr = _transform(tracks_full, scale, ox, oy)
        teacher_vis = vis_full
        cv_tr = _transform(np.concatenate([sample.obs_points, cv_pred[0]], axis=1) * scale_raw, scale, ox, oy)
        v21_tr = _transform(np.concatenate([sample.obs_points, v21_pred], axis=1) * scale_raw, scale, ox, oy)
        v22_top1_tr = _transform(np.concatenate([sample.obs_points, v22_top1], axis=1) * scale_raw, scale, ox, oy)
        v22_oracle_tr = _transform(np.concatenate([sample.obs_points, oracle_pred], axis=1) * scale_raw, scale, ox, oy)
        cv_vis_full = np.concatenate([sample.obs_vis, cv_vis[0] > 0.0], axis=1)
        v21_vis_full = np.concatenate([sample.obs_vis, np.ones_like(sample.fut_vis, dtype=bool)], axis=1)
        v22_vis_full = np.concatenate([sample.obs_vis, np.ones_like(sample.fut_vis, dtype=bool)], axis=1)

        frames = []
        point_indices = np.linspace(0, teacher_tr.shape[0] - 1, num=min(64, teacher_tr.shape[0]), dtype=int)
        for t, frame_path in enumerate(frame_paths[:total_t]):
            img = _load_frame(frame_path)
            draw = ImageDraw.Draw(img)
            draw.rectangle((0, 0, 1280, 88), fill=(0, 0, 0))
            draw.text((14, 6), f"OSTF V22 case={idx:02d} dataset={sample.dataset} item={sample.item_key} obj={sample.object_id} t={t}", fill=(255, 255, 255))
            draw.text((14, 28), "white=teacher yellow=CV cyan=V21 green=V22-top1 red=V22-oracle-best", fill=(255, 220, 160))
            draw.text((14, 50), f"teacher_source=cotracker_official semantic_proto={sample.proto_target} top1_mode={int(np.argmax(v22_logits))} oracle_mode={best_idx}", fill=(180, 230, 255))
            for pidx in point_indices:
                _draw_polyline(draw, teacher_tr[pidx], teacher_vis[pidx], t, TEACHER_COLOR, 2)
                if t >= obs_len:
                    _draw_polyline(draw, cv_tr[pidx], cv_vis_full[pidx], t, CV_COLOR, 2)
                    _draw_polyline(draw, v21_tr[pidx], v21_vis_full[pidx], t, V21_COLOR, 2)
                    _draw_polyline(draw, v22_top1_tr[pidx], v22_vis_full[pidx], t, V22_TOP1_COLOR, 2)
                    _draw_polyline(draw, v22_oracle_tr[pidx], v22_vis_full[pidx], t, V22_ORACLE_COLOR, 1)
            frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))

        out = OUT_DIR / f"case{idx:02d}_{sample.dataset.lower()}_{sample.item_key.replace('::', '_')}_obj{sample.object_id}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=160, loop=0)
        rendered.append(
            {
                "case_index": idx,
                "dataset": sample.dataset,
                "item_key": sample.item_key,
                "object_id": int(sample.object_id),
                "gif_path": str(out.relative_to(ROOT)),
                "top20_cv_hard": bool(item_row.get("top20_cv_hard")),
                "occlusion_hard": bool(item_row.get("occlusion_hard")),
                "interaction_hard": bool(item_row.get("interaction_hard")),
                "nonlinear_hard": bool(item_row.get("nonlinear_hard")),
                "top1_mode_idx": int(item_row["top1_mode_idx"]),
                "best_mode_idx_FDE": best_idx,
                "minFDE_K_px": float(item_row["minFDE_K_px"]),
                "top1_endpoint_error_px": float(item_row["top1_endpoint_error_px"]),
            }
        )

    report = {
        "audit_name": "stwm_ostf_v22_visualization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
        "rendered_case_count": len(rendered),
        "visualization_ready": bool(len(rendered) >= 20),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "cases": rendered,
    }
    _dump(REPORT_PATH, report)
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
