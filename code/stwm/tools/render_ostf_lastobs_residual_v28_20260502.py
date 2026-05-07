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

from stwm.tools.ostf_lastobs_v28_common_20260502 import ROOT, build_v28_rows, predict_last
from stwm.tools.ostf_traceanything_common_v26_20260502 import analytic_constant_velocity_predict, batch_from_samples_v26
from stwm.tools.render_ostf_traceanything_v26_20260502 import (
    CV_COLOR,
    MODE_COLORS,
    ORACLE_COLOR,
    TEACHER_COLOR,
    TOP1_COLOR,
    _draw_polyline,
    _load_frame,
    _scale_meta,
    _transform,
)
from stwm.tools.train_ostf_lastobs_residual_v28_20260502 import _build_model


OUT_DIR = ROOT / "assets/videos/stwm_ostf_v28"
REPORT_PATH = ROOT / "reports/stwm_ostf_v28_visualization_20260502.json"
LAST_COLOR = (0, 220, 255)


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_model(run_report: dict[str, Any]) -> torch.nn.Module:
    ckpt = torch.load(ROOT / run_report["best_checkpoint_path"], map_location="cpu", weights_only=False)
    model = _build_model(run_report["model_kind"], int(run_report["source_combo"].split("_H")[-1]), float(run_report.get("damped_gamma", 0.0)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _choose_cases(run_report: dict[str, Any], count: int = 30) -> list[tuple[str, int]]:
    rows = run_report["item_scores"]
    pools = [
        [r for r in rows if r.get("last_observed_hard_top20")],
        [r for r in rows if r.get("occlusion_reappearance")],
        [r for r in rows if r.get("nonlinear_displacement_hard_top20")],
        [r for r in rows if r.get("semantic_identity_confuser")],
        [r for r in rows if not r.get("last_observed_hard_top20")],
    ]
    ordered = []
    for pool in pools:
        ordered.extend(sorted(pool, key=lambda r: float(r.get("minFDE_K_px", 0.0)), reverse=True)[:12])
    keep: list[tuple[str, int]] = []
    seen = set()
    for row in ordered:
        key = (str(row["item_key"]), int(row["object_id"]))
        if key in seen:
            continue
        keep.append(key)
        seen.add(key)
        if len(keep) >= count:
            break
    return keep


def _predict(model: torch.nn.Module, sample: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = batch_from_samples_v26([sample], torch.device("cpu"))
    with torch.no_grad():
        out = model(
            obs_points=batch["obs_points"],
            obs_vis=batch["obs_vis"],
            obs_conf=batch["obs_conf"],
            rel_xy=batch["rel_xy"],
            anchor_obs=batch["anchor_obs"],
            anchor_obs_vel=batch["anchor_obs_vel"],
            semantic_feat=batch["semantic_feat"],
            semantic_id=batch["semantic_id"],
            box_feat=batch["box_feat"],
            neighbor_feat=batch["neighbor_feat"],
            global_feat=batch["global_feat"],
            tusb_token=batch["tusb_token"],
        )
    return (
        out["point_pred"][0].cpu().numpy(),
        out["top1_point_pred"][0].cpu().numpy(),
        out["point_hypotheses"][0].cpu().numpy(),
        out["hypothesis_logits"][0].cpu().numpy(),
        out["visibility_logits"][0].cpu().numpy() > 0.0,
    )


def _render_case(case_index: int, sample: Any, cache_npz: np.lib.npyio.NpzFile, model: torch.nn.Module, *, run_name: str) -> dict[str, Any]:
    frame_paths = [str(x) for x in cache_npz["frame_paths"].tolist()]
    raw_size = np.asarray(cache_npz["raw_size"], dtype=np.float32)
    scale, ox, oy = _scale_meta(raw_size)
    object_id = int(sample.object_index)
    tracks_full = np.asarray(cache_npz["tracks_xy"], dtype=np.float32)[object_id]
    vis_full = np.asarray(cache_npz["visibility"]).astype(bool)[object_id]
    raw_scale = float(max(raw_size.tolist()))
    obs_len = int(cache_npz["obs_len"].item())
    total_t = tracks_full.shape[1]
    cv_pred, cv_vis, _ = analytic_constant_velocity_predict([sample], proto_count=32, proto_centers=None, semantic_mode="observed_memory")
    last_pred = predict_last([sample])[0]
    weighted_pred, top1_pred, hyp, hyp_logits, pred_vis = _predict(model, sample)
    err = np.abs(hyp - sample.fut_points[:, :, None, :]).sum(axis=-1) * 1000.0
    valid = sample.fut_vis
    endpoint_k = err[:, -1, :][valid[:, -1]].mean(axis=0) if np.any(valid[:, -1]) else err.mean(axis=(0, 1))
    best_idx = int(np.argmin(endpoint_k))
    topk = np.argsort(hyp_logits)[-3:][::-1].tolist()
    teacher_tr = _transform(tracks_full, scale, ox, oy)
    last_tr = _transform(np.concatenate([sample.obs_points, last_pred], axis=1) * raw_scale, scale, ox, oy)
    cv_tr = _transform(np.concatenate([sample.obs_points, cv_pred[0]], axis=1) * raw_scale, scale, ox, oy)
    top1_tr = _transform(np.concatenate([sample.obs_points, top1_pred], axis=1) * raw_scale, scale, ox, oy)
    oracle_tr = _transform(np.concatenate([sample.obs_points, hyp[:, :, best_idx]], axis=1) * raw_scale, scale, ox, oy)
    mode_tr = [_transform(np.concatenate([sample.obs_points, hyp[:, :, idx]], axis=1) * raw_scale, scale, ox, oy) for idx in topk[:3]]
    last_vis_full = np.concatenate([sample.obs_vis, np.repeat(sample.obs_vis[:, -1:], sample.h, axis=1)], axis=1)
    cv_vis_full = np.concatenate([sample.obs_vis, cv_vis[0] > 0.0], axis=1)
    pred_vis_full = np.concatenate([sample.obs_vis, pred_vis], axis=1)
    point_indices = np.linspace(0, teacher_tr.shape[0] - 1, num=min(64, teacher_tr.shape[0]), dtype=int)
    frames = []
    for t, frame_path in enumerate(frame_paths[:total_t]):
        img = _load_frame(frame_path)
        draw = ImageDraw.Draw(img)
        draw.rectangle((0, 0, 1280, 96), fill=(0, 0, 0))
        draw.text((16, 6), f"OSTF V28 case={case_index:02d} run={run_name} dataset={sample.dataset} item={sample.item_key} obj={sample.object_id} t={t}", fill=(255, 255, 255))
        draw.text((16, 29), "white=TraceAnything teacher cyan=last-observed yellow=CV green=V28 top1 red=oracle best mode", fill=(255, 220, 160))
        draw.text((16, 52), f"semantic_id={sample.semantic_id} top_modes={topk} oracle_best={best_idx} teacher-target full clip; model input observed-only", fill=(180, 230, 255))
        draw.text((16, 75), "V28 uses last-observed/damped prior modes; success is judged against last-observed, not CV.", fill=(200, 200, 200))
        for pidx in point_indices:
            _draw_polyline(draw, teacher_tr[pidx], vis_full[pidx], t, TEACHER_COLOR, 2)
            if t >= obs_len:
                _draw_polyline(draw, last_tr[pidx], last_vis_full[pidx], t, LAST_COLOR, 2)
                _draw_polyline(draw, cv_tr[pidx], cv_vis_full[pidx], t, CV_COLOR, 1)
                _draw_polyline(draw, top1_tr[pidx], pred_vis_full[pidx], t, TOP1_COLOR, 2)
                _draw_polyline(draw, oracle_tr[pidx], pred_vis_full[pidx], t, ORACLE_COLOR, 1)
                for rank, tr in enumerate(mode_tr):
                    _draw_polyline(draw, tr[pidx], pred_vis_full[pidx], t, MODE_COLORS[rank], 1)
        frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"case{case_index:02d}_{sample.dataset.lower()}_{sample.item_key.replace('::', '_')}_obj{sample.object_id}_{run_name}.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=140, loop=0)
    return {
        "case_index": case_index,
        "gif_path": str(out.relative_to(ROOT)),
        "dataset": sample.dataset,
        "item_key": sample.item_key,
        "object_id": int(sample.object_id),
        "run_name": run_name,
    }


def main() -> int:
    preferred = [
        "v28_lastobs_m128_h32_seed42",
        "v28_lastobs_m512_h32_seed42",
        "v28_lastobs_m128_h64_seed42",
    ]
    rendered = []
    case_index = 0
    for run_name in preferred:
        report_path = ROOT / "reports/stwm_ostf_v28_runs" / f"{run_name}.json"
        if not report_path.exists():
            continue
        run_report = json.loads(report_path.read_text(encoding="utf-8"))
        rows, _, _, _ = build_v28_rows(run_report["source_combo"], seed=42)
        sample_map = {(s.item_key, int(s.object_id)): s for s in rows["test"]}
        model = _load_model(run_report)
        per_run_count = 10 if run_name != preferred[0] else 30
        for key in _choose_cases(run_report, count=per_run_count):
            sample = sample_map.get(key)
            if sample is None:
                continue
            cache_npz = np.load(ROOT / sample.source_cache_path, allow_pickle=True)
            rendered.append(_render_case(case_index, sample, cache_npz, model, run_name=run_name))
            case_index += 1
    payload = {
        "audit_name": "stwm_ostf_v28_visualization",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(OUT_DIR.relative_to(ROOT)),
        "generated_gif_count": len(rendered),
        "visualization_ready": len(rendered) >= 30,
        "cases": rendered,
    }
    _dump(REPORT_PATH, payload)
    print(REPORT_PATH.relative_to(ROOT))
    return 0 if rendered else 1


if __name__ == "__main__":
    raise SystemExit(main())
