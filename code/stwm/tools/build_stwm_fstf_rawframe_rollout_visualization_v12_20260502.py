#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from stwm.tracewm_v2_stage2.trainers.train_tracewm_stage2_smalltrain import _free_rollout_predict, _to_device
from stwm.tracewm_v2_stage2.utils.future_semantic_feature_targets import stage2_item_key
from stwm.tracewm_v2_stage2.utils.future_semantic_prototype_targets import (
    load_future_semantic_prototype_target_cache,
    prototype_tensors_for_batch,
    semantic_change_tensors,
)
from stwm.tools.overfit_semantic_trace_field_one_batch_20260428 import (
    _batch_slot_count,
    _make_forward_kwargs,
)
from stwm.tools.run_semantic_memory_transition_residual_tiny_overfit_20260428 import (
    _load_observed,
    _observed_for_batch,
)
from stwm.tools.run_semantic_memory_world_model_v3_20260428 import _load_trained_models


def _set_process_title() -> None:
    try:
        import setproctitle  # type: ignore

        setproctitle.setproctitle(str(os.environ.get("STWM_PROC_TITLE", "python")))
    except Exception:
        pass


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _report_batch_path(report_path: Path) -> Path:
    payload = _load_json(report_path)
    for key in ("cache_path", "batch_cache_path", "output_cache"):
        value = payload.get(key)
        if value:
            p = Path(str(value))
            if p.exists():
                return p
    cache_dir = payload.get("cache_dir")
    if cache_dir:
        p = Path(str(cache_dir)) / "eval_batches.pt"
        if p.exists():
            return p
    raise FileNotFoundError(f"could not find batch cache path in {report_path}")


def _color_for_proto(proto: int) -> tuple[int, int, int]:
    if proto < 0:
        return (120, 120, 120)
    # Deterministic high-contrast palette from integer hash.
    x = (int(proto) * 1103515245 + 12345) & 0xFFFFFFFF
    return (64 + (x & 127), 64 + ((x >> 8) & 127), 64 + ((x >> 16) & 127))


def _safe_font(size: int = 20) -> ImageFont.ImageFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(p).exists():
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()


def _open_frame_sequence(current_path: str, horizon: int) -> list[Image.Image]:
    current = Path(current_path)
    if not current.exists():
        return [Image.new("RGB", (640, 360), (28, 30, 34)) for _ in range(horizon)]
    files = sorted([p for p in current.parent.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    try:
        idx = files.index(current)
    except ValueError:
        idx = 0
    frames: list[Image.Image] = []
    for step in range(horizon):
        src_idx = min(len(files) - 1, idx + step + 1)
        try:
            frames.append(Image.open(files[src_idx]).convert("RGB"))
        except Exception:
            frames.append(Image.open(current).convert("RGB"))
    return frames


def _resize_keep(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    out = Image.new("RGB", size, (18, 20, 22))
    scale = min(size[0] / img.width, size[1] / img.height)
    nw = max(1, int(img.width * scale))
    nh = max(1, int(img.height * scale))
    im = img.resize((nw, nh), Image.BILINEAR)
    out.paste(im, ((size[0] - nw) // 2, (size[1] - nh) // 2))
    return out


def _scale_box(box: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]) -> tuple[int, int, int, int]:
    sw, sh = src_size
    dw, dh = dst_size
    scale = min(dw / max(sw, 1), dh / max(sh, 1))
    nw, nh = sw * scale, sh * scale
    ox = (dw - nw) / 2.0
    oy = (dh - nh) / 2.0
    x1, y1, x2, y2 = [float(v) for v in box]
    return (
        int(round(ox + x1 * scale)),
        int(round(oy + y1 * scale)),
        int(round(ox + y1 * 0 + x2 * scale)),
        int(round(oy + y2 * scale)),
    )


def _draw_panel(
    img: Image.Image,
    title: str,
    boxes: np.ndarray,
    labels: np.ndarray,
    changed: np.ndarray,
    src_size: tuple[int, int],
    panel_size: tuple[int, int],
) -> Image.Image:
    font = _safe_font(18)
    small = _safe_font(14)
    panel = _resize_keep(img, panel_size)
    draw = ImageDraw.Draw(panel, "RGBA")
    draw.rectangle([0, 0, panel_size[0], 32], fill=(0, 0, 0, 160))
    draw.text((8, 6), title, font=font, fill=(255, 255, 255, 255))
    for slot, box in enumerate(boxes):
        if not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = _scale_box(box, src_size, panel_size)
        if x2 <= x1 or y2 <= y1:
            continue
        label = int(labels[slot]) if slot < len(labels) else -1
        color = _color_for_proto(label)
        width = 4 if bool(changed[slot]) else 2
        draw.rectangle([x1, y1, x2, y2], outline=(*color, 230), width=width)
        tag = f"{slot}:p{label}"
        if bool(changed[slot]):
            tag += "*"
        tw = int(draw.textlength(tag, font=small)) + 6
        draw.rectangle([x1, max(34, y1 - 18), x1 + tw, max(50, y1)], fill=(*color, 210))
        draw.text((x1 + 3, max(34, y1 - 17)), tag, font=small, fill=(255, 255, 255, 255))
    return panel


def _copy_logits(obs_dist: torch.Tensor, horizon: int) -> torch.Tensor:
    return torch.log(obs_dist.clamp_min(1e-6))[:, None].expand(-1, int(horizon), -1, -1)


def _per_item_top5(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    valid = mask.bool() & (target >= 0)
    if not bool(valid.any().item()):
        return float("nan")
    k = min(5, logits.shape[-1])
    pred = logits.topk(k, dim=-1).indices
    hit = (pred == target.unsqueeze(-1)).any(dim=-1)
    return float(hit[valid].float().mean().detach().cpu().item())


def _evaluate_items(
    *,
    batches: list[dict[str, Any]],
    checkpoint_path: Path,
    prototype_count: int,
    future_cache_report: Path,
    observed_report: Path,
    device: torch.device,
    max_batches: int,
) -> list[dict[str, Any]]:
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_args = dict(checkpoint_payload.get("args", {}))
    future_cache = load_future_semantic_prototype_target_cache(future_cache_report)
    obs_data = _load_observed(observed_report, prototype_count)
    args, models = _load_trained_models(
        checkpoint_path=checkpoint_path,
        prototype_count=prototype_count,
        payload=checkpoint_payload,
        checkpoint_args=checkpoint_args,
        device=device,
        residual_scale=float(checkpoint_payload.get("residual_scale", 0.25)),
    )
    head = models["future_semantic_state_head"]
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch_cpu in enumerate(batches[:max_batches]):
            batch = _to_device(batch_cpu, device, non_blocking=False)
            target, _dist, future_mask, _ = prototype_tensors_for_batch(
                future_cache,
                batch,
                horizon=int(getattr(args, "fut_len", 8)),
                slot_count=_batch_slot_count(batch),
                device=device,
            )
            obs_target, obs_dist, obs_mask = _observed_for_batch(obs_data, batch, device)
            change_target, change_mask, _event_target, _event_mask, _info = semantic_change_tensors(
                future_proto_target=target,
                future_proto_mask=future_mask,
                observed_proto_target=obs_target,
                observed_proto_mask=obs_mask,
            )
            kwargs = _make_forward_kwargs(models, args, batch)
            kwargs["future_semantic_state_head"] = None
            out = _free_rollout_predict(
                **kwargs,
                fut_len=int(getattr(args, "fut_len", 8)),
                observed_semantic_proto_target=None,
                observed_semantic_proto_distribution=None,
                observed_semantic_proto_mask=None,
            )
            state = head(
                out["future_hidden"],
                future_trace_coord=out["pred_coord"],
                observed_semantic_proto_target=obs_target,
                observed_semantic_proto_distribution=obs_dist,
                observed_semantic_proto_mask=obs_mask,
            )
            logits = state.future_semantic_proto_logits
            copy = _copy_logits(obs_dist, int(target.shape[1]))
            pred_label = logits.argmax(dim=-1).detach().cpu()
            copy_label = copy.argmax(dim=-1).detach().cpu()
            target_cpu = target.detach().cpu()
            change_cpu = change_target.detach().cpu()
            change_mask_cpu = change_mask.detach().cpu()
            for item_idx, meta in enumerate(batch_cpu["meta"]):
                overall_mask = change_mask[item_idx]
                changed_mask = change_mask[item_idx] & change_target[item_idx]
                stable_mask = change_mask[item_idx] & (~change_target[item_idx])
                records.append(
                    {
                        "batch_index": batch_idx,
                        "item_index": item_idx,
                        "item_key": stage2_item_key(meta),
                        "dataset": str(meta.get("dataset", "unknown")),
                        "clip_id": str(meta.get("clip_id", "")),
                        "frame_path": str(batch_cpu["semantic_frame_paths"][item_idx]),
                        "boxes": batch_cpu["entity_boxes_over_time"][item_idx].detach().cpu(),
                        "pred_label": pred_label[item_idx],
                        "copy_label": copy_label[item_idx],
                        "target": target_cpu[item_idx],
                        "changed": change_cpu[item_idx] & change_mask_cpu[item_idx],
                        "stwm_overall_top5": _per_item_top5(logits[item_idx], target[item_idx], overall_mask),
                        "copy_overall_top5": _per_item_top5(copy[item_idx], target[item_idx], overall_mask),
                        "stwm_changed_top5": _per_item_top5(logits[item_idx], target[item_idx], changed_mask),
                        "copy_changed_top5": _per_item_top5(copy[item_idx], target[item_idx], changed_mask),
                        "stwm_stable_top5": _per_item_top5(logits[item_idx], target[item_idx], stable_mask),
                        "copy_stable_top5": _per_item_top5(copy[item_idx], target[item_idx], stable_mask),
                        "changed_count": int(changed_mask.detach().sum().cpu().item()),
                    }
                )
    return records


def _pick_cases(records: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    def finite(v: Any) -> float:
        try:
            f = float(v)
        except Exception:
            return float("nan")
        return f

    chosen: list[tuple[str, dict[str, Any]]] = []

    def add(name: str, candidates: list[dict[str, Any]]) -> None:
        used = {id(r) for _, r in chosen}
        for r in candidates:
            if id(r) not in used:
                chosen.append((name, r))
                return

    changed_success = sorted(
        [r for r in records if r["changed_count"] > 0 and finite(r["stwm_changed_top5"]) >= finite(r["copy_changed_top5"])],
        key=lambda r: (finite(r["stwm_changed_top5"]) - finite(r["copy_changed_top5"]), r["changed_count"]),
        reverse=True,
    )
    add("copy_failure_fixed_by_STWM", [r for r in changed_success if finite(r["copy_changed_top5"]) < 0.99])
    add("VSPW_changed_success", [r for r in changed_success if r["dataset"].lower() == "vspw"])
    add("VIPSeg_changed_success", [r for r in changed_success if "vip" in r["dataset"].lower()])
    stable = sorted(
        [r for r in records if not math.isnan(finite(r["stwm_stable_top5"]))],
        key=lambda r: finite(r["stwm_stable_top5"]),
        reverse=True,
    )
    add("stable_preservation_success", stable)
    failure = sorted(
        [r for r in records if r["changed_count"] > 0],
        key=lambda r: finite(r["stwm_changed_top5"]) - finite(r["copy_changed_top5"]),
    )
    add("STWM_failure_case", failure)
    add("domain_shift_or_VIPSeg_failure_case", [r for r in failure if "vip" in r["dataset"].lower()])
    return chosen[:6]


def _render_case(
    *,
    case_name: str,
    record: dict[str, Any],
    figure_dir: Path,
    video_dir: Path,
    horizon: int,
) -> dict[str, Any]:
    frame_path = str(record["frame_path"])
    raw_frames = _open_frame_sequence(frame_path, horizon)
    boxes_all = record["boxes"].numpy()
    if boxes_all.shape[0] >= horizon * 2:
        future_boxes = boxes_all[-horizon:]
    else:
        future_boxes = np.repeat(boxes_all[-1:], horizon, axis=0)
    pred = record["pred_label"].numpy()
    copy = record["copy_label"].numpy()
    target = record["target"].numpy()
    changed = record["changed"].numpy().astype(bool)
    panel_size = (480, 270)
    frames: list[Image.Image] = []
    font = _safe_font(20)
    for t in range(horizon):
        img = raw_frames[min(t, len(raw_frames) - 1)]
        src_size = img.size
        panels = [
            _draw_panel(img, f"GT future t+{t+1}", future_boxes[t], target[t], changed[t], src_size, panel_size),
            _draw_panel(img, f"copy baseline t+{t+1}", future_boxes[t], copy[t], changed[t], src_size, panel_size),
            _draw_panel(img, f"STWM prediction t+{t+1}", future_boxes[t], pred[t], changed[t], src_size, panel_size),
        ]
        canvas = Image.new("RGB", (panel_size[0] * 3, panel_size[1] + 80), (245, 241, 232))
        for i, p in enumerate(panels):
            canvas.paste(p, (i * panel_size[0], 60))
        draw = ImageDraw.Draw(canvas)
        title = (
            f"{case_name} | {record['dataset']}::{record['clip_id']} | "
            f"changed_top5 STWM={record['stwm_changed_top5']:.3f} copy={record['copy_changed_top5']:.3f}"
        )
        draw.text((16, 16), title, font=font, fill=(20, 28, 24))
        frames.append(canvas)
    safe_name = case_name.replace("/", "_")
    gif_path = video_dir / f"{safe_name}.gif"
    png_path = figure_dir / f"{safe_name}_strip.png"
    pdf_path = figure_dir / f"{safe_name}_strip.pdf"
    svg_path = figure_dir / f"{safe_name}_summary.svg"
    video_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=450, loop=0)
    strip = Image.new("RGB", (frames[0].width, min(3, len(frames)) * frames[0].height), (245, 241, 232))
    for i, f in enumerate(frames[:3]):
        strip.paste(f, (0, i * frames[0].height))
    strip.save(png_path)
    strip.save(pdf_path)
    svg_path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="220">
<rect width="1200" height="220" fill="#f5f1e8"/>
<text x="24" y="42" font-family="monospace" font-size="24" font-weight="700">{case_name}</text>
<text x="24" y="82" font-family="monospace" font-size="18">item_key={record['item_key']}</text>
<text x="24" y="114" font-family="monospace" font-size="18">dataset={record['dataset']} clip={record['clip_id']}</text>
<text x="24" y="146" font-family="monospace" font-size="18">changed_top5: STWM={record['stwm_changed_top5']:.4f} copy={record['copy_changed_top5']:.4f}</text>
<text x="24" y="178" font-family="monospace" font-size="18">overall_top5: STWM={record['stwm_overall_top5']:.4f} copy={record['copy_overall_top5']:.4f}</text>
</svg>
""",
        encoding="utf-8",
    )
    return {
        "case": case_name,
        "item_key": record["item_key"],
        "dataset": record["dataset"],
        "clip_id": record["clip_id"],
        "gif": str(gif_path),
        "png_strip": str(png_path),
        "pdf_strip": str(pdf_path),
        "svg_summary": str(svg_path),
        "stwm_changed_top5": record["stwm_changed_top5"],
        "copy_changed_top5": record["copy_changed_top5"],
        "raw_frame_path": frame_path,
        "horizon_steps_rendered": horizon,
    }


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF Raw-Frame Rollout Visualization V12", ""]
    for key in [
        "actual_mp4_or_gif_generated",
        "raw_observed_frames_included",
        "stwm_prediction_included",
        "copy_baseline_included",
        "gt_future_semantic_target_included",
        "case_count",
        "visualization_status",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    lines.append("## Cases")
    for case in payload.get("cases", []):
        lines.append(f"- {case['case']}: {case['gif']} ({case['item_key']})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    _set_process_title()
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="outputs/checkpoints/stwm_mixed_fullscale_v2_20260428/c32_seed456_final.pt")
    p.add_argument("--prototype-count", type=int, default=32)
    p.add_argument("--test-cache-report", default="reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json")
    p.add_argument("--future-cache-report", default="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    p.add_argument("--observed-report", default="reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json")
    p.add_argument("--figure-dir", default="assets/figures/stwm_fstf_rollout_v12")
    p.add_argument("--video-dir", default="assets/videos/stwm_fstf_rollout_v12")
    p.add_argument("--output", default="reports/stwm_fstf_visualization_v12_20260502.json")
    p.add_argument("--doc", default="docs/STWM_FSTF_VISUALIZATION_V12_20260502.md")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-batches", type=int, default=48)
    args = p.parse_args()

    cache_path = _report_batch_path(Path(args.test_cache_report))
    cache = torch.load(cache_path, map_location="cpu")
    batches = list(cache["batches"])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    records = _evaluate_items(
        batches=batches,
        checkpoint_path=Path(args.checkpoint),
        prototype_count=int(args.prototype_count),
        future_cache_report=Path(args.future_cache_report),
        observed_report=Path(args.observed_report),
        device=device,
        max_batches=int(args.max_batches),
    )
    cases = _pick_cases(records)
    rendered = [
        _render_case(case_name=name, record=rec, figure_dir=Path(args.figure_dir), video_dir=Path(args.video_dir), horizon=8)
        for name, rec in cases
    ]
    payload = {
        "audit_name": "stwm_fstf_visualization_v12",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "test_cache_report": str(args.test_cache_report),
        "batch_cache_path": str(cache_path),
        "future_cache_report": str(args.future_cache_report),
        "observed_report": str(args.observed_report),
        "figure_dir": str(args.figure_dir),
        "video_dir": str(args.video_dir),
        "case_count": len(rendered),
        "cases": rendered,
        "actual_mp4_or_gif_generated": bool(rendered),
        "raw_observed_frames_included": bool(rendered),
        "observed_trace_units_included": bool(rendered),
        "observed_semantic_memory_included": "prototype labels and copy-memory colors included",
        "copy_baseline_included": bool(rendered),
        "stwm_prediction_included": bool(rendered),
        "gt_future_semantic_target_included": bool(rendered),
        "changed_unit_highlight_included": bool(rendered),
        "horizon_steps_rendered": 8 if rendered else 0,
        "raw_video_end_to_end_training": False,
        "frozen_video_derived_trace_semantic_cache": True,
        "visualization_status": "raw_frame_gif_generated" if rendered else "failed_no_rendered_cases",
    }
    _dump(Path(args.output), payload)
    _write_doc(Path(args.doc), payload)


if __name__ == "__main__":
    main()
