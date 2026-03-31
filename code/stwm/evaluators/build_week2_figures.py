from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build week-2 visualization cases from mini-val outputs")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/week2_minival")
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/visualizations/week2_figures")
    parser.add_argument("--cases-per-type", type=int, default=2)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _circle_mask(height: int, width: int, center_x: float, center_y: float, radius_norm: float) -> np.ndarray:
    cx = float(np.clip(center_x, 0.0, 1.0)) * max(1, width - 1)
    cy = float(np.clip(center_y, 0.0, 1.0)) * max(1, height - 1)
    radius_px = max(1.0, float(radius_norm) * float(min(height, width)))
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius_px * radius_px


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> Image.Image:
    base = np.array(image).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    out = base.copy()
    out[mask] = (1.0 - alpha) * base[mask] + alpha * overlay[mask]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out)


def _draw_title(image: Image.Image, title: str) -> Image.Image:
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 24), fill=(0, 0, 0))
    draw.text((8, 6), title, fill=(255, 255, 255))
    return image


def _concat_h(images: list[Image.Image]) -> Image.Image:
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), color=(255, 255, 255))
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width
    return canvas


def _draw_path(
    image: Image.Image,
    centers: list[list[float]],
    start: int,
    end: int,
    color: tuple[int, int, int],
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(image)
    pts = []
    for idx in range(start, end):
        if idx >= len(centers):
            continue
        cx, cy = centers[idx]
        x = float(np.clip(cx, 0.0, 1.0)) * max(1, image.width - 1)
        y = float(np.clip(cy, 0.0, 1.0)) * max(1, image.height - 1)
        pts.append((x, y))
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=width)
    for x, y in pts:
        r = 3
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def _draw_visibility_chart(
    full_case: dict[str, Any],
    other_case: dict[str, Any],
    title: str,
    width: int = 640,
    height: int = 240,
) -> Image.Image:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    start, end = full_case["future_range"]
    future_len = max(1, end - start)

    gt = full_case["gt_visibility"][start:end]
    full_pred = full_case["pred_visibility_prob"][start:end]
    other_pred = other_case["pred_visibility_prob"][start:end]

    margin = 30
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(0, 0, 0), width=2)

    def to_xy(index: int, value: float) -> tuple[float, float]:
        x = margin + (width - 2 * margin) * (index / max(1, future_len - 1))
        y = height - margin - (height - 2 * margin) * float(np.clip(value, 0.0, 1.0))
        return x, y

    def draw_series(values: list[float], color: tuple[int, int, int]) -> None:
        points = [to_xy(i, float(v)) for i, v in enumerate(values)]
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)

    draw_series(gt, (20, 140, 60))
    draw_series(full_pred, (40, 90, 220))
    draw_series(other_pred, (220, 80, 40))

    draw.text((10, 8), title, fill=(0, 0, 0))
    draw.text((10, height - 20), "green=gt, blue=full, red=wo_identity", fill=(0, 0, 0))
    return img


def _load_case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in summary.get("per_clip", []):
        case_file = item.get("case_file", "")
        if not case_file:
            continue
        path = Path(case_file)
        if not path.exists():
            continue
        payload = _load_json(path)
        out[payload["clip_id"]] = payload
    return out


def _pick_ids(common_ids: list[str], k: int) -> list[str]:
    if k <= 0:
        return []
    return common_ids[:k]


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_names = ["full", "wo_semantics", "wo_trajectory", "wo_identity_memory"]
    summaries: dict[str, dict[str, Any]] = {}
    case_maps: dict[str, dict[str, dict[str, Any]]] = {}
    for run_name in run_names:
        summary_path = runs_root / run_name / "eval" / "mini_val_summary_last.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary: {summary_path}")
        summaries[run_name] = _load_json(summary_path)
        case_maps[run_name] = _load_case_map(summaries[run_name])

    artifacts: dict[str, list[str]] = {
        "semantic_mask": [],
        "trajectory": [],
        "identity_recovery": [],
        "query_conditioned": [],
    }

    # 1) full vs wo_semantics: future mask comparison.
    semantic_dir = output_dir / "full_vs_wo_semantics_mask"
    semantic_dir.mkdir(parents=True, exist_ok=True)
    common_ids = sorted(set(case_maps["full"]).intersection(case_maps["wo_semantics"]))
    for clip_id in _pick_ids(common_ids, args.cases_per_type):
        full_case = case_maps["full"][clip_id]
        sem_case = case_maps["wo_semantics"][clip_id]
        start, end = full_case["future_range"]
        idx = max(start, end - 1)

        frame = Image.open(full_case["frame_paths"][idx]).convert("RGB")
        gt_mask = np.array(Image.open(full_case["mask_paths"][idx])) > 0

        full_center = full_case["pred_centers"][idx]
        sem_center = sem_case["pred_centers"][idx]
        radius_norm = float(full_case["radius_norm"])

        full_pred_mask = _circle_mask(frame.height, frame.width, full_center[0], full_center[1], radius_norm)
        sem_pred_mask = _circle_mask(frame.height, frame.width, sem_center[0], sem_center[1], radius_norm)

        panel_gt = _draw_title(_overlay_mask(frame.copy(), gt_mask, (20, 170, 80)), "GT future mask")
        panel_full = _draw_title(_overlay_mask(frame.copy(), full_pred_mask, (40, 90, 220)), "full pred mask")
        panel_sem = _draw_title(_overlay_mask(frame.copy(), sem_pred_mask, (220, 140, 40)), "wo_semantics pred mask")

        canvas = _concat_h([panel_gt, panel_full, panel_sem])
        out_path = semantic_dir / f"{clip_id}.png"
        canvas.save(out_path)
        artifacts["semantic_mask"].append(str(out_path))

    # 2) full vs wo_trajectory: future trajectory comparison.
    traj_dir = output_dir / "full_vs_wo_trajectory"
    traj_dir.mkdir(parents=True, exist_ok=True)
    common_ids = sorted(set(case_maps["full"]).intersection(case_maps["wo_trajectory"]))
    for clip_id in _pick_ids(common_ids, args.cases_per_type):
        full_case = case_maps["full"][clip_id]
        traj_case = case_maps["wo_trajectory"][clip_id]
        start, end = full_case["future_range"]
        frame = Image.open(full_case["frame_paths"][start]).convert("RGB")

        panel_full = frame.copy()
        _draw_path(panel_full, full_case["gt_centers"], start, end, (20, 170, 80), width=3)
        _draw_path(panel_full, full_case["pred_centers"], start, end, (40, 90, 220), width=3)
        _draw_title(panel_full, "full: gt(green) vs pred(blue)")

        panel_traj = frame.copy()
        _draw_path(panel_traj, traj_case["gt_centers"], start, end, (20, 170, 80), width=3)
        _draw_path(panel_traj, traj_case["pred_centers"], start, end, (220, 80, 40), width=3)
        _draw_title(panel_traj, "wo_trajectory: gt(green) vs pred(orange)")

        canvas = _concat_h([panel_full, panel_traj])
        out_path = traj_dir / f"{clip_id}.png"
        canvas.save(out_path)
        artifacts["trajectory"].append(str(out_path))

    # 3) full vs wo_identity_memory: occlusion/recovery and visibility dynamics.
    id_dir = output_dir / "full_vs_wo_identity_memory"
    id_dir.mkdir(parents=True, exist_ok=True)
    common_ids = sorted(set(case_maps["full"]).intersection(case_maps["wo_identity_memory"]))

    # Prioritize clips that have recovery events in full case.
    common_ids.sort(
        key=lambda clip: int(case_maps["full"][clip].get("has_occlusion_recovery_event", False)),
        reverse=True,
    )

    for clip_id in _pick_ids(common_ids, args.cases_per_type):
        full_case = case_maps["full"][clip_id]
        id_case = case_maps["wo_identity_memory"][clip_id]
        start, end = full_case["future_range"]
        idx = max(start, end - 1)

        frame = Image.open(full_case["frame_paths"][idx]).convert("RGB")
        panel_track = frame.copy()
        _draw_path(panel_track, full_case["gt_centers"], start, end, (20, 170, 80), width=3)
        _draw_path(panel_track, full_case["pred_centers"], start, end, (40, 90, 220), width=3)
        _draw_path(panel_track, id_case["pred_centers"], start, end, (220, 80, 40), width=3)
        _draw_title(panel_track, "gt(green), full(blue), wo_identity(red)")

        panel_chart = _draw_visibility_chart(full_case, id_case, "Future visibility dynamics")
        canvas = _concat_h([panel_track, panel_chart])

        out_path = id_dir / f"{clip_id}.png"
        canvas.save(out_path)
        artifacts["identity_recovery"].append(str(out_path))

    # 4) query-conditioned cases.
    query_dir = output_dir / "query_conditioned_cases"
    query_dir.mkdir(parents=True, exist_ok=True)
    full_ids = sorted(case_maps["full"].keys())
    for clip_id in _pick_ids(full_ids, args.cases_per_type):
        full_case = case_maps["full"][clip_id]
        start, end = full_case["future_range"]
        idx = max(start, end - 1)

        frame = Image.open(full_case["frame_paths"][idx]).convert("RGB")
        panel = frame.copy()
        _draw_path(panel, full_case["gt_centers"], start, end, (20, 170, 80), width=3)
        _draw_path(panel, full_case["pred_centers"], start, end, (40, 90, 220), width=3)

        draw = ImageDraw.Draw(panel)
        query_label = full_case.get("query_label", "object")
        draw.rectangle((8, panel.height - 34, panel.width - 8, panel.height - 8), fill=(0, 0, 0))
        draw.text((14, panel.height - 28), f"query: {query_label}", fill=(255, 255, 255))
        _draw_title(panel, "query-conditioned localization")

        out_path = query_dir / f"{clip_id}.png"
        panel.save(out_path)
        artifacts["query_conditioned"].append(str(out_path))

    report = {
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "cases_per_type": int(args.cases_per_type),
        "artifacts": artifacts,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
