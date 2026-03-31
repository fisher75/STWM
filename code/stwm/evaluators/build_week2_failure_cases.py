from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build week2 v2.1 failure-focused figure pack")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/week2_minival_v2_1/seed_42")
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/visualizations/week2_figures_v2_1")
    parser.add_argument("--hard-selection-report", default="/home/chen034/workspace/stwm/reports/week2_minival_v2_hard_selection.json")
    parser.add_argument("--cases-per-panel", type=int, default=4)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _load_summary_and_cases(run_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    summary_path = run_dir / "eval" / "mini_val_summary_last.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    summary = _load_json(summary_path)

    per_clip_metrics: dict[str, dict[str, Any]] = {}
    case_map: dict[str, dict[str, Any]] = {}
    for item in summary.get("per_clip", []):
        clip_id = str(item.get("clip_id", ""))
        if not clip_id:
            continue
        per_clip_metrics[clip_id] = item
        case_file = Path(str(item.get("case_file", "")))
        if case_file.exists():
            case_map[clip_id] = _load_json(case_file)
    return summary, per_clip_metrics, case_map


def _draw_point(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], r: int = 5) -> None:
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)


def _query_frame_index(case: dict[str, Any]) -> int:
    start, end = case.get("future_range", [0, 0])
    energy = case.get("semantic_energy_by_frame", [])
    if isinstance(energy, list) and energy:
        local = int(np.argmax(np.asarray(energy, dtype=np.float32)))
    else:
        local = max(0, int(end) - int(start) - 1)
    return int(start) + int(local)


def _target_mask(mask_path: Path, target_label_id: int | None) -> np.ndarray:
    arr = np.asarray(Image.open(mask_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    if target_label_id is None:
        return arr > 0
    target = arr == int(target_label_id)
    if target.any():
        return target
    return arr > 0


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> Image.Image:
    base = np.asarray(image).astype(np.float32)
    out = base.copy()
    overlay = np.zeros_like(base)
    overlay[..., 0] = color[0]
    overlay[..., 1] = color[1]
    overlay[..., 2] = color[2]
    out[mask] = (1.0 - alpha) * out[mask] + alpha * overlay[mask]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _compose_compare_panel(
    clip_id: str,
    full_case: dict[str, Any],
    other_case: dict[str, Any],
    full_metrics: dict[str, Any],
    other_metrics: dict[str, Any],
    other_name: str,
) -> Image.Image:
    q_idx = _query_frame_index(full_case)
    frame_paths = full_case.get("frame_paths", [])
    mask_paths = full_case.get("mask_paths", [])
    q_idx = int(np.clip(q_idx, 0, max(0, len(frame_paths) - 1)))

    frame = Image.open(frame_paths[q_idx]).convert("RGB")
    target_label_id = full_case.get("target_label_id")
    target_mask = None
    if q_idx < len(mask_paths):
        mp = Path(mask_paths[q_idx])
        if mp.exists():
            target_mask = _target_mask(mp, target_label_id)

    left = frame.copy()
    right = frame.copy()
    if target_mask is not None:
        left = _overlay_mask(left, target_mask, (20, 170, 80), alpha=0.35)
        right = _overlay_mask(right, target_mask, (20, 170, 80), alpha=0.35)

    draw_l = ImageDraw.Draw(left)
    draw_r = ImageDraw.Draw(right)

    gt_centers = full_case.get("gt_centers", [])
    full_centers = full_case.get("pred_centers", [])
    other_centers = other_case.get("pred_centers", [])

    if q_idx < len(gt_centers):
        gx = float(gt_centers[q_idx][0]) * max(1, left.width - 1)
        gy = float(gt_centers[q_idx][1]) * max(1, left.height - 1)
        _draw_point(draw_l, gx, gy, (20, 170, 80), r=6)
        _draw_point(draw_r, gx, gy, (20, 170, 80), r=6)

    if q_idx < len(full_centers):
        fx = float(full_centers[q_idx][0]) * max(1, left.width - 1)
        fy = float(full_centers[q_idx][1]) * max(1, left.height - 1)
        _draw_point(draw_l, fx, fy, (40, 90, 220), r=6)

    if q_idx < len(other_centers):
        ox = float(other_centers[q_idx][0]) * max(1, right.width - 1)
        oy = float(other_centers[q_idx][1]) * max(1, right.height - 1)
        _draw_point(draw_r, ox, oy, (220, 80, 40), r=6)

    left_draw = ImageDraw.Draw(left)
    right_draw = ImageDraw.Draw(right)
    left_draw.rectangle((0, 0, left.width, 42), fill=(0, 0, 0))
    right_draw.rectangle((0, 0, right.width, 42), fill=(0, 0, 0))
    left_draw.text((8, 8), f"full | clip={clip_id}", fill=(255, 255, 255))
    left_draw.text((8, 24), f"q_err={_safe_float(full_metrics.get('query_localization_error')):.4f}, q_hit={_safe_float(full_metrics.get('query_hit_rate')):.3f}", fill=(255, 255, 255))
    right_draw.text((8, 8), f"{other_name}", fill=(255, 255, 255))
    right_draw.text((8, 24), f"q_err={_safe_float(other_metrics.get('query_localization_error')):.4f}, q_hit={_safe_float(other_metrics.get('query_hit_rate')):.3f}", fill=(255, 255, 255))

    canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)), color=(255, 255, 255))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    return canvas


def _pick_ids_full_fail_woid_worse(full: dict[str, dict[str, Any]], woid: dict[str, dict[str, Any]], k: int) -> list[str]:
    rows = []
    for clip_id, fm in full.items():
        if clip_id not in woid:
            continue
        delta = _safe_float(woid[clip_id].get("query_localization_error")) - _safe_float(fm.get("query_localization_error"))
        score = _safe_float(fm.get("query_localization_error")) + max(0.0, delta)
        rows.append((clip_id, score, delta))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in rows[:k]]


def _pick_ids_full_success_wosem_fail(full: dict[str, dict[str, Any]], wosem: dict[str, dict[str, Any]], k: int) -> list[str]:
    rows = []
    for clip_id, fm in full.items():
        if clip_id not in wosem:
            continue
        f_hit = _safe_float(fm.get("query_hit_rate"))
        s_hit = _safe_float(wosem[clip_id].get("query_hit_rate"))
        delta_err = _safe_float(wosem[clip_id].get("query_localization_error")) - _safe_float(fm.get("query_localization_error"))
        cond = (f_hit >= 0.5 and s_hit <= 0.5) or (delta_err > 0.01)
        if cond:
            rows.append((clip_id, delta_err, f_hit - s_hit))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return [x[0] for x in rows[:k]]


def _pick_query_hard_cases(
    hard_report: dict[str, Any],
    full_metrics: dict[str, dict[str, Any]],
    k: int,
) -> list[str]:
    ranked = hard_report.get("top_ranked", [])
    if not isinstance(ranked, list):
        return []

    success = []
    failure = []
    for row in ranked:
        clip_id = str(row.get("clip_id", ""))
        if not clip_id or clip_id not in full_metrics:
            continue
        q_hit = _safe_float(full_metrics[clip_id].get("query_hit_rate"))
        if q_hit >= 0.5:
            success.append(clip_id)
        else:
            failure.append(clip_id)

    out = []
    half = max(1, k // 2)
    out.extend(success[:half])
    out.extend(failure[: max(1, k - len(out))])
    if len(out) < k:
        used = set(out)
        for clip_id in success[half:] + failure[max(1, k - len(out)):]:
            if clip_id in used:
                continue
            out.append(clip_id)
            used.add(clip_id)
            if len(out) >= k:
                break
    return out[:k]


def main() -> None:
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, full_metrics, full_cases = _load_summary_and_cases(runs_root / "full")
    _, sem_metrics, sem_cases = _load_summary_and_cases(runs_root / "wo_semantics")
    _, id_metrics, id_cases = _load_summary_and_cases(runs_root / "wo_identity_memory")

    hard_report = {}
    hpath = Path(args.hard_selection_report)
    if hpath.exists():
        hard_report = _load_json(hpath)

    k = int(args.cases_per_panel)
    ids_a = _pick_ids_full_fail_woid_worse(full_metrics, id_metrics, k)
    ids_b = _pick_ids_full_success_wosem_fail(full_metrics, sem_metrics, k)
    ids_c = _pick_query_hard_cases(hard_report, full_metrics, k)

    artifacts: dict[str, list[str]] = {
        "full_fail_wo_identity_worse": [],
        "full_success_wo_semantics_fail": [],
        "query_hard_success_failure": [],
    }

    panel_a_dir = output_dir / "full_fail_wo_identity_worse"
    panel_b_dir = output_dir / "full_success_wo_semantics_fail"
    panel_c_dir = output_dir / "query_hard_success_failure"
    panel_a_dir.mkdir(parents=True, exist_ok=True)
    panel_b_dir.mkdir(parents=True, exist_ok=True)
    panel_c_dir.mkdir(parents=True, exist_ok=True)

    for clip_id in ids_a:
        if clip_id not in full_cases or clip_id not in id_cases:
            continue
        panel = _compose_compare_panel(clip_id, full_cases[clip_id], id_cases[clip_id], full_metrics[clip_id], id_metrics[clip_id], "wo_identity_memory")
        out = panel_a_dir / f"{clip_id}.png"
        panel.save(out)
        artifacts["full_fail_wo_identity_worse"].append(str(out))

    for clip_id in ids_b:
        if clip_id not in full_cases or clip_id not in sem_cases:
            continue
        panel = _compose_compare_panel(clip_id, full_cases[clip_id], sem_cases[clip_id], full_metrics[clip_id], sem_metrics[clip_id], "wo_semantics")
        out = panel_b_dir / f"{clip_id}.png"
        panel.save(out)
        artifacts["full_success_wo_semantics_fail"].append(str(out))

    for clip_id in ids_c:
        if clip_id not in full_cases:
            continue
        # Compare against wo_semantics when available, otherwise wo_identity_memory.
        if clip_id in sem_cases:
            other_case = sem_cases[clip_id]
            other_metrics = sem_metrics[clip_id]
            other_name = "wo_semantics"
        elif clip_id in id_cases:
            other_case = id_cases[clip_id]
            other_metrics = id_metrics[clip_id]
            other_name = "wo_identity_memory"
        else:
            continue
        panel = _compose_compare_panel(clip_id, full_cases[clip_id], other_case, full_metrics[clip_id], other_metrics, other_name)
        out = panel_c_dir / f"{clip_id}.png"
        panel.save(out)
        artifacts["query_hard_success_failure"].append(str(out))

    manifest = {
        "runs_root": str(runs_root),
        "output_dir": str(output_dir),
        "cases_per_panel": k,
        "selected": {
            "full_fail_wo_identity_worse": ids_a,
            "full_success_wo_semantics_fail": ids_b,
            "query_hard_success_failure": ids_c,
        },
        "artifacts": artifacts,
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
