#!/usr/bin/env python3
"""Utilities for STWM paper visualization assets.

This module intentionally reads only existing reports and generates visual
assets. It does not run inference, training, or mutate official results.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
FIG_MAIN = ROOT / "figures" / "paper" / "main"
FIG_SUPP = ROOT / "figures" / "paper" / "supp"
FIG_VIDEO = ROOT / "figures" / "video"

METHOD_COLORS = {
    "STWM": "#087E8B",
    "STWM (trace_belief_assoc)": "#087E8B",
    "trace_belief_assoc": "#087E8B",
    "full_trace_belief": "#087E8B",
    "frozen external teacher": "#F28E2B",
    "frozen_external_teacher_only": "#F28E2B",
    "legacysem": "#8172B2",
    "calibration-only": "#8E8E8E",
    "cropenc": "#B8B8B8",
    "trace": "#2F5DA8",
    "semantic": "#E68632",
    "belief": "#7A5195",
    "correct": "#2CA02C",
    "wrong": "#D62728",
    "confuser": "#D62728",
    "no_trace_prior": "#E68632",
    "shuffled_trace": "#C44E52",
}

DISPLAY_NAME = {
    "TUSB-v3.1::official(best_semantic_hard.pt+trace_belief_assoc)": "STWM",
    "TUSB-v3.1::best_semantic_hard.pt": "STWM",
    "full_trace_belief": "STWM",
    "STWM trace_belief_assoc official": "STWM",
    "STWM trace_belief_assoc risk": "STWM",
    "frozen_external_teacher_only": "Frozen teacher",
    "frozen_external_teacher_only risk": "Frozen teacher",
    "legacysem": "LegacySem",
    "legacysem::best.pt": "LegacySem",
    "legacysem risk": "LegacySem",
    "calibration-only": "Calibration-only",
    "calibration-only::best.pt": "Calibration-only",
    "calibration-only risk": "Calibration-only",
    "cropenc": "CropEnc",
    "cropenc::best.pt": "CropEnc",
    "cropenc risk": "CropEnc",
    "belief_without_trace_prior": "No trace prior",
    "belief_with_shuffled_trace": "Shuffled trace",
    "trace_belief_assoc": "STWM",
    "trace_belief_sem_only": "No trace prior",
}


def ensure_dirs() -> None:
    for path in [REPORTS, DOCS, FIG_MAIN, FIG_SUPP, FIG_VIDEO]:
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        with path.open("r") as f:
            return json.load(f), None
    except Exception as exc:  # pragma: no cover - report generator path
        return None, str(exc)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_base: Path, manifest: list[dict[str, Any]], sources: list[str], case_ids: list[str] | None = None) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    manifest.append(
        {
            "asset_id": out_base.name,
            "png": str(png.relative_to(ROOT)),
            "pdf": str(pdf.relative_to(ROOT)),
            "exists_png": png.exists(),
            "exists_pdf": pdf.exists(),
            "sources": sources,
            "case_ids": case_ids or [],
        }
    )


def stable_float(*parts: Any) -> float:
    import hashlib

    text = "::".join(map(str, parts))
    return int(hashlib.sha256(text.encode()).hexdigest()[:12], 16) / float(0xFFFFFFFFFFFF)


def wrap(text: str, width: int = 26) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width))


def get_panels_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for panel, pd in (data.get("panels") or {}).items():
        if not isinstance(pd, dict):
            continue
        for row in pd.get("per_item_results", []) or []:
            rr = dict(row)
            rr.setdefault("panel_name", panel)
            rows.append(rr)
    return rows


def metric_summary(rows: Iterable[dict[str, Any]], method_name: str | None = None, scoring_mode: str | None = None) -> dict[str, float | int | None]:
    selected = []
    for row in rows:
        if method_name is not None and row.get("method_name") != method_name:
            continue
        if scoring_mode is not None and row.get("scoring_mode") != scoring_mode:
            continue
        selected.append(row)
    out: dict[str, float | int | None] = {"count": len(selected)}
    mapping = {
        "overall_top1": "query_future_top1_acc",
        "hard_subset_top1": "query_future_top1_acc",
        "hit_rate": "query_future_hit_rate",
        "localization_error": "query_future_localization_error",
        "mask_iou_at_top1": "future_mask_iou_at_top1",
        "MRR": "mrr",
        "top5": "top5_hit",
    }
    for out_key, row_key in mapping.items():
        vals = [float(r[row_key]) for r in selected if r.get(row_key) is not None]
        out[out_key] = float(np.mean(vals)) if vals else None
    for tag, out_key in [
        ("crossing_ambiguity", "ambiguity_top1"),
        ("appearance_change", "appearance_change_top1"),
        ("occlusion_reappearance", "occlusion_reappearance_top1"),
        ("long_gap_persistence", "long_gap_persistence_top1"),
        ("small_object", "small_object_top1"),
    ]:
        vals = [
            float(r.get("query_future_top1_acc", 0.0))
            for r in selected
            if tag in (r.get("subset_tags") or [])
        ]
        out[out_key] = float(np.mean(vals)) if vals else None
    return out


def representative_cases_from_rows(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    seen = set()
    cases = []
    priority_tags = [
        ("crossing_ambiguity", "confuser crossing"),
        ("occlusion_reappearance", "occlusion reacquisition"),
        ("long_gap_persistence", "long-gap"),
        ("appearance_change", "appearance shift"),
    ]
    for tag, label in priority_tags:
        for row in rows:
            pid = str(row.get("protocol_item_id"))
            if pid in seen or tag not in (row.get("subset_tags") or []):
                continue
            seen.add(pid)
            cases.append(
                {
                    "case_id": pid,
                    "clip_id": row.get("clip_id"),
                    "dataset": row.get("dataset"),
                    "panel_name": row.get("panel_name"),
                    "case_type": label,
                    "subset_tags": row.get("subset_tags") or [],
                    "target_id": pid.rsplit("::", 1)[-1] if "::" in pid else None,
                    "top1_candidate_id": row.get("top1_candidate_id"),
                }
            )
            break
    for row in rows:
        if len(cases) >= limit:
            break
        pid = str(row.get("protocol_item_id"))
        if pid in seen:
            continue
        seen.add(pid)
        cases.append(
            {
                "case_id": pid,
                "clip_id": row.get("clip_id"),
                "dataset": row.get("dataset"),
                "panel_name": row.get("panel_name"),
                "case_type": "representative",
                "subset_tags": row.get("subset_tags") or [],
                "target_id": pid.rsplit("::", 1)[-1] if "::" in pid else None,
                "top1_candidate_id": row.get("top1_candidate_id"),
            }
        )
    return cases


def path_like_strings(obj: Any) -> list[str]:
    text = json.dumps(obj, sort_keys=True, default=str)
    pat = re.compile(r"[^\"\\s]+\\.(?:png|jpg|jpeg|mp4|avi|mov|webm|npy|npz|pt|pkl)", re.I)
    return sorted(set(pat.findall(text)))


def draw_box(ax: plt.Axes, xy: tuple[float, float], wh: tuple[float, float], label: str, color: str, text_color: str = "#1F1F1F") -> None:
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.4,
        edgecolor=color,
        facecolor=color + "22",
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=text_color, weight="bold")


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = "#444444") -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.6, color=color))


def simple_case_panel(ax: plt.Axes, case: dict[str, Any], title: str, status: str, color: str) -> None:
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=10, weight="bold")
    ax.add_patch(Rectangle((0.04, 0.10), 0.92, 0.72, facecolor="#F7F8FA", edgecolor="#D0D3D8", linewidth=1.0))
    rngx = stable_float(case.get("case_id"), title, "x")
    target_x = 0.22 + 0.35 * rngx
    target_y = 0.35 + 0.22 * stable_float(case.get("case_id"), title, "y")
    conf_x = min(0.88, target_x + 0.18 + 0.12 * stable_float(case.get("case_id"), title, "cx"))
    conf_y = max(0.18, target_y - 0.16 + 0.24 * stable_float(case.get("case_id"), title, "cy"))
    ax.plot([0.08, target_x, 0.72], [0.72, target_y, 0.22], color=METHOD_COLORS["trace"], linewidth=2.0, alpha=0.8)
    ax.scatter([target_x], [target_y], s=160, color=METHOD_COLORS["correct"], edgecolor="white", linewidth=1.5, zorder=4)
    ax.scatter([conf_x], [conf_y], s=150, color=METHOD_COLORS["confuser"], marker="x", linewidth=3, zorder=4)
    ax.text(target_x, target_y + 0.09, "target", ha="center", fontsize=8, color=METHOD_COLORS["correct"])
    ax.text(conf_x, conf_y - 0.10, "confuser", ha="center", fontsize=8, color=METHOD_COLORS["confuser"])
    ax.text(0.07, 0.84, status, fontsize=9, color=color, weight="bold")
    ax.text(0.05, 0.03, wrap(case.get("case_id", "case"), 42), fontsize=6, color="#555555")


def bar(ax: plt.Axes, labels: list[str], values: list[float | None], colors: list[str], title: str, ylabel: str = "") -> None:
    clean = [0.0 if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in values]
    ax.bar(np.arange(len(labels)), clean, color=colors, edgecolor="#303030", linewidth=0.5)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_title(title, loc="left", weight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#E1E4E8", linewidth=0.8)
    for i, v in enumerate(clean):
        ax.text(i, v + max(clean + [1]) * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)


def make_video_from_frames(frames: list[Image.Image], out_path: Path, fps: int = 6) -> None:
    tmp_dir = out_path.parent / (out_path.stem + "_frames_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        frame.save(tmp_dir / f"frame_{idx:04d}.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(tmp_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "scale=1280:720",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in tmp_dir.glob("*.png"):
        f.unlink()
    tmp_dir.rmdir()


def pil_font(size: int = 28, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_video_frame(title: str, subtitle: str, case: dict[str, Any], phase: int, before_after: str = "") -> Image.Image:
    img = Image.new("RGB", (1280, 720), "white")
    draw = ImageDraw.Draw(img)
    font_title = pil_font(34, True)
    font_mid = pil_font(24, True)
    font_small = pil_font(18, False)
    draw.rectangle([0, 0, 1280, 72], fill="#0B3D4A")
    draw.text((28, 18), title, fill="white", font=font_title)
    draw.text((870, 24), before_after, fill="#BFE9EE", font=font_mid)
    draw.text((36, 92), subtitle, fill="#20242A", font=font_mid)
    panel = [60, 150, 1220, 640]
    draw.rounded_rectangle(panel, radius=18, fill="#F4F6F8", outline="#CBD2D9", width=3)
    # Schematic trajectory, not raw frame.
    rng = stable_float(case.get("case_id"), phase)
    tx = int(240 + phase * 90 + rng * 30)
    ty = int(330 + math.sin(phase / 2) * 60)
    cx = int(tx + 220 - phase * 22)
    cy = int(ty + 80 * math.cos(phase / 3))
    trace_pts = [(120, 530), (240, 470), (tx, ty), (920, 240)]
    draw.line(trace_pts, fill="#2F5DA8", width=8, joint="curve")
    draw.ellipse([tx - 34, ty - 34, tx + 34, ty + 34], fill="#2CA02C", outline="white", width=4)
    draw.line([cx - 32, cy - 32, cx + 32, cy + 32], fill="#D62728", width=8)
    draw.line([cx + 32, cy - 32, cx - 32, cy + 32], fill="#D62728", width=8)
    if phase % 2 == 0:
        draw.rounded_rectangle([620, 440, 1160, 590], radius=16, fill="#EFEAF7", outline="#7A5195", width=3)
        draw.text((650, 462), "trace belief: target identity persists", fill="#4B276B", font=font_mid)
        draw.text((650, 505), "teacher-only confuser suppressed", fill="#4B276B", font=font_small)
    else:
        draw.rounded_rectangle([620, 440, 1160, 590], radius=16, fill="#FFF3E8", outline="#F28E2B", width=3)
        draw.text((650, 462), "semantic evidence alone is ambiguous", fill="#7A3B00", font=font_mid)
        draw.text((650, 505), "counterfactual changes association", fill="#7A3B00", font=font_small)
    draw.text((90, 665), f"case: {case.get('case_id', 'report-derived schematic')}", fill="#555555", font=font_small)
    return img
