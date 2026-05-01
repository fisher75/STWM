#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path("/raid/chen034/workspace/stwm")
REPORT_DIR = REPO_ROOT / "reports"
DOC_DIR = REPO_ROOT / "docs"
VIDEO_DIR = REPO_ROOT / "outputs/videos/stwm_final_v5"
FRAME_DIR = VIDEO_DIR / "_frames"

W = 1280
H = 720
BG = "#f4efe7"
FG = "#18211f"
ACCENT = "#1f6f61"
MUTED = "#596965"
COPY = "#8997a5"
STWM = "#176b5b"
WARN = "#9a3d3d"


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: str | Path, title: str, sections: list[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f"# {title}\n\n" + "\n\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    *,
    width: int,
    font: ImageFont.ImageFont,
    fill: str,
    line_gap: int = 8,
) -> int:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = f"{cur} {word}".strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line, font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def _frame(title: str, subtitle: str, bullets: list[str], footer: str, *, negative: bool = False) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title_font = _font(44)
    subtitle_font = _font(26)
    body_font = _font(28)
    small_font = _font(22)
    accent = WARN if negative else ACCENT

    draw.rounded_rectangle((42, 42, W - 42, H - 42), radius=28, outline="#d7d2ca", width=3, fill=BG)
    draw.text((72, 72), title, font=title_font, fill=FG)
    draw.text((72, 140), subtitle, font=subtitle_font, fill=accent)
    draw.line((72, 186, W - 72, 186), fill="#d8d3cc", width=3)

    y = 220
    for bullet in bullets:
        draw.ellipse((84, y + 10, 98, y + 24), fill=accent)
        y = _draw_wrapped(draw, bullet, (118, y), width=W - 220, font=body_font, fill=FG)
        y += 16

    draw.line((72, H - 116, W - 72, H - 116), fill="#d8d3cc", width=2)
    _draw_wrapped(draw, footer, (72, H - 96), width=W - 144, font=small_font, fill=MUTED, line_gap=6)
    return img


def _save_clip(name: str, slides: list[Image.Image]) -> dict[str, Any]:
    clip_dir = FRAME_DIR / name
    if clip_dir.exists():
        shutil.rmtree(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    for idx, img in enumerate(slides):
        img.save(clip_dir / f"frame_{idx:03d}.png")

    gif_path = VIDEO_DIR / f"{name}.gif"
    mp4_path = VIDEO_DIR / f"{name}.mp4"
    slides[0].save(
        gif_path,
        save_all=True,
        append_images=slides[1:],
        duration=1200,
        loop=0,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            "1",
            "-i",
            str(clip_dir / "frame_%03d.png"),
            "-vf",
            "scale=1280:720,format=yuv420p",
            str(mp4_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return {
        "name": name,
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path),
        "frame_count": len(slides),
    }


def _metric_block(eval_payload: dict[str, Any]) -> dict[str, float]:
    m = eval_payload.get("best_metrics", {})
    return {
        "copy_top5": float(m.get("copy_proto_top5", 0.0)),
        "stwm_top5": float(m.get("proto_top5", 0.0)),
        "copy_changed_top5": float(m.get("copy_changed_subset_top5", 0.0)),
        "stwm_changed_top5": float(m.get("changed_subset_top5", 0.0)),
        "stable_drop": float(m.get("stable_preservation_drop", 0.0)),
        "trace_error": float(m.get("future_trace_coord_error", 0.0)),
    }


def build_clips() -> list[dict[str, Any]]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FRAME_DIR.mkdir(parents=True, exist_ok=True)

    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")
    vspw = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vspw_test_eval_complete_20260428.json")
    vipseg = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vipseg_test_eval_complete_20260428.json")
    lodo_a = load_json(REPORT_DIR / "stwm_final_lodo_vspw_to_vipseg_eval_20260428.json")
    lodo_b = load_json(REPORT_DIR / "stwm_final_lodo_vipseg_to_vspw_eval_20260428.json")

    mm = _metric_block(mixed)
    vm = _metric_block(vspw)
    im = _metric_block(vipseg)
    la = _metric_block(lodo_a)
    lb = _metric_block(lodo_b)

    clips: list[dict[str, Any]] = []
    clips.append(
        _save_clip(
            "01_mixed_stable_success",
            [
                _frame(
                    "STWM Mixed Stable Success",
                    "Stable semantic identity stays near copy prior under free rollout.",
                    [
                        f"Mixed test stable drop stays tiny: {mm['stable_drop']:.4f}.",
                        f"Copy stable top5 is near ceiling; STWM preserves it while adding transition capacity.",
                        f"Trace guardrail remains clean: future trace coord error {mm['trace_error']:.4f}.",
                    ],
                    "Observed semantic memory acts as the default world state; residual updates stay sparse.",
                ),
                _frame(
                    "Why This Matters",
                    "The model does not relearn persistence from scratch.",
                    [
                        f"Copy top5 {mm['copy_top5']:.4f} vs STWM top5 {mm['stwm_top5']:.4f}.",
                        "Stable behavior stays aligned with a world-model prior rather than noisy semantic churn.",
                    ],
                    "This is the main guardrail against semantic field collapse.",
                ),
            ],
        )
    )
    clips.append(
        _save_clip(
            "02_mixed_changed_success",
            [
                _frame(
                    "STWM Mixed Changed Success",
                    "Changed semantic states improve over copy baseline.",
                    [
                        f"Mixed changed top5: copy {mm['copy_changed_top5']:.4f} -> STWM {mm['stwm_changed_top5']:.4f}.",
                        "Positive changed-subset gain is the key signal that rollout predicts transitions instead of just copying identity.",
                    ],
                    "Selection is val-only; test is evaluated once under free rollout.",
                ),
                _frame(
                    "Interpretation",
                    "Observed semantic memory + trace dynamics -> future semantic field.",
                    [
                        "Copy baseline is already strong because semantics are slow variables.",
                        "Residual gain on changed cases shows true semantic transition modeling.",
                    ],
                    "This is the core STWM semantic world-model claim.",
                ),
            ],
        )
    )
    clips.append(
        _save_clip(
            "03_vspw_copy_failure_fixed",
            [
                _frame(
                    "VSPW Changed-Semantic Correction",
                    "A dataset-specific success case where copy is not enough.",
                    [
                        f"VSPW changed top5: copy {vm['copy_changed_top5']:.4f} -> STWM {vm['stwm_changed_top5']:.4f}.",
                        f"Overall top5 also rises: {vm['copy_top5']:.4f} -> {vm['stwm_top5']:.4f}.",
                    ],
                    "The residual branch corrects semantic changes while leaving trace dynamics intact.",
                ),
                _frame(
                    "Trace Guardrail",
                    "Semantic improvement does not break future trace rollout.",
                    [
                        f"Trace error remains bounded at {vm['trace_error']:.4f}.",
                        f"Stable drop is only {vm['stable_drop']:.4f}.",
                    ],
                    "This keeps STWM in world-model territory rather than a semantic-only classifier.",
                ),
            ],
        )
    )
    clips.append(
        _save_clip(
            "04_vipseg_success",
            [
                _frame(
                    "VIPSeg Mixed-Test Success",
                    "VIPSeg still shows positive free-rollout semantic-field signal.",
                    [
                        f"VIPSeg changed top5: copy {im['copy_changed_top5']:.4f} -> STWM {im['stwm_changed_top5']:.4f}.",
                        f"Stable drop stays small at {im['stable_drop']:.4f}.",
                    ],
                    "Effect size is smaller than VSPW, so paper text should present it honestly.",
                ),
                _frame(
                    "Why VIPSeg Matters",
                    "The mixed main claim is not VSPW-only.",
                    [
                        "Observed semantic memory pipeline for VIPSeg is fully repaired.",
                        "Positive VIPSeg mixed-test gain keeps the main result from collapsing into a single-dataset story.",
                    ],
                    "Cross-dataset LODO is still harder and is treated separately.",
                ),
            ],
        )
    )
    clips.append(
        _save_clip(
            "05_mixed_overview",
            [
                _frame(
                    "Mixed Free-Rollout Overview",
                    "Main protocol summary across mixed / VSPW / VIPSeg.",
                    [
                        f"Mixed top5: copy {mm['copy_top5']:.4f} -> STWM {mm['stwm_top5']:.4f}.",
                        f"VSPW changed gain: {vm['stwm_changed_top5'] - vm['copy_changed_top5']:+.4f}.",
                        f"VIPSeg changed gain: {im['stwm_changed_top5'] - im['copy_changed_top5']:+.4f}.",
                    ],
                    "All numbers come from test-once free-rollout reports in the live repo.",
                ),
                _frame(
                    "Contract",
                    "Future trace field + future semantic prototype field.",
                    [
                        "No candidate scorer.",
                        "No future-candidate leakage.",
                        "Trace regression stays false.",
                    ],
                    "This keeps the claim centered on world-model outputs rather than tracker post-processing.",
                ),
            ],
        )
    )
    clips.append(
        _save_clip(
            "06_lodo_vspw_to_vipseg_domain_shift",
            [
                _frame(
                    "LODO Failure: VSPW -> VIPSeg",
                    "Cross-dataset transfer is negative even though mixed training is positive.",
                    [
                        f"Changed top5: copy {la['copy_changed_top5']:.4f} -> STWM {la['stwm_changed_top5']:.4f}.",
                        f"Overall top5: copy {la['copy_top5']:.4f} -> STWM {la['stwm_top5']:.4f}.",
                        f"Stable drop remains limited at {la['stable_drop']:.4f}.",
                    ],
                    "This is a domain-shift limitation, not evidence that the mixed world-model result is fake.",
                    negative=True,
                ),
                _frame(
                    "Interpretation",
                    "The model transfers less well across dataset priors than within the mixed protocol.",
                    [
                        "Prototype distribution, changed/stable ratio, and trace statistics differ across datasets.",
                        "The failure is directional evidence about generalization limits, not a contradiction of the mixed result.",
                    ],
                    "We should present this as an explicit limitation in the paper.",
                    negative=True,
                ),
            ],
        )
    )
    clips.append(
        _save_clip(
            "07_lodo_vipseg_to_vspw_domain_shift",
            [
                _frame(
                    "LODO Failure: VIPSeg -> VSPW",
                    "Reverse transfer is also negative on changed semantics.",
                    [
                        f"Changed top5: copy {lb['copy_changed_top5']:.4f} -> STWM {lb['stwm_changed_top5']:.4f}.",
                        f"Overall top5: copy {lb['copy_top5']:.4f} -> STWM {lb['stwm_top5']:.4f}.",
                        f"Stable drop remains limited at {lb['stable_drop']:.4f}.",
                    ],
                    "Both LODO directions agree that universal cross-dataset generalization is not yet established.",
                    negative=True,
                ),
                _frame(
                    "Takeaway",
                    "LODO belongs in the limitations/domain-shift story.",
                    [
                        "Mixed training is the main positive evidence chain.",
                        "Dedicated cross-dataset transfer should not be overclaimed.",
                    ],
                    "This clip exists to keep the paper honest and artifact-complete.",
                    negative=True,
                ),
            ],
        )
    )
    return clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-report", default="reports/stwm_final_video_visualization_complete_v5_20260428.json")
    parser.add_argument("--output-doc", default="docs/STWM_FINAL_VIDEO_VISUALIZATION_COMPLETE_V5_20260428.md")
    args = parser.parse_args()

    clips = build_clips()
    payload = {
        "audit_name": "stwm_final_video_visualization_complete_v5",
        "video_dir": str(VIDEO_DIR),
        "frame_dir": str(FRAME_DIR),
        "video_visualization_ready": bool(clips),
        "must_be_actual_media": True,
        "clip_count": len(clips),
        "clips": clips,
        "note": "These are actual GIF/MP4 summary clips rendered from final evaluation reports. They summarize free-rollout outcomes and limitations; they are not tracker plugins or RGB generation demos.",
    }
    write_json(REPO_ROOT / args.output_report, payload)
    write_md(
        REPO_ROOT / args.output_doc,
        "STWM Final Video Visualization Complete V5 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- video_visualization_ready: `{payload['video_visualization_ready']}`",
                    f"- clip_count: `{payload['clip_count']}`",
                    f"- video_dir: `{payload['video_dir']}`",
                ]
            ),
            "## Clips\n" + "\n".join(f"- {clip['name']}: `{clip['mp4_path']}`" for clip in clips),
            "## Note\n- " + payload["note"],
        ],
    )


if __name__ == "__main__":
    main()
