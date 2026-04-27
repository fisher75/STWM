from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stwm.tools.external_baselines.common_io import DOCS, REPORTS, ROOT, write_json, write_markdown  # noqa: E402


OUT_DIR = ROOT / "baselines" / "outputs" / "manifest_sanity"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def decode_rle(rle: dict[str, Any] | None) -> np.ndarray | None:
    if not rle:
        return None
    counts = rle["counts"].encode("ascii") if isinstance(rle.get("counts"), str) else rle["counts"]
    return mask_utils.decode({"size": rle["size"], "counts": counts}).astype(bool)


def overlay_mask(img: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> Image.Image:
    if mask is None or not mask.any():
        return img
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, color + (0,))
    a = (mask.astype(np.uint8) * int(255 * alpha))
    overlay.putalpha(Image.fromarray(a, mode="L").resize(base.size))
    return Image.alpha_composite(base, overlay).convert("RGB")


def draw_box(draw: ImageDraw.ImageDraw, bbox: list[float] | None, color: tuple[int, int, int], width: int = 4) -> None:
    if not bbox:
        return
    xy = [float(x) for x in bbox[:4]]
    for i in range(width):
        draw.rectangle([xy[0] - i, xy[1] - i, xy[2] + i, xy[3] + i], outline=color)


def fit(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.copy()
    img.thumbnail(size)
    canvas = Image.new("RGB", size, (245, 245, 245))
    x = (size[0] - img.width) // 2
    y = (size[1] - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def render_item(item: dict[str, Any], out_path: Path) -> tuple[bool, str | None]:
    try:
        frames = item["frame_paths"]
        obs_idx = int(item["observed_prompt_frame_index"])
        fut_idx = int(item["future_frame_index"])
        obs = Image.open(frames[obs_idx]).convert("RGB")
        fut = Image.open(frames[fut_idx]).convert("RGB")
        observed = item.get("observed_target") or {}
        obs_mask = decode_rle(observed.get("mask_rle"))
        obs = overlay_mask(obs, obs_mask, (20, 180, 80), 0.35)
        draw_obs = ImageDraw.Draw(obs)
        draw_box(draw_obs, observed.get("bbox"), (20, 180, 80), 5)
        gt = str(item.get("gt_candidate_id"))
        draw_fut = ImageDraw.Draw(fut)
        for cand in item.get("future_candidates", []):
            is_gt = str(cand.get("candidate_id")) == gt
            color = (20, 180, 80) if is_gt else (210, 60, 60)
            if is_gt:
                fut = overlay_mask(fut, decode_rle(cand.get("mask_rle")), color, 0.35)
                draw_fut = ImageDraw.Draw(fut)
            draw_box(draw_fut, cand.get("bbox"), color, 3 if is_gt else 1)
        obs_small = fit(obs, (640, 360))
        fut_small = fit(fut, (640, 360))
        sheet = Image.new("RGB", (1280, 470), (255, 255, 255))
        sheet.paste(obs_small, (0, 70))
        sheet.paste(fut_small, (640, 70))
        draw = ImageDraw.Draw(sheet)
        title = item["protocol_item_id"]
        tags = ", ".join(k for k, v in item.get("subset_tags", {}).items() if v)
        draw.text((20, 12), title[:120], fill=(0, 0, 0))
        draw.text((20, 38), f"Observed prompt, target={observed.get('target_id')} | tags={tags[:90]}", fill=(40, 40, 40))
        draw.text((660, 38), f"Future candidates, GT={gt}, candidates={len(item.get('future_candidates', []))}", fill=(40, 40, 40))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(out_path)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}:{exc}"


def main() -> None:
    manifest_path = REPORTS / "stwm_external_baseline_item_manifest_20260426.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {"items": []}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ready = [
        item
        for item in manifest.get("items", [])
        if item.get("readiness", {}).get("cutie_ready") or item.get("readiness", {}).get("sam2_ready")
    ]
    selected = ready[:20]
    paths = []
    errors = Counter()
    for idx, item in enumerate(selected, start=1):
        out = OUT_DIR / f"manifest_contact_{idx:03d}.png"
        ok, err = render_item(item, out)
        if ok:
            paths.append(str(out))
        else:
            errors[err or "unknown_error"] += 1
    report = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_manifest": str(manifest_path),
        "visualized_item_count": len(paths),
        "contact_sheet_paths": paths,
        "visual_sanity_passed": len(paths) >= 20,
        "common_visual_errors": dict(errors),
        "exact_blocking_reason": None if paths else "no_ready_manifest_items_available_for_visualization",
    }
    write_json(REPORTS / "stwm_external_baseline_manifest_visual_sanity_20260426.json", report)
    lines = [
        f"- visualized_item_count: `{report['visualized_item_count']}`",
        f"- visual_sanity_passed: `{report['visual_sanity_passed']}`",
        f"- exact_blocking_reason: `{report['exact_blocking_reason']}`",
        "",
        "## Contact Sheets",
    ]
    lines.extend(f"- `{p}`" for p in paths)
    if errors:
        lines.append("")
        lines.append("## Common Visual Errors")
        lines.extend(f"- `{k}`: {v}" for k, v in errors.items())
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_MANIFEST_VISUAL_SANITY_20260426.md", "STWM External Baseline Manifest Visual Sanity 20260426", lines)


if __name__ == "__main__":
    main()

