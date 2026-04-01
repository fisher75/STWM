from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import subprocess
from typing import Any

from PIL import Image, ImageDraw


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Package STWM V4.2 figure assets into demo frames/video")
    parser.add_argument(
        "--figure-dirs",
        default="/home/chen034/workspace/stwm/outputs/visualizations/stwm_v4_2_1b_multiseed_casebook,/home/chen034/workspace/stwm/outputs/visualizations/stwm_v4_2_1b_state_identifiability_figures",
    )
    parser.add_argument("--output-dir", default="/home/chen034/workspace/stwm/outputs/visualizations/stwm_v4_2_1b_demo")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    return parser


def _collect_image_paths_from_obj(obj: Any, out: list[Path]) -> None:
    if isinstance(obj, str):
        p = Path(obj)
        if p.suffix.lower() in IMAGE_EXTS and p.exists():
            out.append(p)
        return
    if isinstance(obj, list):
        for item in obj:
            _collect_image_paths_from_obj(item, out)
        return
    if isinstance(obj, dict):
        for val in obj.values():
            _collect_image_paths_from_obj(val, out)


def _collect_from_manifest(dir_path: Path) -> list[Path]:
    manifest = dir_path / "figure_manifest.json"
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text())
    out: list[Path] = []
    _collect_image_paths_from_obj(payload, out)
    return out


def _collect_images(dir_path: Path) -> list[Path]:
    out = _collect_from_manifest(dir_path)
    if out:
        return out
    return sorted([p for p in dir_path.rglob("*") if p.suffix.lower() in IMAGE_EXTS])


def _dedupe_keep_order(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _sample_evenly(paths: list[Path], max_items: int) -> list[Path]:
    if len(paths) <= max_items:
        return paths
    if max_items <= 1:
        return [paths[0]]

    out: list[Path] = []
    total = len(paths)
    for i in range(max_items):
        idx = int(round(i * (total - 1) / float(max_items - 1)))
        out.append(paths[idx])
    return out


def _render_frame(src: Path, canvas_w: int, canvas_h: int, title: str) -> Image.Image:
    img = Image.open(src).convert("RGB")
    top_bar = 56
    avail_h = max(1, canvas_h - top_bar)

    scale = min(canvas_w / max(1, img.width), avail_h / max(1, img.height))
    new_w = max(1, int(img.width * scale))
    new_h = max(1, int(img.height * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(12, 14, 18))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas_w, top_bar), fill=(24, 28, 36))
    draw.text((14, 18), title, fill=(240, 240, 240))

    x = (canvas_w - new_w) // 2
    y = top_bar + (avail_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _build_video_ffmpeg(frame_glob: str, fps: int, out_path: Path) -> tuple[bool, str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg_not_found"

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(max(1, int(fps))),
        "-i",
        frame_glob,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout).strip()[:500]
    return True, "ok"


def main() -> None:
    args = build_parser().parse_args()

    figure_dirs = [Path(x.strip()) for x in str(args.figure_dirs).split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    frames_dir = out_dir / "storyboard_frames"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    all_images: list[Path] = []
    per_dir: dict[str, list[str]] = {}
    for d in figure_dirs:
        imgs = _collect_images(d)
        imgs = _dedupe_keep_order(imgs)
        per_dir[str(d)] = [str(x) for x in imgs]
        all_images.extend(imgs)

    all_images = _dedupe_keep_order(all_images)
    selected = _sample_evenly(all_images, max(1, int(args.max_frames)))

    rendered_frames: list[str] = []
    for idx, img_path in enumerate(selected, start=1):
        rel_title = str(img_path)
        frame = _render_frame(
            img_path,
            canvas_w=max(320, int(args.frame_width)),
            canvas_h=max(240, int(args.frame_height)),
            title=f"STWM V4.2 1B Demo | {idx}/{len(selected)} | {rel_title}",
        )
        out_path = frames_dir / f"frame_{idx:04d}.png"
        frame.save(out_path)
        rendered_frames.append(str(out_path))

    video_path = out_dir / "stwm_v4_2_1b_demo.mp4"
    ok, video_msg = _build_video_ffmpeg(str(frames_dir / "frame_%04d.png"), int(args.fps), video_path)

    manifest = {
        "figure_dirs": [str(x) for x in figure_dirs],
        "per_dir_images": per_dir,
        "total_unique_images": len(all_images),
        "selected_images": [str(x) for x in selected],
        "rendered_frames": rendered_frames,
        "video": {
            "path": str(video_path),
            "created": bool(ok),
            "message": video_msg,
            "fps": int(args.fps),
        },
    }

    (out_dir / "demo_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps({"output_dir": str(out_dir), "selected_frames": len(rendered_frames), "video_created": bool(ok)}, indent=2))


if __name__ == "__main__":
    main()
