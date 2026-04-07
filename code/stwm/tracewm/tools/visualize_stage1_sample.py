#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

import numpy as np
import torch

from stwm.tracewm.datasets.stage1_kubric import Stage1KubricDataset
from stwm.tracewm.datasets.stage1_pointodyssey import Stage1PointOdysseyDataset
from stwm.tracewm.datasets.stage1_tapvid import Stage1TapVidDataset
from stwm.tracewm.datasets.stage1_tapvid3d import Stage1TapVid3DDataset
from stwm.tracewm.datasets.stage1_unified import load_stage1_minisplits

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    plt = None
    _MPL_IMPORT_ERR = str(exc)
else:
    _MPL_IMPORT_ERR = ""

try:
    from PIL import Image
except Exception as exc:  # pragma: no cover
    Image = None
    _PIL_IMPORT_ERR = str(exc)
else:
    _PIL_IMPORT_ERR = ""


DATA_ROOT = Path("/home/chen034/workspace/data")
STWM_ROOT = Path("/home/chen034/workspace/stwm")

MINISPLIT_PATH = DATA_ROOT / "_manifests" / "stage1_minisplits_20260408.json"
OUTPUT_DIR = STWM_ROOT / "outputs" / "stage1_visual_checks"
DOC_PATH = STWM_ROOT / "docs" / "STAGE1_VISUAL_SMOKE_20260408.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _records_for(minisplits: Dict[str, Any], dataset: str, split: str) -> List[Dict[str, Any]]:
    datasets = minisplits.get("datasets", {}) if isinstance(minisplits, dict) else {}
    ds = datasets.get(dataset, {}) if isinstance(datasets, dict) else {}
    out = ds.get(split, []) if isinstance(ds, dict) else []
    return [x for x in out if isinstance(x, dict)]


def _first_image_or_blank(obs_frames: Any, size: int = 512) -> np.ndarray:
    blank = np.ones((size, size, 3), dtype=np.uint8) * 245
    if not isinstance(obs_frames, list) or not obs_frames:
        return blank

    if Image is None:
        return blank

    first = str(obs_frames[0])
    p = Path(first)
    if not p.exists() or not p.is_file():
        return blank

    try:
        img = Image.open(p).convert("RGB").resize((size, size))
        return np.asarray(img)
    except Exception:
        return blank


def _project_tracks3d(tracks3d: torch.Tensor) -> torch.Tensor:
    xy = tracks3d[..., :2]
    mn = xy.amin(dim=(0, 1), keepdim=True)
    mx = xy.amax(dim=(0, 1), keepdim=True)
    denom = torch.clamp(mx - mn, min=1e-6)
    return (xy - mn) / denom


def _plot_sample_2d(ax: Any, bg: np.ndarray, tracks2d: torch.Tensor, obs_len: int, title: str) -> None:
    ax.imshow(bg)
    ax.set_title(title)
    ax.axis("off")

    t_all = tracks2d.shape[0]
    n_points = tracks2d.shape[1]
    k = min(16, n_points)

    for pid in range(k):
        traj = tracks2d[:, pid, :].detach().cpu().numpy()
        x = traj[:, 0] * (bg.shape[1] - 1)
        y = traj[:, 1] * (bg.shape[0] - 1)

        ax.plot(x[:obs_len], y[:obs_len], color="#1f77b4", linewidth=1.2)
        ax.plot(x[obs_len - 1 : t_all], y[obs_len - 1 : t_all], color="#d62728", linewidth=1.2)


def _render_dataset_sample(dataset_name: str, sample: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "dataset": dataset_name,
        "output": str(out_path),
        "ok": False,
        "note": "",
    }

    if plt is None:
        result["note"] = f"matplotlib_unavailable:{_MPL_IMPORT_ERR}"
        return result

    obs_len = int(sample["obs_valid"].shape[0]) if isinstance(sample.get("obs_valid"), torch.Tensor) else 8

    tracks2d = sample.get("obs_tracks_2d")
    fut2d = sample.get("fut_tracks_2d")
    tracks3d = sample.get("obs_tracks_3d")
    fut3d = sample.get("fut_tracks_3d")

    if isinstance(tracks2d, torch.Tensor) and isinstance(fut2d, torch.Tensor):
        all_tracks = torch.cat([tracks2d, fut2d], dim=0)
        bg = _first_image_or_blank(sample.get("obs_frames"), size=512)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _plot_sample_2d(ax, bg, all_tracks, obs_len=obs_len, title=f"{dataset_name} 2D trace smoke")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        result["ok"] = True
        result["note"] = "2d_overlay_done"
        return result

    if isinstance(tracks3d, torch.Tensor) and isinstance(fut3d, torch.Tensor):
        all_tracks3d = torch.cat([tracks3d, fut3d], dim=0)
        proj = _project_tracks3d(all_tracks3d)
        bg = np.ones((512, 512, 3), dtype=np.uint8) * 245
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        _plot_sample_2d(ax, bg, proj, obs_len=obs_len, title=f"{dataset_name} 3D->2D projection smoke")
        text = (
            f"points={all_tracks3d.shape[1]}\\n"
            f"obs={tracks3d.shape[0]} fut={fut3d.shape[0]}\\n"
            "mode=limited_eval_projection"
        )
        ax.text(10, 20, text, color="black", fontsize=9, bbox={"facecolor": "white", "alpha": 0.65})
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        result["ok"] = True
        result["note"] = "3d_projection_done"
        return result

    result["note"] = "no_tracks_available_for_visualization"
    return result


def main() -> int:
    minisplits = load_stage1_minisplits(MINISPLIT_PATH)

    point_ds = Stage1PointOdysseyDataset(
        data_root=DATA_ROOT,
        split="train_mini",
        minisplit_records=_records_for(minisplits, "pointodyssey", "train_mini"),
    )
    kubric_ds = Stage1KubricDataset(
        data_root=DATA_ROOT,
        split="train_mini",
        minisplit_records=_records_for(minisplits, "kubric", "train_mini"),
    )
    tapvid_ds = Stage1TapVidDataset(
        split="eval_mini",
        minisplit_records=_records_for(minisplits, "tapvid", "eval_mini"),
    )
    tapvid3d_ds = Stage1TapVid3DDataset(
        data_root=DATA_ROOT,
        split="eval_mini",
        minisplit_records=_records_for(minisplits, "tapvid3d", "eval_mini"),
    )

    dataset_to_sample = {
        "pointodyssey": point_ds[0],
        "kubric": kubric_ds[0],
        "tapvid": tapvid_ds[0],
        "tapvid3d": tapvid3d_ds[0],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    visual_results: List[Dict[str, Any]] = []
    for dataset_name, sample in dataset_to_sample.items():
        out_path = OUTPUT_DIR / f"{dataset_name}_stage1_visual_smoke_20260408.png"
        visual_results.append(_render_dataset_sample(dataset_name, sample, out_path))

    lines = [
        "# Stage 1 Visual Smoke (2026-04-08)",
        "",
        f"- generated_at_utc: {now_iso()}",
        f"- minisplit_path: {MINISPLIT_PATH}",
        f"- output_dir: {OUTPUT_DIR}",
        "",
        "| dataset | ok | note | output |",
        "|---|---:|---|---|",
    ]

    for row in visual_results:
        lines.append(
            f"| {row['dataset']} | {str(row['ok']).lower()} | {row['note']} | {row['output']} |"
        )

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[visual-smoke] wrote: {DOC_PATH}")
    for row in visual_results:
        print(f"[visual-smoke] {row['dataset']}: ok={row['ok']} note={row['note']} output={row['output']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
