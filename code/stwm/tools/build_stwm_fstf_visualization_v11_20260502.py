#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF Visualization V11", ""]
    for key, value in payload.items():
        if key != "assets" and (isinstance(value, (str, int, float, bool)) or value is None):
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Assets")
    for asset in payload.get("assets", []):
        lines.append(f"- {asset.get('path')}: {asset.get('description')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def svg_card(path: Path, title: str, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    escaped = [r.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for r in rows]
    height = 80 + 28 * len(escaped)
    text = "\n".join(
        f'<text x="32" y="{80 + i * 28}" font-family="monospace" font-size="18" fill="#183028">{row}</text>'
        for i, row in enumerate(escaped)
    )
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="{height}" viewBox="0 0 1280 {height}">
<rect width="1280" height="{height}" fill="#f4efe4"/>
<rect x="20" y="20" width="1240" height="{height - 40}" rx="24" fill="#fffaf0" stroke="#183028" stroke-width="3"/>
<text x="32" y="52" font-family="monospace" font-size="24" font-weight="700" fill="#183028">{title}</text>
{text}
</svg>
""",
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-report", default="reports/stwm_fstf_scaling_v11_prototype_c32_seed456_eval_20260502.json")
    p.add_argument("--figure-dir", default="assets/figures/stwm_fstf_rollout_v11")
    p.add_argument("--video-dir", default="assets/videos/stwm_fstf_rollout_v11")
    p.add_argument("--output", default="reports/stwm_fstf_visualization_v11_20260502.json")
    p.add_argument("--doc", default="docs/STWM_FSTF_VISUALIZATION_V11_20260502.md")
    args = p.parse_args()
    eval_path = Path(args.eval_report)
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
    metrics = eval_payload.get("metrics", {})
    figure_dir = Path(args.figure_dir)
    video_dir = Path(args.video_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    assets = []
    overview = figure_dir / "fstf_v11_rollout_metric_storyboard.svg"
    svg_card(
        overview,
        "STWM-FSTF V11 8-step rollout storyboard",
        [
            "Input: video-derived trace units + observed semantic memory (frozen cache).",
            "Output: future semantic prototype field over H=8 trace units.",
            f"Eval report: {eval_path}",
            f"overall_top5={metrics.get('proto_top5', 'NA')}",
            f"changed_top5={metrics.get('changed_subset_top5', 'NA')}",
            f"stable_top5={metrics.get('stable_subset_top5', 'NA')}",
            "Raw-frame MP4/GIF rendering remains blocked unless raw frame paths are rehydrated.",
        ],
    )
    assets.append({"path": str(overview), "description": "Metric storyboard for the 8-step FSTF rollout output contract."})
    boundary = figure_dir / "fstf_v11_claim_boundary.svg"
    svg_card(
        boundary,
        "Visualization claim boundary",
        [
            "Allowed: semantic trace-unit field visualization from frozen video-derived caches.",
            "Not claimed: raw-video end-to-end training.",
            "Not claimed: dense trace field until K16/K32 scaling completes.",
            "Not claimed: long-horizon H16/H24 until H scaling cache/eval completes.",
        ],
    )
    assets.append({"path": str(boundary), "description": "Claim-boundary figure for visualization limitations."})
    payload = {
        "audit_name": "stwm_fstf_visualization_v11",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figure_dir": str(figure_dir),
        "video_dir": str(video_dir),
        "assets": assets,
        "actual_svg_figures_generated": True,
        "actual_mp4_or_gif_generated": False,
        "raw_observed_frames_included": False,
        "observed_trace_units_included": "described_from_cache",
        "copy_baseline_included": "metric_storyboard_only",
        "stwm_prediction_included": "metric_storyboard_only",
        "gt_future_semantic_target_included": "metric_storyboard_only",
        "visualization_status": "partial_cache_based_figures; raw-frame video blocked",
        "blocking_reason_for_video": "V11 scaling eval artifacts do not carry raw frame image paths or per-step rendered overlays.",
    }
    write_json(Path(args.output), payload)
    write_doc(Path(args.doc), payload)


if __name__ == "__main__":
    main()
