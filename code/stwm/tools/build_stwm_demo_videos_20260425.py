#!/usr/bin/env python3
"""Build STWM demo video assets from existing visual manifest/case ids.

Videos are report-derived schematic clips when raw frame paths are absent.
They are explicitly not CARLA/closed-loop or raw-video evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stwm_vis_utils_20260425 import (
    DOCS,
    FIG_VIDEO,
    REPORTS,
    ROOT,
    draw_video_frame,
    dump_json,
    ensure_dirs,
    load_json,
    make_video_from_frames,
    write_text,
)


VIDEO_SPECS = [
    ("clip01_confuser_crossing", "Confuser crossing", "teacher-only wrong · STWM trace belief correct"),
    ("clip02_occlusion_reacquisition", "Occlusion reacquisition", "target disappears and is reacquired after gap"),
    ("clip03_true_ood_case", "True OOD hard case", "held-out context-preserving case · report-derived schematic"),
    ("clip04_counterfactual_trace_effect", "Counterfactual trace effect", "before / after belief-level intervention"),
]


def case_pool(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    ids = manifest.get("self_check", {}).get("case_ids_used", []) or []
    if not ids:
        ids = [f"report-derived-case-{i}" for i in range(8)]
    cases = []
    for idx, cid in enumerate(ids):
        cases.append(
            {
                "case_id": cid,
                "case_type": VIDEO_SPECS[idx % len(VIDEO_SPECS)][1],
            }
        )
    return cases


def build_clip(out_name: str, title: str, subtitle: str, case: dict[str, Any], counterfactual: bool = False) -> dict[str, Any]:
    frames = []
    for i in range(18):
        if counterfactual:
            before_after = "before" if i < 9 else "after intervention"
        else:
            before_after = f"frame {i + 1:02d}"
        frames.append(draw_video_frame(title, subtitle, case, i, before_after=before_after))
    out_path = FIG_VIDEO / f"{out_name}.mp4"
    make_video_from_frames(frames, out_path, fps=6)
    return {
        "asset_id": out_name,
        "path": str(out_path.relative_to(ROOT)),
        "exists": out_path.exists(),
        "case_ids": [case.get("case_id")],
        "uses_raw_video_frames": False,
        "uses_report_derived_schematic_frames": True,
        "source_limitation": "Required source reports do not contain raw frame/video paths, so this mp4 is a schematic paper-ready v1 clip generated from report case ids and readout annotations.",
    }


def main() -> None:
    ensure_dirs()
    manifest_path = REPORTS / "stwm_visual_asset_manifest_20260425.json"
    manifest, err = load_json(manifest_path)
    if err or not isinstance(manifest, dict):
        raise SystemExit(f"Missing visual manifest, run build_stwm_paper_figures_20260425.py first: {err}")
    cases = case_pool(manifest)
    videos = []
    for idx, (out_name, title, subtitle) in enumerate(VIDEO_SPECS):
        videos.append(build_clip(out_name, title, subtitle, cases[idx % len(cases)], counterfactual="counterfactual" in out_name))
    # Overview video: concatenate schematic phases into one mp4.
    overview_frames = []
    for idx, spec in enumerate(VIDEO_SPECS):
        out_name, title, subtitle = spec
        for phase in range(8):
            overview_frames.append(draw_video_frame(f"STWM demo: {title}", subtitle, cases[idx % len(cases)], phase, before_after=f"segment {idx+1}/4"))
    overview_path = FIG_VIDEO / "STWM_demo_20260425.mp4"
    make_video_from_frames(overview_frames, overview_path, fps=6)
    videos.append(
        {
            "asset_id": "STWM_demo_20260425",
            "path": str(overview_path.relative_to(ROOT)),
            "exists": overview_path.exists(),
            "case_ids": [c.get("case_id") for c in cases[:4]],
            "uses_raw_video_frames": False,
            "uses_report_derived_schematic_frames": True,
            "source_limitation": "Overview schematic assembled from the four generated report-derived clips.",
        }
    )
    manifest["videos"] = videos
    manifest.setdefault("self_check", {})
    manifest["self_check"]["video_all_generated"] = all(v["exists"] for v in videos[:4])
    manifest["self_check"]["video_overview_generated"] = videos[-1]["exists"]
    manifest["self_check"]["video_source_raw_frames_available"] = False
    manifest["self_check"]["video_exact_missing_source"] = "No raw frame/video path references were present in required JSON reports."
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    dump_json(manifest_path, manifest)
    # Update markdown manifest with video table.
    md_path = DOCS / "STWM_VISUAL_ASSET_MANIFEST_20260425.md"
    existing = md_path.read_text() if md_path.exists() else "# STWM Visual Asset Manifest 20260425\n"
    if "## Videos" in existing:
        existing = existing.split("## Videos")[0].rstrip() + "\n\n"
    existing += "## Videos\n\n| asset | path | raw frames? | generated |\n|---|---|---:|---:|\n"
    for v in videos:
        existing += f"| {v['asset_id']} | `{v['path']}` | {v['uses_raw_video_frames']} | {v['exists']} |\n"
    existing += f"\n- Video all generated = `{manifest['self_check']['video_all_generated']}`\n- Video source raw frames available = `False`\n- Video missing source = `{manifest['self_check']['video_exact_missing_source']}`\n"
    write_text(md_path, existing)
    print(
        {
            "video_all_generated": manifest["self_check"]["video_all_generated"],
            "videos": [v["path"] for v in videos],
            "manifest": "reports/stwm_visual_asset_manifest_20260425.json",
        }
    )


if __name__ == "__main__":
    main()
