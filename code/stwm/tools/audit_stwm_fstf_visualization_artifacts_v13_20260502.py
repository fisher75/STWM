#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF Visualization Artifact Audit V13", ""]
    for key in [
        "visualization_report_exists",
        "actual_visual_assets_in_snapshot",
        "paper_ready_visualization_pack_ready",
        "video_count",
        "figure_count",
        "artifact_pack_path",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    lines.append("## Checks")
    for key, value in payload.get("required_content_checks", {}).items():
        lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _collect(root: Path, suffixes: set[str]) -> list[Path]:
    if not root.exists():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualization-report", default="reports/stwm_fstf_visualization_v12_20260502.json")
    parser.add_argument("--video-dir", default="assets/videos/stwm_fstf_rollout_v12")
    parser.add_argument("--figure-dir", default="assets/figures/stwm_fstf_rollout_v12")
    parser.add_argument("--artifact-pack", default="artifacts/stwm_fstf_visualization_pack_v13_20260502.tar.gz")
    parser.add_argument("--output", default="reports/stwm_fstf_visualization_artifact_audit_v13_20260502.json")
    parser.add_argument("--doc", default="docs/STWM_FSTF_VISUALIZATION_ARTIFACT_AUDIT_V13_20260502.md")
    args = parser.parse_args()
    report = _load(Path(args.visualization_report))
    video_dir = Path(args.video_dir)
    figure_dir = Path(args.figure_dir)
    videos = _collect(video_dir, {".gif", ".mp4", ".webm"})
    figures = _collect(figure_dir, {".png", ".pdf", ".svg", ".jpg", ".jpeg"})
    files = videos + figures + ([Path(args.visualization_report)] if Path(args.visualization_report).exists() else [])
    file_rows = [
        {
            "path": str(p),
            "size_bytes": int(p.stat().st_size),
            "mtime": p.stat().st_mtime,
            "sha256": _sha256(p),
        }
        for p in files
    ]
    checks = {
        "raw_frame_paths_included": bool(report.get("raw_observed_frames_included")),
        "observed_trace_overlay_included": bool(report.get("observed_trace_units_included")),
        "copy_baseline_included": bool(report.get("copy_baseline_included")),
        "stwm_prediction_included": bool(report.get("stwm_prediction_included")),
        "gt_future_target_included": bool(report.get("gt_future_semantic_target_included")),
        "changed_unit_highlight_included": bool(report.get("changed_unit_highlight_included")),
        "horizon_at_least_8": int(report.get("horizon_steps_rendered", 0) or 0) >= 8,
    }
    pack = Path(args.artifact_pack)
    pack.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(pack, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=str(path))
    pack_ready = bool(videos and figures and all(checks.values()) and pack.exists() and pack.stat().st_size > 0)
    payload = {
        "audit_name": "stwm_fstf_visualization_artifact_audit_v13",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "visualization_report": args.visualization_report,
        "visualization_report_exists": bool(report),
        "video_dir": str(video_dir),
        "figure_dir": str(figure_dir),
        "video_count": len(videos),
        "figure_count": len(figures),
        "files": file_rows,
        "required_content_checks": checks,
        "actual_visual_assets_in_snapshot": bool(videos and figures),
        "paper_ready_visualization_pack_ready": pack_ready,
        "artifact_pack_path": str(pack),
        "artifact_pack_size_bytes": int(pack.stat().st_size) if pack.exists() else 0,
        "artifact_pack_sha256": _sha256(pack) if pack.exists() else "",
    }
    _dump(Path(args.output), payload)
    _write_doc(Path(args.doc), payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
