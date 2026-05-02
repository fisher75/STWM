#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CACHE_BASE = ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16"
VIDEO_DIR = ROOT / "assets/videos/stwm_real_teacher_object_dense_v16"
LOG_DIR = ROOT / "logs/stwm_cotracker_object_dense_teacher_v16_20260502"


def _jsonable(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _sha256(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            read += len(chunk)
            if limit_bytes is not None and read >= limit_bytes:
                break
    return h.hexdigest()


def _file_rows(paths: list[Path], checksum_limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    for p in paths:
        stat = p.stat()
        rows.append(
            {
                "path": str(p.relative_to(ROOT)),
                "size_bytes": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": _sha256(p, checksum_limit),
                "checksum_partial": checksum_limit is not None,
            }
        )
    return rows


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM Real Teacher Cache Artifact Manifest V16", ""]
    for key in ["cache_file_count", "video_file_count", "log_file_count", "total_cache_size_bytes", "artifact_pack_path", "exported_zip_contains_caches_videos"]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    cache_files = sorted(CACHE_BASE.glob("M*_H*/*/*.npz"))
    video_files = sorted(VIDEO_DIR.glob("*.gif")) if VIDEO_DIR.exists() else []
    log_files = sorted(LOG_DIR.glob("*.log")) if LOG_DIR.exists() else []
    pack_path = ROOT / "artifacts/stwm_real_teacher_cache_manifest_v16_20260502.tar.gz"
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(pack_path, "w:gz") as tar:
        for p in [*video_files, *log_files]:
            tar.add(p, arcname=str(p.relative_to(ROOT)))
        # Do not pack all caches by default; they are reproducible and can be large. Include reports/docs/logs/videos.
        for p in sorted((ROOT / "reports").glob("*v16*20260502.json")) + sorted((ROOT / "docs").glob("*V16*20260502.md")):
            tar.add(p, arcname=str(p.relative_to(ROOT)))
    payload = {
        "audit_name": "stwm_real_teacher_cache_artifact_manifest_v16",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_base": str(CACHE_BASE.relative_to(ROOT)),
        "cache_file_count": len(cache_files),
        "video_dir": str(VIDEO_DIR.relative_to(ROOT)),
        "video_file_count": len(video_files),
        "log_dir": str(LOG_DIR.relative_to(ROOT)),
        "log_file_count": len(log_files),
        "total_cache_size_bytes": sum(p.stat().st_size for p in cache_files),
        "total_video_size_bytes": sum(p.stat().st_size for p in video_files),
        "cache_paths_by_combo": {
            combo.name: len(list(combo.glob("*/*.npz"))) for combo in sorted(CACHE_BASE.glob("M*_H*")) if combo.is_dir()
        },
        "cache_file_samples": _file_rows(cache_files[:20], checksum_limit=2 * 1024 * 1024),
        "video_files": _file_rows(video_files),
        "log_files": _file_rows(log_files),
        "artifact_pack_path": str(pack_path.relative_to(ROOT)),
        "artifact_pack_size_bytes": pack_path.stat().st_size,
        "exported_zip_contains_caches_videos": False,
        "exported_zip_note": "assets/ and artifacts/ are gitignored; live repo contains caches/videos, exported snapshots may omit them.",
    }
    _dump(ROOT / "reports/stwm_real_teacher_cache_artifact_manifest_v16_20260502.json", payload)
    _write_doc(ROOT / "docs/STWM_REAL_TEACHER_CACHE_ARTIFACT_MANIFEST_V16_20260502.md", payload)
    print("reports/stwm_real_teacher_cache_artifact_manifest_v16_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
