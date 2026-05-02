#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def _rel(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(ROOT)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _file_row(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": _rel(path),
        "size_bytes": int(st.st_size),
        "mtime": float(st.st_mtime),
        "sha256": _sha256(path),
    }


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM Repo Redundancy Cleanup V13", ""]
    for key in [
        "execute",
        "shard_report_count",
        "legacy_log_count",
        "python_cache_dir_count",
        "python_bytecode_file_count",
        "scratch_path_count",
        "deleted_file_count",
        "deleted_dir_count",
        "archived_file_count",
        "archived_size_bytes",
        "reclaimed_raw_bytes_estimate",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    lines.append("## Archives")
    for key in ["shard_report_archive", "legacy_log_archive"]:
        val = payload.get(key)
        if val:
            lines.append(f"- {key}: `{val}`")
    lines.append("")
    lines.append("## Preserved Core Areas")
    for item in payload.get("preserved_core_areas", []):
        lines.append(f"- `{item}`")
    lines.append("")
    lines.append("## Policy")
    for item in payload.get("cleanup_policy", []):
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _archive(paths: list[Path], archive_path: Path, *, execute: bool) -> dict[str, Any]:
    rows = [_file_row(p) for p in paths if p.exists() and p.is_file()]
    if execute and rows:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "w:gz") as tar:
            for row in rows:
                tar.add(ROOT / row["path"], arcname=row["path"])
    return {
        "archive_path": _rel(archive_path),
        "file_count": len(rows),
        "source_size_bytes": int(sum(row["size_bytes"] for row in rows)),
        "archive_size_bytes": int(archive_path.stat().st_size) if archive_path.exists() else 0,
        "files": rows,
    }


def _safe_unlink(path: Path, *, execute: bool) -> bool:
    if not _inside_repo(path):
        raise RuntimeError(f"refusing to remove path outside repo: {path}")
    if execute and path.exists():
        path.unlink()
    return True


def _safe_rmtree(path: Path, *, execute: bool) -> bool:
    if not _inside_repo(path):
        raise RuntimeError(f"refusing to remove path outside repo: {path}")
    if execute and path.exists():
        shutil.rmtree(path)
    return True


def _collect_shard_reports() -> list[Path]:
    out: set[Path] = set()
    for base in [ROOT / "reports", ROOT / "docs"]:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            name = path.name
            if path.is_file() and ("shard" in name.lower() or "SHARD" in name):
                out.add(path)
    return sorted(out)


def _collect_legacy_logs() -> list[Path]:
    logs = ROOT / "logs"
    if not logs.exists():
        return []
    out = []
    for path in logs.rglob("*.log"):
        rel = _rel(path)
        # Keep recent FSTF proof logs in place because artifact audits reference them directly.
        if "fstf" in rel.lower() or "stwm_fstf" in rel.lower():
            continue
        if path.stat().st_size >= 1_000_000 or path.name.startswith("download_") or path.name.startswith("d1_fc_"):
            out.append(path)
    return sorted(out)


def _collect_python_cache_dirs() -> list[Path]:
    names = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    scan_roots = [
        ROOT / "code",
        ROOT / "scripts",
        ROOT / "configs",
        ROOT / "env",
        ROOT / "tests",
    ]
    out: list[Path] = []
    for base in scan_roots:
        if base.exists():
            out.extend(p for p in base.rglob("*") if p.is_dir() and p.name in names)
    return sorted(set(out))


def _collect_bytecode() -> list[Path]:
    scan_roots = [
        ROOT / "code",
        ROOT / "scripts",
        ROOT / "configs",
        ROOT / "env",
        ROOT / "tests",
    ]
    out: list[Path] = []
    for base in scan_roots:
        if base.exists():
            out.extend(p for p in base.rglob("*") if p.is_file() and p.suffix in {".pyc", ".pyo"})
    return sorted(set(out))


def _collect_scratch_paths() -> list[Path]:
    candidates = [
        ROOT / "tmp",
        ROOT / "outputs" / "profiler",
        ROOT / "outputs" / "run_status",
    ]
    return [p for p in candidates if p.exists()]


def _dir_size(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    execute = bool(args.execute)

    shard_reports = _collect_shard_reports()
    legacy_logs = _collect_legacy_logs()
    cache_dirs = _collect_python_cache_dirs()
    bytecode = _collect_bytecode()
    scratch_paths = _collect_scratch_paths()
    scratch_rows = [{"path": _rel(p), "size_bytes": _dir_size(p), "type": "dir" if p.is_dir() else "file"} for p in scratch_paths]

    shard_archive = ROOT / "artifacts" / "stwm_redundant_shard_reports_v13_20260502.tar.gz"
    log_archive = ROOT / "artifacts" / "stwm_legacy_large_logs_v13_20260502.tar.gz"
    shard_archive_info = _archive(shard_reports, shard_archive, execute=execute)
    log_archive_info = _archive(legacy_logs, log_archive, execute=execute)

    deleted_files: list[str] = []
    deleted_dirs: list[str] = []
    if execute:
        for path in shard_reports + legacy_logs + bytecode:
            if path.exists() and path.is_file():
                _safe_unlink(path, execute=True)
                deleted_files.append(_rel(path))
        # Remove cache dirs after bytecode files; dirs may already be empty/non-empty.
        for path in cache_dirs:
            if path.exists():
                _safe_rmtree(path, execute=True)
                deleted_dirs.append(_rel(path))
        for path in scratch_paths:
            if path.exists():
                _safe_rmtree(path, execute=True)
                deleted_dirs.append(_rel(path))

    archive_bytes = int(shard_archive_info["archive_size_bytes"]) + int(log_archive_info["archive_size_bytes"])
    raw_bytes = (
        int(shard_archive_info["source_size_bytes"])
        + int(log_archive_info["source_size_bytes"])
        + sum(int(x["size_bytes"]) for x in scratch_rows)
        + sum(int(p.stat().st_size) for p in bytecode if p.exists())
    )
    payload = {
        "audit_name": "stwm_repo_redundancy_cleanup_v13",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "execute": execute,
        "cleanup_policy": [
            "Preserve data, models, outputs/cache, outputs/checkpoints, final assets, scripts, source code, and key summary reports.",
            "Archive shard-level reports/docs before removing scattered originals.",
            "Archive large legacy logs before removing originals; keep FSTF proof logs in place.",
            "Delete Python bytecode/cache and scratch smoke/profiler outputs because they are reproducible derived state.",
        ],
        "preserved_core_areas": [
            "data/",
            "models/",
            "outputs/cache/",
            "outputs/checkpoints/",
            "assets/",
            "artifacts/stwm_fstf_visualization_pack_v13_20260502.tar.gz",
            "reports/*summary*.json and final V8/V10/V12/V13 reports",
            "logs/fstf_* and logs/stwm_fstf_*",
        ],
        "shard_report_count": len(shard_reports),
        "legacy_log_count": len(legacy_logs),
        "python_cache_dir_count": len(cache_dirs),
        "python_bytecode_file_count": len(bytecode),
        "scratch_path_count": len(scratch_paths),
        "scratch_paths": scratch_rows,
        "shard_report_archive": shard_archive_info["archive_path"],
        "legacy_log_archive": log_archive_info["archive_path"],
        "archived_file_count": int(shard_archive_info["file_count"]) + int(log_archive_info["file_count"]),
        "archived_size_bytes": archive_bytes,
        "reclaimed_raw_bytes_estimate": max(raw_bytes - archive_bytes, 0) if execute else raw_bytes,
        "deleted_file_count": len(deleted_files),
        "deleted_dir_count": len(deleted_dirs),
        "deleted_files_sample": deleted_files[:100],
        "deleted_dirs_sample": deleted_dirs[:100],
        "shard_archive_manifest": shard_archive_info,
        "legacy_log_archive_manifest": log_archive_info,
        "dry_run_note": "Run with --execute to archive and remove candidates." if not execute else "",
    }
    _dump(ROOT / "reports" / "stwm_repo_redundancy_cleanup_v13_20260502.json", payload)
    _write_doc(ROOT / "docs" / "STWM_REPO_REDUNDANCY_CLEANUP_V13_20260502.md", payload)
    print(ROOT / "reports" / "stwm_repo_redundancy_cleanup_v13_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
