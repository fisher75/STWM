#!/usr/bin/env python3
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path("/home/chen034/workspace/stwm").resolve()
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
DATE_TAG = "20260414"
SUMMARY_JSON = REPORTS / f"storage_direct_cleanup_summary_{DATE_TAG}.json"
ACTIVE_JSON = REPORTS / f"storage_cleanup_active_processes_recheck_{DATE_TAG}.json"
BEFORE_JSON = REPORTS / f"storage_audit_before_direct_cleanup_{DATE_TAG}.json"
AFTER_JSON = REPORTS / f"storage_audit_after_direct_cleanup_{DATE_TAG}.json"
SAFE_DELETE_JSON = REPORTS / f"storage_cleanup_direct_safe_delete_manifest_{DATE_TAG}.json"
CHECKPOINT_DELETE_JSON = REPORTS / f"storage_cleanup_direct_checkpoint_delete_manifest_{DATE_TAG}.json"
TMP_DELETE_JSON = REPORTS / f"storage_cleanup_direct_tmp_delete_manifest_{DATE_TAG}.json"
LOGS_DELETE_JSON = REPORTS / f"storage_cleanup_direct_logs_delete_manifest_{DATE_TAG}.json"
RAW_ARCHIVE_JSON = REPORTS / f"storage_cleanup_raw_archive_validation_{DATE_TAG}.json"
REVIEW_JSON = REPORTS / f"storage_cleanup_review_required_after_direct_cleanup_{DATE_TAG}.json"
BEFORE_MD = DOCS / f"STWM_STORAGE_AUDIT_BEFORE_DIRECT_CLEANUP_{DATE_TAG}.md"
SUMMARY_MD = DOCS / f"STWM_STORAGE_DIRECT_CLEANUP_SUMMARY_{DATE_TAG}.md"

PROTECTED_TOP_LEVEL = {"code", "docs", "manifests", "reports", "third_party"}
FAST_SKIP_TOP_LEVEL_FOR_CACHE_SCAN = {"data", "models", "third_party"}
PROTECTED_CHECKPOINT_DIRS = {
    ROOT / "outputs/checkpoints/stage1_v2_longtrain_220m_mainline_20260408",
}
KEEP_CHECKPOINT_NAMES = {"best.pt", "latest.pt", "best_semantic_hard.pt", "semantic_hard_best.pt"}
INTERMEDIATE_PATTERNS = (
    "step_*.pt",
    "epoch_*.pt",
    "global_step_*.pt",
    "intermediate_*.pt",
    "optimizer*.pt",
    "*optimizer*state*.pt",
)
RAW_ARCHIVE_SPECS = [
    {
        "dataset_name": "BURST/TAO train",
        "archive_path": ROOT / "data/raw/burst/1-TAO_TRAIN.zip",
        "extracted_root": ROOT / "data/external/burst",
        "required_paths": [
            ROOT / "data/external/burst/images/train/frames/train",
            ROOT / "data/external/burst/annotations/train/train.json",
        ],
        "dependency_kind": "optional_extension",
    },
    {
        "dataset_name": "BURST/TAO val",
        "archive_path": ROOT / "data/raw/burst/2-TAO_VAL.zip",
        "extracted_root": ROOT / "data/external/burst",
        "required_paths": [
            ROOT / "data/external/burst/images/val/frames/val",
            ROOT / "data/external/burst/annotations/val/all_classes.json",
        ],
        "dependency_kind": "optional_extension",
    },
    {
        "dataset_name": "BURST/TAO test",
        "archive_path": ROOT / "data/raw/burst/3-TAO_TEST.zip",
        "extracted_root": ROOT / "data/external/burst",
        "required_paths": [
            ROOT / "data/external/burst/images/test/frames/test",
            ROOT / "data/external/burst/annotations/test/all_classes.json",
        ],
        "dependency_kind": "optional_extension",
    },
    {
        "dataset_name": "VSPW",
        "archive_path": ROOT / "data/raw/vspw/VSPW_data.tar",
        "extracted_root": ROOT / "data/external/vspw/VSPW",
        "required_paths": [
            ROOT / "data/external/vspw/VSPW/train.txt",
            ROOT / "data/external/vspw/VSPW/val.txt",
        ],
        "dependency_kind": "core",
    },
    {
        "dataset_name": "VIPSeg",
        "archive_path": ROOT / "data/raw/vipseg/vipseg_archive.zip",
        "extracted_root": ROOT / "data/external/vipseg/VIPSeg",
        "required_paths": [
            ROOT / "data/external/vipseg/VIPSeg/train.txt",
            ROOT / "data/external/vipseg/VIPSeg/val.txt",
        ],
        "dependency_kind": "core",
    },
    {
        "dataset_name": "VISOR",
        "archive_path": ROOT / "data/raw/visor/visor_complete.zip",
        "extracted_root": ROOT / "data/external/visor/2v6cgv1x04ol22qp9rm9x2j6a7",
        "required_paths": [
            ROOT / "data/external/visor/2v6cgv1x04ol22qp9rm9x2j6a7",
        ],
        "dependency_kind": "manual_gate_extension",
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(cmd: list[str] | str, *, shell: bool = False, timeout: int = 120) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=timeout)
        return {
            "cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
            "returncode": int(proc.returncode),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {
            "cmd": cmd if isinstance(cmd, str) else " ".join(cmd),
            "returncode": -1,
            "stdout": "",
            "stderr": repr(exc),
        }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def fmt_bytes(num: int | float) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if abs(value) < 1024.0 or unit == "PB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def top_level_name(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).parts[0]
    except Exception:
        return ""


def is_protected_top(path: Path) -> bool:
    return top_level_name(path) in PROTECTED_TOP_LEVEL


def file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def path_size(path: Path) -> int:
    if not path.exists():
        return 0
    proc = run_cmd(["du", "-sb", str(path)], timeout=1800)
    if proc["returncode"] == 0 and proc["stdout"].strip():
        try:
            return int(proc["stdout"].split()[0])
        except Exception:
            pass
    if path.is_file():
        return file_size(path)
    total = 0
    for base, _dirs, files in os.walk(path):
        for name in files:
            total += file_size(Path(base) / name)
    return total


def mtime_age_days(path: Path) -> float:
    try:
        return max(0.0, (time.time() - path.stat().st_mtime) / 86400.0)
    except Exception:
        return 0.0


def walk_files(base: Path) -> Iterable[Path]:
    for root, dirs, files in os.walk(base):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if not is_protected_top(root_path / name)]
        for name in files:
            p = root_path / name
            if p.is_file() and not p.is_symlink():
                yield p


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def delete_path(path: Path) -> tuple[bool, int, str]:
    size = path_size(path)
    try:
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()
        return True, size, ""
    except Exception as exc:
        return False, 0, repr(exc)


def grep_active_run_names_from_ps(ps_stdout: str) -> set[str]:
    names: set[str] = set()
    for match in re.findall(r"(stage2_[A-Za-z0-9_]+_\d{8})", ps_stdout):
        names.add(match)
    return names


def active_run_names_from_summary_reports() -> set[str]:
    active: set[str] = set()
    for report in REPORTS.glob("*summary*.json"):
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = data.get("run_rows", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            if str(row.get("status", "")).lower() == "running":
                name = str(row.get("run_name", "")).strip()
                if name:
                    active.add(name)
    return active


def active_process_recheck() -> dict[str, Any]:
    ps = run_cmd(
        'ps -eo pid,ppid,etimes,cmd | grep -E "train_tracewm|stage2_|tracewm|stwm" | grep -v grep || true',
        shell=True,
        timeout=30,
    )
    nvidia = run_cmd(["nvidia-smi"], timeout=60)
    tmux = run_cmd('tmux ls 2>/dev/null || true', shell=True, timeout=30)
    ps_lines = [line for line in ps["stdout"].splitlines() if line.strip()]
    stwm_lines = [line for line in ps_lines if str(ROOT) in line or "tracewm" in line or "stwm" in line]
    running_summary_names = sorted(active_run_names_from_summary_reports())
    active_run_names = sorted(grep_active_run_names_from_ps(ps["stdout"]).union(running_summary_names))
    payload = {
        "generated_at_utc": now_iso(),
        "commands": {
            "ps": ps,
            "nvidia_smi": nvidia,
            "tmux_ls": tmux,
        },
        "active_process_detected": bool(stwm_lines),
        "active_run_detected": bool(active_run_names),
        "active_process_lines": stwm_lines,
        "active_run_names": active_run_names,
        "active_run_names_from_summary_reports": running_summary_names,
    }
    write_json(ACTIVE_JSON, payload)
    return payload


def safe_delete_candidates() -> list[Path]:
    candidates: set[Path] = set()
    for root, dirs, files in os.walk(ROOT):
        root_path = Path(root)
        if is_protected_top(root_path):
            dirs[:] = []
            continue
        if top_level_name(root_path) in FAST_SKIP_TOP_LEVEL_FOR_CACHE_SCAN:
            dirs[:] = []
            continue
        dirs[:] = [name for name in dirs if not is_protected_top(root_path / name)]
        for name in list(dirs):
            p = root_path / name
            if name in {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints"}:
                candidates.add(p)
        for name in files:
            p = root_path / name
            if is_protected_top(p):
                continue
            if name.endswith((".pyc", ".pyo")) or name == ".DS_Store" or fnmatch.fnmatch(name, "core.*"):
                candidates.add(p)
            elif name.endswith((".tmp", ".temp", ".bak", ".swp")) or name.endswith("~"):
                if mtime_age_days(p) > 3.0:
                    candidates.add(p)
    return sorted(candidates, key=lambda item: str(item))


def candidate_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        size = path_size(path)
        rows.append(
            {
                "path": rel(path),
                "bytes": size,
                "human": fmt_bytes(size),
            }
        )
    return rows


def top_level_sizes() -> list[dict[str, Any]]:
    rows = []
    for child in sorted(ROOT.iterdir()):
        size = path_size(child)
        rows.append({"path": rel(child), "bytes": size, "human": fmt_bytes(size)})
    return sorted(rows, key=lambda row: int(row["bytes"]), reverse=True)


def top_raw_archives() -> list[dict[str, Any]]:
    rows = []
    raw_root = ROOT / "data/raw"
    if not raw_root.exists():
        return rows
    for path in raw_root.rglob("*"):
        if path.is_file():
            size = file_size(path)
            rows.append({"path": rel(path), "bytes": size, "human": fmt_bytes(size)})
    return sorted(rows, key=lambda row: int(row["bytes"]), reverse=True)[:50]


def top_largest_files(limit: int = 200) -> list[dict[str, Any]]:
    cmd = (
        f"find {shell_quote(str(ROOT))} -type f -printf '%s\\t%p\\n' 2>/dev/null "
        f"| sort -nr -k1,1 | head -n {int(limit)}"
    )
    proc = run_cmd(cmd, shell=True, timeout=7200)
    rows = []
    if proc["returncode"] == 0:
        for line in proc["stdout"].splitlines():
            if "\t" not in line:
                continue
            size_text, path_text = line.split("\t", 1)
            try:
                size = int(size_text.strip())
            except Exception:
                continue
            rows.append({"path": rel(Path(path_text.strip())), "bytes": size, "human": fmt_bytes(size)})
        if rows:
            return rows[:limit]
    rows = []
    for path in ROOT.rglob("*"):
        if path.is_file() and not path.is_symlink():
            rows.append({"path": rel(path), "bytes": file_size(path), "human": fmt_bytes(file_size(path))})
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    return rows[:limit]


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def before_or_after_audit(*, active: dict[str, Any], safe_candidates: list[Path], review_candidates: list[dict[str, Any]], target_json: Path, target_md: Path) -> dict[str, Any]:
    total_bytes = path_size(ROOT)
    outputs_ckpt_bytes = path_size(ROOT / "outputs/checkpoints")
    logs_bytes = path_size(ROOT / "logs")
    outputs_tmp_bytes = path_size(ROOT / "outputs/tmp")
    payload = {
        "generated_at_utc": now_iso(),
        "root": str(ROOT),
        "du_sh": run_cmd(["du", "-sh", str(ROOT)], timeout=1800),
        "du_sb": run_cmd(["du", "-sb", str(ROOT)], timeout=1800),
        "total_bytes": total_bytes,
        "total_human": fmt_bytes(total_bytes),
        "active_run_detected": bool(active.get("active_run_detected", False)),
        "active_run_names": list(active.get("active_run_names", [])),
        "top_level_directory_sizes": top_level_sizes(),
        "outputs_checkpoints_total_bytes": outputs_ckpt_bytes,
        "outputs_checkpoints_total_human": fmt_bytes(outputs_ckpt_bytes),
        "logs_total_bytes": logs_bytes,
        "logs_total_human": fmt_bytes(logs_bytes),
        "outputs_tmp_total_bytes": outputs_tmp_bytes,
        "outputs_tmp_total_human": fmt_bytes(outputs_tmp_bytes),
        "data_raw_top_archives": top_raw_archives(),
        "top_200_largest_files": top_largest_files(200),
        "safe_delete_candidates": {
            "count": len(safe_candidates),
            "bytes": sum(path_size(path) for path in safe_candidates),
            "rows": candidate_rows(safe_candidates)[:500],
        },
        "validation_required_candidates": review_candidates,
    }
    write_json(target_json, payload)
    lines = [
        "# STWM Storage Audit Before Direct Cleanup 20260414" if "before" in target_json.name else "# STWM Storage Audit After Direct Cleanup 20260414",
        "",
        f"- root: `{ROOT}`",
        f"- total: {payload['total_human']}",
        f"- active_run_detected: {payload['active_run_detected']}",
        f"- active_run_names: {', '.join(payload['active_run_names']) or 'none'}",
        f"- outputs/checkpoints: {payload['outputs_checkpoints_total_human']}",
        f"- logs: {payload['logs_total_human']}",
        f"- outputs/tmp: {payload['outputs_tmp_total_human']}",
        "",
        "## Top-Level Sizes",
        "",
        "| path | size |",
        "|---|---:|",
    ]
    for row in payload["top_level_directory_sizes"][:20]:
        lines.append(f"| `{row['path']}` | {row['human']} |")
    write_md(target_md, lines)
    return payload


def checkpoint_keep_files(run_dir: Path) -> list[str]:
    keep: list[str] = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name in KEEP_CHECKPOINT_NAMES or ("semantic_hard" in path.name and path.suffix == ".pt"):
            keep.append(path.name)
    return keep


def is_intermediate_checkpoint(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in KEEP_CHECKPOINT_NAMES:
        return False
    if "semantic_hard" in path.name and path.suffix == ".pt":
        return False
    return any(fnmatch.fnmatch(path.name, pattern) for pattern in INTERMEDIATE_PATTERNS)


def completed_status_for_run(run_name: str) -> tuple[bool, Path]:
    final_json = REPORTS / f"{run_name}_final.json"
    if not final_json.exists():
        return False, final_json
    data = read_json(final_json)
    return str(data.get("status", "")).lower() == "completed", final_json


def prune_completed_checkpoints(active: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ckpt_root = ROOT / "outputs/checkpoints"
    active_runs = set(active.get("active_run_names", []))
    rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    total_deleted_bytes = 0
    total_deleted_count = 0
    if ckpt_root.exists():
        for run_dir in sorted(path for path in ckpt_root.iterdir() if path.is_dir()):
            run_name = run_dir.name
            completed, final_json = completed_status_for_run(run_name)
            best_exists = (run_dir / "best.pt").exists()
            latest_exists = (run_dir / "latest.pt").exists()
            keep_files = checkpoint_keep_files(run_dir)
            delete_candidates = sorted(path for path in run_dir.iterdir() if is_intermediate_checkpoint(path))
            delete_bytes = sum(file_size(path) for path in delete_candidates)
            row = {
                "run_name": run_name,
                "final_json_exists": final_json.exists(),
                "final_json_path": rel(final_json),
                "completed_status": completed,
                "best_exists": best_exists,
                "latest_exists": latest_exists,
                "active_run": run_name in active_runs,
                "protected_stage1_backbone": str(run_dir.resolve()) in {str(path.resolve()) for path in PROTECTED_CHECKPOINT_DIRS},
                "candidate_checkpoint_count": len(delete_candidates),
                "candidate_checkpoint_bytes": delete_bytes,
                "candidate_checkpoint_human": fmt_bytes(delete_bytes),
                "deleted_checkpoint_count": 0,
                "deleted_checkpoint_bytes": 0,
                "kept_files": keep_files,
                "skipped_reason_if_any": "",
            }
            if run_dir in PROTECTED_CHECKPOINT_DIRS:
                row["skipped_reason_if_any"] = "protected_stage1_frozen_backbone"
                review_rows.append(
                    {
                        "path": rel(run_dir),
                        "bytes": delete_bytes,
                        "human": fmt_bytes(delete_bytes),
                        "reason": row["skipped_reason_if_any"],
                    }
                )
                rows.append(row)
                continue
            if run_name in active_runs:
                row["skipped_reason_if_any"] = "active_run_directory"
                review_rows.append(
                    {
                        "path": rel(run_dir),
                        "bytes": delete_bytes,
                        "human": fmt_bytes(delete_bytes),
                        "reason": row["skipped_reason_if_any"],
                    }
                )
                rows.append(row)
                continue
            if not completed:
                row["skipped_reason_if_any"] = "final_json_missing_or_not_completed"
                review_rows.append(
                    {
                        "path": rel(run_dir),
                        "bytes": delete_bytes,
                        "human": fmt_bytes(delete_bytes),
                        "reason": row["skipped_reason_if_any"],
                    }
                )
                rows.append(row)
                continue
            if not best_exists or not latest_exists:
                row["skipped_reason_if_any"] = "best_or_latest_missing"
                review_rows.append(
                    {
                        "path": rel(run_dir),
                        "bytes": delete_bytes,
                        "human": fmt_bytes(delete_bytes),
                        "reason": row["skipped_reason_if_any"],
                    }
                )
                rows.append(row)
                continue
            deleted_files = []
            failed_files = []
            for path in delete_candidates:
                ok, size, error = delete_path(path)
                if ok:
                    deleted_files.append(rel(path))
                    row["deleted_checkpoint_count"] += 1
                    row["deleted_checkpoint_bytes"] += size
                    total_deleted_count += 1
                    total_deleted_bytes += size
                else:
                    failed_files.append({"path": rel(path), "error": error})
                    review_rows.append(
                        {
                            "path": rel(path),
                            "bytes": file_size(path),
                            "human": fmt_bytes(file_size(path)),
                            "reason": f"checkpoint_delete_failed:{error}",
                        }
                    )
            row["deleted_files"] = deleted_files
            row["failed_files"] = failed_files
            rows.append(row)
    payload = {
        "generated_at_utc": now_iso(),
        "active_run_names": sorted(active_runs),
        "deleted_checkpoint_count": total_deleted_count,
        "deleted_checkpoint_bytes": total_deleted_bytes,
        "deleted_checkpoint_human": fmt_bytes(total_deleted_bytes),
        "rows": rows,
    }
    write_json(CHECKPOINT_DELETE_JSON, payload)
    return payload, review_rows


def delete_safe_low_risk_cache() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = safe_delete_candidates()
    deleted = []
    failed = []
    total_deleted = 0
    for path in candidates:
        ok, size, error = delete_path(path)
        row = {"path": rel(path), "bytes": size if ok else path_size(path), "human": fmt_bytes(size if ok else path_size(path))}
        if ok:
            deleted.append(row)
            total_deleted += size
        else:
            row["error"] = error
            failed.append(row)
    payload = {
        "generated_at_utc": now_iso(),
        "deleted_count": len(deleted),
        "deleted_bytes": total_deleted,
        "deleted_human": fmt_bytes(total_deleted),
        "deleted": deleted,
        "failed": failed,
    }
    write_json(SAFE_DELETE_JSON, payload)
    review_rows = [
        {
            "path": row["path"],
            "bytes": row["bytes"],
            "human": row["human"],
            "reason": f"safe_delete_failed:{row['error']}",
        }
        for row in failed
    ]
    return payload, review_rows


def delete_tmp_outputs(active: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tmp_root = ROOT / "outputs/tmp"
    active_runs = set(active.get("active_run_names", []))
    deleted = []
    review_rows = []
    total_deleted = 0
    if tmp_root.exists():
        candidates = []
        for path in sorted(tmp_root.iterdir()):
            lower = path.name.lower()
            if any(token in lower for token in ("tmp", "smoke", "debug", "resume_smoke")) and mtime_age_days(path) > 3.0:
                candidates.append(path)
        for path in candidates:
            path_text = str(path)
            if any(run_name in path_text for run_name in active_runs):
                review_rows.append(
                    {
                        "path": rel(path),
                        "bytes": path_size(path),
                        "human": fmt_bytes(path_size(path)),
                        "reason": "tmp_path_matches_active_run_name",
                    }
                )
                continue
            ok, size, error = delete_path(path)
            if ok:
                deleted.append({"path": rel(path), "bytes": size, "human": fmt_bytes(size)})
                total_deleted += size
            else:
                review_rows.append(
                    {
                        "path": rel(path),
                        "bytes": path_size(path),
                        "human": fmt_bytes(path_size(path)),
                        "reason": f"tmp_delete_failed:{error}",
                    }
                )
    payload = {
        "generated_at_utc": now_iso(),
        "deleted_count": len(deleted),
        "deleted_bytes": total_deleted,
        "deleted_human": fmt_bytes(total_deleted),
        "deleted": deleted,
    }
    write_json(TMP_DELETE_JSON, payload)
    return payload, review_rows


def corresponding_completed_run_for_log(path: Path) -> str | None:
    base_name = path.name
    for suffix in (".log.gz", ".log", ".out", ".err", ".txt"):
        if base_name.endswith(suffix):
            candidate = base_name[: -len(suffix)]
            if candidate:
                completed, _final_json = completed_status_for_run(candidate)
                if completed:
                    return candidate
    return None


def delete_old_logs(active: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    logs_root = ROOT / "logs"
    active_runs = set(active.get("active_run_names", []))
    active_log_tokens = set(active_runs)
    active_log_tokens.add("stage2_final_evidence_closure_20260414")
    deleted = []
    review_rows = []
    total_deleted = 0
    if logs_root.exists():
        for path in sorted(logs_root.iterdir()):
            if not path.is_file():
                continue
            if path.name in {"README.md", ".gitkeep"}:
                continue
            age_days = mtime_age_days(path)
            if age_days <= 14.0:
                continue
            if any(token in path.name for token in active_log_tokens):
                review_rows.append(
                    {
                        "path": rel(path),
                        "bytes": file_size(path),
                        "human": fmt_bytes(file_size(path)),
                        "reason": "active_run_log_preserved",
                    }
                )
                continue
            matched_run = corresponding_completed_run_for_log(path)
            if matched_run is not None:
                ok, size, error = delete_path(path)
                if ok:
                    deleted.append(
                        {
                            "path": rel(path),
                            "bytes": size,
                            "human": fmt_bytes(size),
                            "matched_completed_run": matched_run,
                        }
                    )
                    total_deleted += size
                else:
                    review_rows.append(
                        {
                            "path": rel(path),
                            "bytes": file_size(path),
                            "human": fmt_bytes(file_size(path)),
                            "reason": f"log_delete_failed:{error}",
                        }
                    )
            elif age_days > 30.0:
                ok, size, error = delete_path(path)
                if ok:
                    deleted.append(
                        {
                            "path": rel(path),
                            "bytes": size,
                            "human": fmt_bytes(size),
                            "matched_completed_run": None,
                        }
                    )
                    total_deleted += size
                else:
                    review_rows.append(
                        {
                            "path": rel(path),
                            "bytes": file_size(path),
                            "human": fmt_bytes(file_size(path)),
                            "reason": f"log_delete_failed:{error}",
                        }
                    )
            else:
                review_rows.append(
                    {
                        "path": rel(path),
                        "bytes": file_size(path),
                        "human": fmt_bytes(file_size(path)),
                        "reason": "older_than_14_days_but_completion_not_proven",
                    }
                )
    payload = {
        "generated_at_utc": now_iso(),
        "deleted_count": len(deleted),
        "deleted_bytes": total_deleted,
        "deleted_human": fmt_bytes(total_deleted),
        "deleted": deleted,
    }
    write_json(LOGS_DELETE_JSON, payload)
    return payload, review_rows


def stage2_dataset_bundle_map() -> dict[str, dict[str, Any]]:
    bundle = read_json(REPORTS / "stage2_dataset_evidence_bundle_20260409.json")
    dataset_rows = bundle.get("datasets", [])
    result: dict[str, dict[str, Any]] = {}
    if isinstance(dataset_rows, list):
        for row in dataset_rows:
            if isinstance(row, dict):
                result[str(row.get("dataset_name", "")).lower()] = row
    return result


def stage2_contract_map() -> dict[str, dict[str, Any]]:
    contract = read_json(REPORTS / "stage2_bootstrap_data_contract_20260408.json")
    dataset_rows = contract.get("datasets", [])
    result: dict[str, dict[str, Any]] = {}
    if isinstance(dataset_rows, list):
        for row in dataset_rows:
            if isinstance(row, dict):
                result[str(row.get("dataset_name", "")).lower()] = row
    return result


def verify_archive_deletable(spec: dict[str, Any], dataset_bundle: dict[str, dict[str, Any]], contract_map: dict[str, dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    archive = Path(spec["archive_path"])
    extracted_root = Path(spec["extracted_root"])
    required_paths = [Path(path) for path in spec["required_paths"]]
    dataset_name = str(spec["dataset_name"])
    key = "burst" if "BURST" in dataset_name else dataset_name.lower()
    bundle_row = dataset_bundle.get(key, {})
    contract_row = contract_map.get(key, {})
    checks = {
        "archive_exists": archive.exists(),
        "extracted_root_exists": extracted_root.exists(),
        "required_paths_exist": {str(path): path.exists() for path in required_paths},
        "bundle_row_exists": bool(bundle_row),
        "contract_row_exists": bool(contract_row),
        "current_pipeline_local_path": str(contract_row.get("local_path", "")),
    }
    if key in {"vspw", "vipseg"}:
        live_probe = bundle_row.get("live_probe", {}) if isinstance(bundle_row.get("live_probe", {}), dict) else {}
        checks["core_binding_zero_missing"] = bool(live_probe.get("core_binding_zero_missing", False))
        checks["semantic_crop_prerequisites_ok"] = bool(live_probe.get("semantic_crop_prerequisites_ok", False))
        current_pipeline_dependency = False
        if str(contract_row.get("local_path", "")):
            current_pipeline_dependency = Path(str(contract_row.get("local_path", ""))).resolve() == archive.resolve()
        ok = (
            checks["archive_exists"]
            and checks["extracted_root_exists"]
            and all(checks["required_paths_exist"].values())
            and bool(bundle_row)
            and str(bundle_row.get("completeness_status", "")) == "core_ready"
            and checks["core_binding_zero_missing"]
            and checks["semantic_crop_prerequisites_ok"]
            and not current_pipeline_dependency
        )
    elif key == "burst":
        live_probe = bundle_row.get("live_probe", {}) if isinstance(bundle_row.get("live_probe", {}), dict) else {}
        current_pipeline_dependency = False
        if str(contract_row.get("local_path", "")):
            current_pipeline_dependency = Path(str(contract_row.get("local_path", ""))).resolve() == archive.resolve()
        ok = (
            checks["archive_exists"]
            and checks["extracted_root_exists"]
            and all(checks["required_paths_exist"].values())
            and bool(bundle_row)
            and str(bundle_row.get("completeness_status", "")) == "optional_extension_ready"
            and bool(live_probe.get("root_exists", False))
            and not current_pipeline_dependency
        )
        checks["burst_live_probe"] = live_probe
    else:
        live_probe = bundle_row.get("live_probe", {}) if isinstance(bundle_row.get("live_probe", {}), dict) else {}
        current_pipeline_dependency = False
        if str(contract_row.get("local_path", "")):
            current_pipeline_dependency = Path(str(contract_row.get("local_path", ""))).resolve() == archive.resolve()
        ok = (
            checks["archive_exists"]
            and checks["extracted_root_exists"]
            and all(checks["required_paths_exist"].values())
            and bool(bundle_row)
            and str(bundle_row.get("completeness_status", "")) == "manual_gate"
            and bool(live_probe.get("root_exists", False))
            and not current_pipeline_dependency
        )
        checks["visor_live_probe"] = live_probe
    checks["current_pipeline_dependency"] = bool(current_pipeline_dependency)
    return ok, checks


def validate_and_delete_raw_archives() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset_bundle = stage2_dataset_bundle_map()
    contract_map = stage2_contract_map()
    deleted_rows = []
    review_rows = []
    total_deleted = 0
    rows = []
    for spec in RAW_ARCHIVE_SPECS:
        archive = Path(spec["archive_path"])
        ok_to_delete, checks = verify_archive_deletable(spec, dataset_bundle, contract_map)
        row = {
            "dataset_name": spec["dataset_name"],
            "archive_path": str(archive),
            "archive_relpath": rel(archive),
            "archive_exists_before": archive.exists(),
            "archive_bytes_before": file_size(archive),
            "archive_human_before": fmt_bytes(file_size(archive)),
            "extracted_root": str(spec["extracted_root"]),
            "extracted_root_verified": bool(checks.get("extracted_root_exists", False) and all(checks.get("required_paths_exist", {}).values())),
            "verification_checks": checks,
            "current_pipeline_dependency": bool(checks.get("current_pipeline_dependency", True)),
            "deleted": False,
            "deleted_bytes": 0,
            "notes": "",
        }
        if not archive.exists():
            row["notes"] = "archive_already_missing"
            rows.append(row)
            continue
        if ok_to_delete:
            ok, size, error = delete_path(archive)
            if ok:
                row["deleted"] = True
                row["deleted_bytes"] = size
                row["deleted_human"] = fmt_bytes(size)
                row["deleted_archive_path"] = str(archive)
                row["current_pipeline_dependency"] = False
                deleted_rows.append(row)
                total_deleted += size
            else:
                row["notes"] = f"delete_failed:{error}"
                review_rows.append(
                    {
                        "path": rel(archive),
                        "bytes": file_size(archive),
                        "human": fmt_bytes(file_size(archive)),
                        "reason": row["notes"],
                    }
                )
            rows.append(row)
        else:
            row["notes"] = "verification_not_strong_enough_for_direct_delete"
            rows.append(row)
            review_rows.append(
                {
                    "path": rel(archive),
                    "bytes": file_size(archive),
                    "human": fmt_bytes(file_size(archive)),
                    "reason": row["notes"],
                }
            )
    payload = {
        "generated_at_utc": now_iso(),
        "deleted_archive_count": len(deleted_rows),
        "deleted_archive_bytes": total_deleted,
        "deleted_archive_human": fmt_bytes(total_deleted),
        "rows": rows,
    }
    write_json(RAW_ARCHIVE_JSON, payload)
    return payload, review_rows


def build_review_required(*groups: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        rows.extend(group)
    total = sum(int(row.get("bytes", 0)) for row in rows)
    payload = {
        "generated_at_utc": now_iso(),
        "review_required_count": len(rows),
        "review_required_size": total,
        "review_required_human": fmt_bytes(total),
        "rows": rows,
    }
    write_json(REVIEW_JSON, payload)
    return payload


def write_summary_doc(before: dict[str, Any], after: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# STWM Storage Direct Cleanup Summary 20260414",
        "",
        f"- before_total_size: {summary['before_total_size']}",
        f"- after_total_size: {summary['after_total_size']}",
        f"- permanently_deleted_size: {summary['permanently_deleted_size']}",
        f"- checkpoint_deleted_size: {summary['checkpoint_deleted_size']}",
        f"- tmp_deleted_size: {summary['tmp_deleted_size']}",
        f"- logs_deleted_size: {summary['logs_deleted_size']}",
        f"- raw_archive_deleted_size: {summary['raw_archive_deleted_size']}",
        f"- review_required_size: {summary['review_required_size']}",
        f"- active_run_detected: {summary['active_run_detected']}",
        f"- protected_path_touched: {summary['protected_path_touched']}",
        "",
        "## Before / After",
        "",
        f"- before total bytes: {before['total_bytes']}",
        f"- after total bytes: {after['total_bytes']}",
    ]
    write_md(SUMMARY_MD, lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    active = active_process_recheck()
    safe_candidates = safe_delete_candidates()
    review_candidates_before = [
        {
            "path": str(spec["archive_path"]),
            "bytes": file_size(Path(spec["archive_path"])),
            "human": fmt_bytes(file_size(Path(spec["archive_path"]))),
            "reason": "requires_extracted_root_and_pipeline_dependency_validation_before_delete",
        }
        for spec in RAW_ARCHIVE_SPECS
        if Path(spec["archive_path"]).exists()
    ]
    before = before_or_after_audit(
        active=active,
        safe_candidates=safe_candidates,
        review_candidates=review_candidates_before,
        target_json=BEFORE_JSON,
        target_md=BEFORE_MD,
    )

    safe_manifest, safe_review = delete_safe_low_risk_cache()
    checkpoint_manifest, checkpoint_review = prune_completed_checkpoints(active)
    tmp_manifest, tmp_review = delete_tmp_outputs(active)
    logs_manifest, logs_review = delete_old_logs(active)
    raw_manifest, raw_review = validate_and_delete_raw_archives()
    review = build_review_required(safe_review, checkpoint_review, tmp_review, logs_review, raw_review)

    after = before_or_after_audit(
        active=active,
        safe_candidates=safe_delete_candidates(),
        review_candidates=review.get("rows", []),
        target_json=AFTER_JSON,
        target_md=DOCS / "STWM_STORAGE_AUDIT_AFTER_DIRECT_CLEANUP_20260414.md",
    )

    summary = {
        "generated_at_utc": now_iso(),
        "before_total_size": before["total_human"],
        "before_total_bytes": before["total_bytes"],
        "after_total_size": after["total_human"],
        "after_total_bytes": after["total_bytes"],
        "permanently_deleted_size": fmt_bytes(
            int(safe_manifest["deleted_bytes"])
            + int(checkpoint_manifest["deleted_checkpoint_bytes"])
            + int(tmp_manifest["deleted_bytes"])
            + int(logs_manifest["deleted_bytes"])
            + int(raw_manifest["deleted_archive_bytes"])
        ),
        "permanently_deleted_bytes": int(safe_manifest["deleted_bytes"])
        + int(checkpoint_manifest["deleted_checkpoint_bytes"])
        + int(tmp_manifest["deleted_bytes"])
        + int(logs_manifest["deleted_bytes"])
        + int(raw_manifest["deleted_archive_bytes"]),
        "checkpoint_deleted_size": checkpoint_manifest["deleted_checkpoint_human"],
        "checkpoint_deleted_bytes": checkpoint_manifest["deleted_checkpoint_bytes"],
        "tmp_deleted_size": tmp_manifest["deleted_human"],
        "tmp_deleted_bytes": tmp_manifest["deleted_bytes"],
        "logs_deleted_size": logs_manifest["deleted_human"],
        "logs_deleted_bytes": logs_manifest["deleted_bytes"],
        "raw_archive_deleted_size": raw_manifest["deleted_archive_human"],
        "raw_archive_deleted_bytes": raw_manifest["deleted_archive_bytes"],
        "review_required_size": review["review_required_human"],
        "review_required_bytes": review["review_required_size"],
        "active_run_detected": bool(active.get("active_run_detected", False)),
        "active_run_names": list(active.get("active_run_names", [])),
        "protected_path_touched": False,
    }
    write_json(SUMMARY_JSON, summary)
    write_summary_doc(before, after, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
