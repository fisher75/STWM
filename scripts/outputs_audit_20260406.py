#!/usr/bin/env python3
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List


REPO_ROOT = "/home/chen034/workspace/stwm"
OUTPUTS_ROOT = os.path.join(REPO_ROOT, "outputs")
AUDIT_MD = os.path.join(REPO_ROOT, "docs", "OUTPUTS_AUDIT_20260406.md")
AUDIT_JSON = os.path.join(REPO_ROOT, "reports", "outputs_audit_20260406.json")
AUDIT_CSV = os.path.join(REPO_ROOT, "reports", "outputs_audit_20260406.csv")

REFERENCE_PATTERNS = [
    "outputs/training",
    "outputs/eval",
    "outputs/visualizations",
    "outputs/queue",
    "outputs/audits",
    "outputs/benchmarks",
]

TEXT_ROOTS = [
    os.path.join(REPO_ROOT, "docs"),
    os.path.join(REPO_ROOT, "scripts"),
    os.path.join(REPO_ROOT, "code"),
]

TEXT_EXTS = {
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".py",
    ".ts",
    ".js",
    ".tsx",
    ".jsx",
    ".csv",
}

CHECKPOINT_EXTS = {".pt", ".pth", ".ckpt", ".bin", ".safetensors"}
FIGURE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".gif", ".webp"}
LOG_EXTS = {".log", ".out", ".err"}
QUEUE_HINTS = {
    "queue",
    "pending",
    "running",
    "failed",
    "done",
    "lease",
    "pid",
    "lock",
    "state",
    "job",
    "parked",
    "lane",
}


def to_repo_rel(path: str) -> str:
    rel = os.path.relpath(path, REPO_ROOT)
    return rel.replace(os.sep, "/")


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    v = float(n)
    for u in units:
        if v < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(v)}{u}"
            return f"{v:.2f}{u}"
        v /= 1024.0
    return f"{int(n)}B"


def iso_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def path_depth_under_outputs(repo_rel_path: str) -> int:
    if not repo_rel_path.startswith("outputs/"):
        return -1
    sub = repo_rel_path[len("outputs/") :]
    if not sub:
        return 0
    return len(sub.split("/"))


def file_flags(repo_rel_file: str) -> Dict[str, bool]:
    low = repo_rel_file.lower()
    base = os.path.basename(low)
    ext = os.path.splitext(base)[1]
    tokens = set(t for t in low.replace("\\", "/").split("/") if t)

    has_checkpoint = ext in CHECKPOINT_EXTS or "checkpoint" in base or "/checkpoints/" in low
    has_eval_json = ext == ".json" and (
        "eval" in low or "metric" in base or "summary" in base or "report" in base
    )
    has_figure = ext in FIGURE_EXTS
    has_log = ext in LOG_EXTS or "log" in base
    has_queue_state = bool(tokens & QUEUE_HINTS) or any(h in base for h in QUEUE_HINTS)

    return {
        "has_checkpoint": has_checkpoint,
        "has_eval_json": has_eval_json,
        "has_figure": has_figure,
        "has_log": has_log,
        "has_queue_state": has_queue_state,
    }


def is_probably_text(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTS:
        return True
    return ext == ""


def build_outputs_stats() -> Dict[str, Dict]:
    all_dir_stats: Dict[str, Dict] = {}
    top_files: List[Dict] = []

    def ensure_dir(repo_rel: str, abs_path: str) -> Dict:
        if repo_rel not in all_dir_stats:
            try:
                mtime = os.path.getmtime(abs_path)
            except OSError:
                mtime = 0.0
            all_dir_stats[repo_rel] = {
                "relative_path": repo_rel,
                "size_bytes": 0,
                "file_count": 0,
                "latest_mtime": mtime,
                "has_checkpoint": False,
                "has_eval_json": False,
                "has_figure": False,
                "has_log": False,
                "has_queue_state": False,
            }
        return all_dir_stats[repo_rel]

    for dirpath, _, filenames in os.walk(OUTPUTS_ROOT):
        repo_rel_dir = to_repo_rel(dirpath)
        ensure_dir(repo_rel_dir, dirpath)

        for fn in filenames:
            fp = os.path.join(dirpath, fn)
            try:
                st = os.stat(fp)
            except OSError:
                continue

            size = int(st.st_size)
            mtime = float(st.st_mtime)
            repo_rel_file = to_repo_rel(fp)
            flags = file_flags(repo_rel_file)

            top_files.append(
                {
                    "relative_path": repo_rel_file,
                    "size_bytes": size,
                    "size_human": human_size(size),
                    "modified_time": iso_time(mtime),
                    **flags,
                }
            )

            cur = dirpath
            while True:
                repo_rel_cur = to_repo_rel(cur)
                stats = ensure_dir(repo_rel_cur, cur)
                stats["size_bytes"] += size
                stats["file_count"] += 1
                if mtime > stats["latest_mtime"]:
                    stats["latest_mtime"] = mtime
                if flags["has_checkpoint"]:
                    stats["has_checkpoint"] = True
                if flags["has_eval_json"]:
                    stats["has_eval_json"] = True
                if flags["has_figure"]:
                    stats["has_figure"] = True
                if flags["has_log"]:
                    stats["has_log"] = True
                if flags["has_queue_state"]:
                    stats["has_queue_state"] = True

                if os.path.normpath(cur) == os.path.normpath(OUTPUTS_ROOT):
                    break
                parent = os.path.dirname(cur)
                if parent == cur:
                    break
                cur = parent

    for stats in all_dir_stats.values():
        stats["size_human"] = human_size(int(stats["size_bytes"]))
        stats["latest_modified"] = iso_time(float(stats["latest_mtime"]))
        stats.pop("latest_mtime", None)

    top_files_sorted = sorted(top_files, key=lambda x: x["size_bytes"], reverse=True)[:100]
    top_dirs_sorted = sorted(all_dir_stats.values(), key=lambda x: x["size_bytes"], reverse=True)[:100]

    depth123_dirs = []
    for repo_rel, stats in all_dir_stats.items():
        d = path_depth_under_outputs(repo_rel)
        if 1 <= d <= 3:
            depth123_dirs.append(stats)
    depth123_dirs.sort(key=lambda x: x["relative_path"])

    top_level = [s for s in all_dir_stats.values() if path_depth_under_outputs(s["relative_path"]) == 1]
    top_level.sort(key=lambda x: x["size_bytes"], reverse=True)

    return {
        "all_dir_stats": all_dir_stats,
        "top_files": top_files_sorted,
        "top_dirs": top_dirs_sorted,
        "depth123_dirs": depth123_dirs,
        "top_level": top_level,
    }


def collect_reference_hits() -> Dict[str, List[Dict]]:
    hits: Dict[str, List[Dict]] = {k: [] for k in REFERENCE_PATTERNS}

    for root in TEXT_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            for fn in files:
                path = os.path.join(dirpath, fn)
                if not is_probably_text(path):
                    continue
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f, start=1):
                            line_low = line.lower()
                            for pat in REFERENCE_PATTERNS:
                                if pat in line_low:
                                    hits[pat].append(
                                        {
                                            "file": to_repo_rel(path),
                                            "line": i,
                                            "text": line.rstrip("\n")[:500],
                                        }
                                    )
                except OSError:
                    continue
    return hits


def write_csv(path: str, stats: Dict, refs: Dict[str, List[Dict]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "row_type",
                "relative_path",
                "size_bytes",
                "size_human",
                "file_count",
                "latest_modified",
                "has_checkpoint",
                "has_eval_json",
                "has_figure",
                "has_log",
                "has_queue_state",
                "rank",
                "reference_pattern",
                "reference_file",
                "reference_line",
                "reference_text",
            ]
        )

        for row in stats["depth123_dirs"]:
            w.writerow(
                [
                    "dir_depth_1_3",
                    row["relative_path"],
                    row["size_bytes"],
                    row["size_human"],
                    row["file_count"],
                    row["latest_modified"],
                    row["has_checkpoint"],
                    row["has_eval_json"],
                    row["has_figure"],
                    row["has_log"],
                    row["has_queue_state"],
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for idx, row in enumerate(stats["top_dirs"], start=1):
            w.writerow(
                [
                    "top_dir",
                    row["relative_path"],
                    row["size_bytes"],
                    row["size_human"],
                    row["file_count"],
                    row["latest_modified"],
                    row["has_checkpoint"],
                    row["has_eval_json"],
                    row["has_figure"],
                    row["has_log"],
                    row["has_queue_state"],
                    idx,
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for idx, row in enumerate(stats["top_files"], start=1):
            w.writerow(
                [
                    "top_file",
                    row["relative_path"],
                    row["size_bytes"],
                    row["size_human"],
                    "",
                    row["modified_time"],
                    row["has_checkpoint"],
                    row["has_eval_json"],
                    row["has_figure"],
                    row["has_log"],
                    row["has_queue_state"],
                    idx,
                    "",
                    "",
                    "",
                    "",
                ]
            )

        for pat, rows in refs.items():
            for row in rows:
                w.writerow(
                    [
                        "reference_hit",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        pat,
                        row["file"],
                        row["line"],
                        row["text"],
                    ]
                )


def write_markdown(path: str, stats: Dict, refs: Dict[str, List[Dict]], generated_at: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines: List[str] = []
    lines.append("# Outputs Audit 20260406")
    lines.append("")
    lines.append(f"- Generated: {generated_at}")
    lines.append(f"- Root: {OUTPUTS_ROOT}")
    lines.append("- Scope: recursive scan under outputs; full-directory aggregates; depth-1/2/3 directory inventory")
    lines.append(f"- Depth-1/2/3 directories counted: {len(stats['depth123_dirs'])}")
    lines.append(f"- Top directories listed: {len(stats['top_dirs'])}")
    lines.append(f"- Top files listed: {len(stats['top_files'])}")
    lines.append("")

    lines.append("## Top-Level Directory Size Ranking")
    lines.append("")
    lines.append("| Rank | Path | Size | Files | Last Modified |")
    lines.append("|---:|---|---:|---:|---|")
    for i, row in enumerate(stats["top_level"], start=1):
        lines.append(
            f"| {i} | {row['relative_path']} | {row['size_human']} | {row['file_count']} | {row['latest_modified']} |"
        )
    lines.append("")

    lines.append("## Depth 1-3 Directory Inventory")
    lines.append("")
    lines.append(
        "| Path | Size | Files | Last Modified | checkpoint | eval json | figure | log | queue state |"
    )
    lines.append("|---|---:|---:|---|---|---|---|---|---|")
    for row in stats["depth123_dirs"]:
        lines.append(
            "| "
            + f"{row['relative_path']} | {row['size_human']} | {row['file_count']} | {row['latest_modified']} | "
            + f"{str(row['has_checkpoint'])} | {str(row['has_eval_json'])} | {str(row['has_figure'])} | {str(row['has_log'])} | {str(row['has_queue_state'])} |"
        )
    lines.append("")

    lines.append("## Top 100 Largest Directories")
    lines.append("")
    lines.append("| Rank | Path | Size | Files | Last Modified |")
    lines.append("|---:|---|---:|---:|---|")
    for i, row in enumerate(stats["top_dirs"], start=1):
        lines.append(
            f"| {i} | {row['relative_path']} | {row['size_human']} | {row['file_count']} | {row['latest_modified']} |"
        )
    lines.append("")

    lines.append("## Top 100 Largest Files")
    lines.append("")
    lines.append("| Rank | Path | Size | Modified Time | checkpoint | eval json | figure | log | queue state |")
    lines.append("|---:|---|---:|---|---|---|---|---|---|")
    for i, row in enumerate(stats["top_files"], start=1):
        lines.append(
            "| "
            + f"{i} | {row['relative_path']} | {row['size_human']} | {row['modified_time']} | "
            + f"{str(row['has_checkpoint'])} | {str(row['has_eval_json'])} | {str(row['has_figure'])} | {str(row['has_log'])} | {str(row['has_queue_state'])} |"
        )
    lines.append("")

    lines.append("## Direct Reference Hits In docs/scripts/code")
    lines.append("")
    for pat in REFERENCE_PATTERNS:
        rows = refs.get(pat, [])
        lines.append(f"### {pat}")
        lines.append(f"- Hits: {len(rows)}")
        if rows:
            lines.append("")
            lines.append("| File | Line | Text |")
            lines.append("|---|---:|---|")
            for row in rows:
                safe_text = row["text"].replace("|", "\\|")
                lines.append(f"| {row['file']} | {row['line']} | {safe_text} |")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> None:
    os.makedirs(os.path.join(REPO_ROOT, "docs"), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "reports"), exist_ok=True)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stats = build_outputs_stats()
    refs = collect_reference_hits()

    payload = {
        "generated_at": generated_at,
        "repo_root": REPO_ROOT,
        "outputs_root": OUTPUTS_ROOT,
        "scope": {
            "directory_depths": [1, 2, 3],
            "reference_roots": [to_repo_rel(p) for p in TEXT_ROOTS if os.path.exists(p)],
            "reference_patterns": REFERENCE_PATTERNS,
        },
        "summary": {
            "depth_1_2_3_dir_count": len(stats["depth123_dirs"]),
            "top_dir_count": len(stats["top_dirs"]),
            "top_file_count": len(stats["top_files"]),
            "reference_hit_count": sum(len(v) for v in refs.values()),
        },
        "top_level_sizes": stats["top_level"],
        "depth_1_2_3_directories": stats["depth123_dirs"],
        "top_100_directories": stats["top_dirs"],
        "top_100_files": stats["top_files"],
        "reference_hits": refs,
    }

    with open(AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    write_csv(AUDIT_CSV, stats, refs)
    write_markdown(AUDIT_MD, stats, refs, generated_at)

    print(AUDIT_MD)
    print(AUDIT_JSON)
    print(AUDIT_CSV)


if __name__ == "__main__":
    main()
