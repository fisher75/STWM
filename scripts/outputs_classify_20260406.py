#!/usr/bin/env python3
import json
import os
from datetime import datetime
from typing import Dict, List, Set


REPO_ROOT = "/home/chen034/workspace/stwm"
AUDIT_JSON = os.path.join(REPO_ROOT, "reports", "outputs_audit_20260406.json")
CLASS_MD = os.path.join(REPO_ROOT, "docs", "OUTPUTS_CLASSIFICATION_20260406.md")
CLASS_JSON = os.path.join(REPO_ROOT, "reports", "outputs_classification_20260406.json")
DELETE_CANDIDATE_TXT = os.path.join(REPO_ROOT, "reports", "outputs_delete_candidates_20260406.txt")
DELETE_CANDIDATE_JSON = os.path.join(REPO_ROOT, "reports", "outputs_delete_candidates_20260406.json")
ARCHIVE_PLAN_JSON = os.path.join(REPO_ROOT, "reports", "outputs_archive_plan_20260406.json")
ARCHIVE_LIST_DIR = os.path.join(REPO_ROOT, "reports", "archive_lists_20260406")


def load_audit() -> Dict:
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def map_stats(depth_dirs: List[Dict]) -> Dict[str, Dict]:
    return {x["relative_path"]: x for x in depth_dirs}


def list_depth2_prefix(depth_dirs: List[Dict], prefix: str) -> List[str]:
    rows = []
    for x in depth_dirs:
        p = x["relative_path"]
        if p.startswith(prefix + "/") and p.count("/") == 2:
            rows.append(p)
    rows.sort()
    return rows


def path_size(stats_map: Dict[str, Dict], rel_path: str) -> int:
    row = stats_map.get(rel_path)
    if row:
        return int(row.get("size_bytes", 0))

    abs_path = os.path.join(REPO_ROOT, rel_path)
    if os.path.isfile(abs_path):
        try:
            return int(os.path.getsize(abs_path))
        except OSError:
            return 0

    total = 0
    for dp, _, files in os.walk(abs_path):
        for fn in files:
            fp = os.path.join(dp, fn)
            try:
                total += int(os.path.getsize(fp))
            except OSError:
                continue
    return total


def human_size(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    v = float(num)
    for u in units:
        if v < 1024 or u == units[-1]:
            if u == "B":
                return f"{int(v)}{u}"
            return f"{v:.2f}{u}"
        v /= 1024.0
    return f"{num}B"


def main() -> None:
    os.makedirs(os.path.join(REPO_ROOT, "docs"), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "reports"), exist_ok=True)
    os.makedirs(ARCHIVE_LIST_DIR, exist_ok=True)

    audit = load_audit()
    depth_dirs = audit["depth_1_2_3_directories"]
    stats_map = map_stats(depth_dirs)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # KEEP: current mainline and high-value direct protocol references; active tiny infra dirs.
    keep: Set[str] = {
        "outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_semteacher_mainline_seed42_v1",
        "outputs/training/stwm_v4_2_real_220m",
        "outputs/training/stwm_v4_2_real_1b",
        "outputs/eval/stwm_v4_2_completed_protocol_eval_20260403",
        "outputs/eval/stwm_v4_2_completed_protocol_eval_real_evalonly_20260403",
        "outputs/monitoring/stwm_hourly_push",
        "outputs/baselines",
        "outputs/queue/stwm_protocol_v2",
        "outputs/queue/stwm_protocol_v2_frontend_default_v1",
    }

    training_depth2 = list_depth2_prefix(depth_dirs, "outputs/training")
    eval_depth2 = list_depth2_prefix(depth_dirs, "outputs/eval")
    visual_depth2 = list_depth2_prefix(depth_dirs, "outputs/visualizations")
    queue_depth2 = list_depth2_prefix(depth_dirs, "outputs/queue")
    audits_depth2 = list_depth2_prefix(depth_dirs, "outputs/audits")
    bench_depth2 = list_depth2_prefix(depth_dirs, "outputs/benchmarks")
    smoke_depth2 = list_depth2_prefix(depth_dirs, "outputs/smoke_tests")

    archive_by_group: Dict[str, List[str]] = {
        "training": [p for p in training_depth2 if p not in keep],
        "eval": [p for p in eval_depth2 if p not in keep],
        "visualizations": visual_depth2,
        "queue": [
            p
            for p in queue_depth2
            if p not in {"outputs/queue/stwm_protocol_v2", "outputs/queue/stwm_protocol_v2_frontend_default_v1"}
        ],
        "audits": audits_depth2,
        "benchmarks": bench_depth2,
        "smoke_tests": smoke_depth2,
        "background_jobs": ["outputs/background_jobs"],
    }

    archive: Set[str] = set()
    for paths in archive_by_group.values():
        archive.update(paths)

    # DELETE_CANDIDATE: only obvious ephemeral files already covered by background_jobs archive package.
    delete_candidates: List[str] = []
    bg_root = os.path.join(REPO_ROOT, "outputs", "background_jobs")
    if os.path.isdir(bg_root):
        for dp, _, files in os.walk(bg_root):
            for fn in files:
                low = fn.lower()
                if low.endswith(".pid") or low.endswith(".lock") or low.endswith(".tmp"):
                    rel = os.path.relpath(os.path.join(dp, fn), REPO_ROOT).replace(os.sep, "/")
                    delete_candidates.append(rel)
    delete_candidates = sorted(set(delete_candidates))

    # Ensure no overlap.
    keep = set(sorted(keep))
    archive = set(sorted(archive))
    archive -= keep
    delete_candidates = [p for p in delete_candidates if p not in keep and p not in archive]

    reason_keep = {
        "outputs/training/stwm_v4_2_220m_protocol_frozen_frontend_semteacher_mainline_seed42_v1": "Current semteacher mainline evidence; keep in-place for active comparison and quick checkpoint access.",
        "outputs/training/stwm_v4_2_real_220m": "Direct protocol/evaluator regression reference used repeatedly in docs; keep in-place.",
        "outputs/training/stwm_v4_2_real_1b": "Direct protocol/evaluator regression reference used repeatedly in docs; keep in-place.",
        "outputs/eval/stwm_v4_2_completed_protocol_eval_20260403": "Completed protocol eval artifact; small and still directly referable.",
        "outputs/eval/stwm_v4_2_completed_protocol_eval_real_evalonly_20260403": "Completed real eval-only artifact; small and still directly referable.",
        "outputs/monitoring/stwm_hourly_push": "Tiny active monitoring state and report trail; keep in-place.",
        "outputs/baselines": "Default evaluator output location in code; keep directory skeleton.",
        "outputs/queue/stwm_protocol_v2": "Current queue namespace referenced by scripts; keep in-place.",
        "outputs/queue/stwm_protocol_v2_frontend_default_v1": "Current queue namespace referenced by scripts/docs; keep in-place.",
    }

    def reason_archive(path: str) -> str:
        if path.startswith("outputs/training/"):
            return "Pre-freeze exploratory/minival/ablation or non-mainline run root; archive before any cleanup."
        if path.startswith("outputs/eval/"):
            return "Detached/smoke eval artifact from older flow; archive for provenance."
        if path.startswith("outputs/visualizations/"):
            return "Legacy visualization assets; retain via archive package."
        if path.startswith("outputs/queue/"):
            return "Old queue traces/backups/parked states; archive for auditability."
        if path.startswith("outputs/audits/"):
            return "Freeze-period audit trace; preserve as archive rather than in-place bulk storage."
        if path.startswith("outputs/benchmarks/"):
            return "Older benchmark/raw cache evidence; preserve in archive."
        if path.startswith("outputs/smoke_tests/"):
            return "Baseline smoke evidence referenced in docs; archive, not direct delete."
        if path == "outputs/background_jobs":
            return "Historical watcher logs/status/pid traces; archive first, then delete only explicit ephemeral files."
        return "Conservative archive by default when uncertain."

    classification_records = []
    for p in sorted(keep):
        classification_records.append(
            {
                "path": p,
                "class": "KEEP",
                "size_bytes": path_size(stats_map, p),
                "reason": reason_keep.get(p, "Keep for direct reference value."),
            }
        )
    for p in sorted(archive):
        classification_records.append(
            {
                "path": p,
                "class": "ARCHIVE",
                "size_bytes": path_size(stats_map, p),
                "reason": reason_archive(p),
            }
        )
    for p in delete_candidates:
        classification_records.append(
            {
                "path": p,
                "class": "DELETE_CANDIDATE",
                "size_bytes": path_size(stats_map, p),
                "reason": "Ephemeral runtime lock/pid/tmp file; explicitly covered by background_jobs archive package.",
            }
        )

    for r in classification_records:
        r["size_human"] = human_size(r["size_bytes"])

    payload = {
        "generated_at": generated_at,
        "source_audit": "reports/outputs_audit_20260406.json",
        "policy": {
            "no_direct_rm_outputs": True,
            "archive_before_delete": True,
            "prefer_archive_when_uncertain": True,
        },
        "keep": sorted(keep),
        "archive": sorted(archive),
        "delete_candidate": delete_candidates,
        "archive_by_group": archive_by_group,
        "records": classification_records,
    }

    with open(CLASS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(DELETE_CANDIDATE_TXT, "w", encoding="utf-8") as f:
        for p in delete_candidates:
            f.write(p + "\n")

    with open(DELETE_CANDIDATE_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": generated_at,
                "delete_candidate_count": len(delete_candidates),
                "paths": delete_candidates,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with open(ARCHIVE_PLAN_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generated_at": generated_at,
                "archive_by_group": archive_by_group,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Write per-group path lists for tar packaging.
    for group, paths in archive_by_group.items():
        list_path = os.path.join(ARCHIVE_LIST_DIR, f"{group}_paths.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for p in sorted(paths):
                f.write(p + "\n")

    # Markdown report
    lines: List[str] = []
    lines.append("# Outputs Classification 20260406")
    lines.append("")
    lines.append(f"- Generated: {generated_at}")
    lines.append("- Policy: KEEP minimal direct-value artifacts, ARCHIVE broad historical evidence, DELETE_CANDIDATE only obvious ephemeral files")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- KEEP: {len(payload['keep'])} paths")
    lines.append(f"- ARCHIVE: {len(payload['archive'])} paths")
    lines.append(f"- DELETE_CANDIDATE: {len(payload['delete_candidate'])} paths")
    lines.append("")

    lines.append("## KEEP")
    lines.append("")
    lines.append("| Path | Size | Reason |")
    lines.append("|---|---:|---|")
    for p in payload["keep"]:
        sz = human_size(path_size(stats_map, p))
        lines.append(f"| {p} | {sz} | {reason_keep.get(p, 'Direct reference value')} |")
    lines.append("")

    lines.append("## ARCHIVE")
    lines.append("")
    lines.append("| Path | Size | Reason |")
    lines.append("|---|---:|---|")
    for p in payload["archive"]:
        sz = human_size(path_size(stats_map, p))
        lines.append(f"| {p} | {sz} | {reason_archive(p)} |")
    lines.append("")

    lines.append("## DELETE_CANDIDATE")
    lines.append("")
    if payload["delete_candidate"]:
        lines.append("| Path | Size | Reason |")
        lines.append("|---|---:|---|")
        for p in payload["delete_candidate"]:
            sz = human_size(path_size(stats_map, p))
            lines.append(
                f"| {p} | {sz} | Ephemeral runtime lock/pid/tmp file; archive coverage exists via outputs/background_jobs package. |"
            )
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Archive Groups")
    lines.append("")
    for group, paths in archive_by_group.items():
        lines.append(f"### {group}")
        lines.append(f"- Path count: {len(paths)}")
        lines.append(f"- List file: reports/archive_lists_20260406/{group}_paths.txt")
        lines.append("")

    with open(CLASS_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")

    print(CLASS_MD)
    print(CLASS_JSON)
    print(ARCHIVE_PLAN_JSON)
    print(DELETE_CANDIDATE_TXT)
    print(DELETE_CANDIDATE_JSON)


if __name__ == "__main__":
    main()
