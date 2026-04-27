from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("STWM_EXTERNAL_OUTPUT_PHASE", "full_eval")
os.environ.setdefault("STWM_EXTERNAL_SMOKE_MAX_SIDE", "384")

try:
    import setproctitle

    setproctitle.setproctitle("python")
    SETPROCTITLE_STATUS = {"requested_title": "python", "setproctitle_ok": True, "exact_error": None}
except Exception as exc:  # pragma: no cover - depends on local package availability
    SETPROCTITLE_STATUS = {"requested_title": "python", "setproctitle_ok": False, "exact_error": f"{type(exc).__name__}: {exc}"}

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stwm.tools.external_baselines.common_io import (  # noqa: E402
    DOCS,
    OUTPUTS,
    REPORTS,
    REPOS,
    ROOT,
    load_json,
    sha256_json,
    write_json,
    write_markdown,
)
from stwm.tools.external_baselines.run_external_baseline_smoke_20260426 import (  # noqa: E402
    CoTrackerSmokeRunner,
    CutieSmokeRunner,
    SAM2SmokeRunner,
    PreparedItem,
    rel,
    safe_name,
    subset_flags,
)


MANIFEST = REPORTS / "stwm_external_baseline_item_manifest_20260426.json"
SMOKE_SUMMARY = REPORTS / "stwm_external_baseline_smoke_20260426.json"
SMOKE_DECISION = REPORTS / "stwm_external_baseline_smoke_decision_20260426.json"
CUTIE_SMOKE = REPORTS / "stwm_external_baseline_cutie_smoke_20260426.json"
SAM2_SMOKE = REPORTS / "stwm_external_baseline_sam2_smoke_20260426.json"
COTRACKER_SMOKE = REPORTS / "stwm_external_baseline_cotracker_smoke_20260426.json"

STWM_SOURCES = {
    "belief_final_eval": REPORTS / "stwm_belief_final_eval_20260424.json",
    "belief_strict_bootstrap": REPORTS / "stwm_belief_strict_bootstrap_20260424.json",
    "reacquisition_v2_eval": REPORTS / "stwm_reacquisition_v2_eval_20260425.json",
    "reacquisition_v2_bootstrap": REPORTS / "stwm_reacquisition_v2_bootstrap_20260425.json",
    "false_confuser_analysis": REPORTS / "stwm_false_confuser_analysis_20260425.json",
}

BASELINE_REPORT_PATHS = {
    "cutie": REPORTS / "stwm_external_baseline_cutie_full_eval_20260426.json",
    "sam2": REPORTS / "stwm_external_baseline_sam2_full_eval_20260426.json",
    "cotracker": REPORTS / "stwm_external_baseline_cotracker_full_eval_20260426.json",
}

DISPLAY_NAMES = {
    "cutie": "Cutie",
    "sam2": "SAM2",
    "cotracker": "CoTracker",
    "stwm_trace_belief_assoc": "STWM trace_belief_assoc",
    "frozen_external_teacher_only": "frozen_external_teacher_only",
    "legacysem": "legacysem",
    "calibration-only": "calibration-only",
    "cropenc": "cropenc",
}

STWM_METHOD_MAP = {
    "TUSB-v3.1::official(best_semantic_hard.pt+trace_belief_assoc)": "stwm_trace_belief_assoc",
    "calibration-only::best.pt": "calibration-only",
    "cropenc::best.pt": "cropenc",
    "legacysem::best.pt": "legacysem",
}

BOOTSTRAP_SEED = 20260426
BOOTSTRAP_N = int(os.environ.get("STWM_EXTERNAL_BOOTSTRAP_N", "2000"))


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_md_json(path: Path, title: str, payload: dict[str, Any]) -> None:
    write_markdown(path, title, ["```json", json.dumps(payload, indent=2, sort_keys=True), "```"])


def read_report(path: Path) -> tuple[bool, bool, dict[str, Any] | None, str | None]:
    if not path.exists():
        return False, False, None, f"missing:{path}"
    try:
        return True, True, load_json(path), None
    except Exception as exc:
        return True, False, None, f"invalid_json:{type(exc).__name__}:{exc}"


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo), text=True, timeout=10)
        return out.strip()
    except Exception:
        return None


def checkpoint_from_smoke(report: dict[str, Any] | None, baseline: str) -> str | None:
    if not report:
        return None
    for row in report.get("per_item_results", []):
        if row.get("checkpoint_used"):
            return row.get("checkpoint_used")
    if baseline == "cutie":
        return "baselines/repos/Cutie/weights/cutie-base-mega.pth"
    return None


def source_audit() -> dict[str, Any]:
    paths = {
        "manifest": MANIFEST,
        "smoke_summary": SMOKE_SUMMARY,
        "smoke_decision": SMOKE_DECISION,
        "cutie_smoke": CUTIE_SMOKE,
        "sam2_smoke": SAM2_SMOKE,
        "cotracker_smoke": COTRACKER_SMOKE,
        **STWM_SOURCES,
    }
    loaded: dict[str, dict[str, Any] | None] = {}
    audit_entries: dict[str, Any] = {}
    for name, path in paths.items():
        exists, valid, data, error = read_report(path)
        loaded[name] = data
        audit_entries[name] = {"path": rel(path), "exists": exists, "valid_json": valid, "exact_error": error}
    manifest = loaded.get("manifest") or {}
    smoke = loaded.get("smoke_summary") or {}
    decision = loaded.get("smoke_decision") or {}
    cutie_smoke = loaded.get("cutie_smoke") or {}
    sam2_smoke = loaded.get("sam2_smoke") or {}
    cotracker_smoke = loaded.get("cotracker_smoke") or {}
    stwm_available = {k: bool(audit_entries[k]["exists"] and audit_entries[k]["valid_json"]) for k in STWM_SOURCES}
    audit = {
        "created_at": now(),
        "setproctitle_status": SETPROCTITLE_STATUS,
        "reports": audit_entries,
        "manifest_exists": audit_entries["manifest"]["exists"],
        "manifest_valid_json": audit_entries["manifest"]["valid_json"],
        "total_manifest_items": len(manifest.get("items") or []),
        "cutie_smoke_pass": bool(smoke.get("cutie_smoke_pass") and decision.get("cutie_enter_full_eval")),
        "sam2_smoke_pass": bool(smoke.get("sam2_smoke_pass") and decision.get("sam2_enter_full_eval")),
        "cotracker_smoke_pass": bool(smoke.get("cotracker_smoke_pass") and decision.get("cotracker_enter_full_eval")),
        "cutie_checkpoint_used": checkpoint_from_smoke(cutie_smoke, "cutie"),
        "sam2_checkpoint_used": checkpoint_from_smoke(sam2_smoke, "sam2"),
        "cotracker_checkpoint_used": checkpoint_from_smoke(cotracker_smoke, "cotracker"),
        "stwm_reference_reports_available": stwm_available,
        "audit_passed": bool(
            audit_entries["manifest"]["valid_json"]
            and audit_entries["smoke_decision"]["valid_json"]
            and (decision.get("cutie_enter_full_eval") or decision.get("sam2_enter_full_eval") or decision.get("cotracker_enter_full_eval"))
        ),
    }
    write_json(REPORTS / "stwm_external_baseline_full_eval_source_audit_20260426.json", audit)
    write_md_json(DOCS / "STWM_EXTERNAL_BASELINE_FULL_EVAL_SOURCE_AUDIT_20260426.md", "STWM External Baseline Full Eval Source Audit 20260426", audit)
    return audit


def subset_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for item in items:
        for key, value in subset_flags(item).items():
            if value:
                counts[key] += 1
    return dict(counts)


def item_plan(items: list[dict[str, Any]], smoke_summary: dict[str, Any]) -> dict[str, Any]:
    avg_runtime = smoke_summary.get("per_baseline_average_runtime") or {}
    total = len(items)
    estimate = {
        name: {
            "items": total,
            "seconds_per_item_from_smoke": avg_runtime.get(name),
            "estimated_seconds": None if avg_runtime.get(name) is None else round(float(avg_runtime[name]) * total, 1),
        }
        for name in ["cutie", "sam2", "cotracker"]
    }
    report = {
        "created_at": now(),
        "policy": {
            "default_use_all_materialized_items": True,
            "full_eval_not_smoke": True,
            "deterministic_shard_if_any": False,
        },
        "total_items": total,
        "cutie_items_to_run": total,
        "sam2_items_to_run": total,
        "cotracker_items_to_run": total,
        "shard_plan_if_any": None,
        "subset_counts": subset_counts(items),
        "expected_runtime_estimate": estimate,
        "item_ids_hash": sha256_json([x.get("item_id") for x in items]),
    }
    write_json(REPORTS / "stwm_external_baseline_full_eval_item_plan_20260426.json", report)
    lines = [
        f"- total_items: `{total}`",
        f"- subset_counts: `{report['subset_counts']}`",
        f"- shard_plan_if_any: `{report['shard_plan_if_any']}`",
        "",
        "| baseline | items | smoke sec/item | estimated seconds |",
        "|---|---:|---:|---:|",
    ]
    for name, est in estimate.items():
        lines.append(f"| {name} | {est['items']} | {est['seconds_per_item_from_smoke']} | {est['estimated_seconds']} |")
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_FULL_EVAL_ITEM_PLAN_20260426.md", "STWM External Baseline Full Eval Item Plan 20260426", lines)
    return report


def top5_hit(row: dict[str, Any]) -> bool:
    return str(row.get("gt_candidate_id")) in [str(x) for x in (row.get("top5_candidates") or [])]


def normalize_result(row: dict[str, Any]) -> dict[str, Any]:
    success = bool(row.get("success"))
    correct = bool(row.get("top1_correct")) if success else False
    row["false_confuser"] = None if not success else (not correct)
    row["false_reacquisition"] = None if not success else (not correct)
    row["top5_hit"] = None if not success else top5_hit(row)
    return row


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("success")]
    if not valid:
        return {
            "count": 0,
            "top1": None,
            "top5": None,
            "MRR": None,
            "false_confuser_rate": None,
            "false_reacquisition_rate": None,
            "runtime_per_item": None,
            "successful_items": 0,
            "failed_items": len(rows),
        }
    return {
        "count": len(valid),
        "top1": mean([1.0 if r.get("top1_correct") else 0.0 for r in valid]),
        "top5": mean([1.0 if r.get("top5_hit") else 0.0 for r in valid]),
        "MRR": mean([float(r.get("mrr") or 0.0) for r in valid]),
        "false_confuser_rate": mean([1.0 if r.get("false_confuser") else 0.0 for r in valid]),
        "false_reacquisition_rate": mean([1.0 if r.get("false_reacquisition") else 0.0 for r in valid]),
        "runtime_per_item": mean([float(r.get("runtime_seconds") or 0.0) for r in valid]),
        "successful_items": len(valid),
        "failed_items": len(rows) - len(valid),
    }


def subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for subset in ["occlusion_reappearance", "long_gap_persistence", "crossing_ambiguity", "appearance_change", "OOD_hard"]:
        selected = [r for r in rows if subset_flags(r).get(subset)]
        out[subset] = aggregate(selected)
        out[f"{subset}_top1"] = out[subset].get("top1")
    return out


def checkpoint_used_from_report(report: dict[str, Any], baseline: str) -> str | None:
    for row in report.get("per_item_results", []):
        if row.get("checkpoint_used"):
            return row.get("checkpoint_used")
    if baseline == "cutie":
        return "baselines/repos/Cutie/weights/cutie-base-mega.pth"
    return None


def write_baseline_full_report(name: str, report: dict[str, Any]) -> None:
    path = BASELINE_REPORT_PATHS[name]
    write_json(path, report)
    title = {"cutie": "Cutie", "sam2": "SAM2", "cotracker": "CoTracker"}[name]
    s = report["summary"]
    lines = [
        f"- completed: `{report['completed']}`",
        f"- checkpoint_used: `{report.get('checkpoint_used')}`",
        f"- successful_items: `{s.get('successful_items')}`",
        f"- failed_items: `{s.get('failed_items')}`",
        f"- top1: `{s.get('top1')}`",
        f"- MRR: `{s.get('MRR')}`",
        f"- false_confuser_rate: `{s.get('false_confuser_rate')}`",
        "",
        "| item_id | success | pred | gt | top1 | mrr | failure |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in report.get("per_item_results", [])[:40]:
        lines.append(
            f"| `{row.get('item_id')}` | `{row.get('success')}` | `{row.get('predicted_candidate_id')}` | `{row.get('gt_candidate_id')}` | `{row.get('top1_correct')}` | {float(row.get('mrr') or 0):.3f} | {str(row.get('failure_reason_if_any') or '')[:120]} |"
        )
    write_markdown(DOCS / f"STWM_EXTERNAL_BASELINE_{title.upper()}_FULL_EVAL_20260426.md", f"STWM External Baseline {title} Full Eval 20260426", lines)


def run_external_baseline(name: str, runner: Any, items: list[dict[str, Any]], smoke_pass: bool) -> dict[str, Any]:
    if not smoke_pass:
        report = {
            "created_at": now(),
            "baseline_name": name,
            "completed": False,
            "checkpoint_used": None,
            "repo_commit": git_commit(REPOS / {"cutie": "Cutie", "sam2": "sam2", "cotracker": "co-tracker"}[name]),
            "per_item_results": [],
            "summary": {"successful_items": 0, "failed_items": 0},
            "skipped_reason_counts": {"smoke_pass_false": len(items)},
            "exact_blocking_reason": "smoke_pass_false",
        }
        write_baseline_full_report(name, report)
        return report
    runner.load()
    if not runner.model_loaded:
        report = {
            "created_at": now(),
            "baseline_name": name,
            "completed": False,
            "checkpoint_used": None,
            "repo_commit": git_commit(REPOS / {"cutie": "Cutie", "sam2": "sam2", "cotracker": "co-tracker"}[name]),
            "per_item_results": [],
            "summary": {"successful_items": 0, "failed_items": 0},
            "skipped_reason_counts": {"model_load_failed": len(items)},
            "exact_blocking_reason": (runner.model_load_error or {}).get("exact_error") or "model_load_failed",
            "model_load_error": runner.model_load_error,
        }
        write_baseline_full_report(name, report)
        return report

    results: list[dict[str, Any]] = []
    started = time.time()
    partial_path = BASELINE_REPORT_PATHS[name]
    for idx, item in enumerate(items, start=1):
        row = normalize_result(runner.run_item(item))
        results.append(row)
        if idx % 25 == 0:
            partial = {
                "created_at": now(),
                "baseline_name": name,
                "completed": False,
                "partial": True,
                "processed_items": idx,
                "total_items": len(items),
                "per_item_results": results,
                "summary": aggregate(results),
                "per_item_results_hash": sha256_json(results),
            }
            write_json(partial_path, partial)
    summary = aggregate(results)
    report = {
        "created_at": now(),
        "baseline_name": name,
        "completed": summary["successful_items"] == len(items),
        "partial": False,
        "repo_commit": git_commit(REPOS / {"cutie": "Cutie", "sam2": "sam2", "cotracker": "co-tracker"}[name]),
        "checkpoint_used": checkpoint_used_from_report({"per_item_results": results}, name),
        "point_sampling_strategy": "mask pixels subsampled to <=64 points" if name == "cotracker" else None,
        "predictor_api_used": "sam2.sam2_video_predictor.SAM2VideoPredictor" if name == "sam2" else None,
        "extension_warning_if_any": "SAM2 optional C++ post-processing extension unavailable warning may appear; inference still ran." if name == "sam2" else None,
        "wall_time_seconds": round(time.time() - started, 3),
        "total_items": len(items),
        "per_item_results": results,
        "summary": summary,
        "per_subset_results": subset_metrics(results),
        "skipped_reason_counts": Counter([r.get("failure_reason_if_any") or "success" for r in results if not r.get("success")]),
        "per_item_results_hash": sha256_json(results),
        "exact_blocking_reason": None if summary["successful_items"] else "all_items_failed",
    }
    report["skipped_reason_counts"] = dict(report["skipped_reason_counts"])
    write_baseline_full_report(name, report)
    try:
        del runner.model
        runner.model = None
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return report


def normalize_tags(tags: Any, item_id: str, manifest_tags: dict[str, dict[str, bool]]) -> dict[str, bool]:
    base = manifest_tags.get(item_id, {}).copy()
    if isinstance(tags, list):
        for t in tags:
            if t == "ambiguity":
                base["crossing_ambiguity"] = True
            else:
                base[t] = True
    elif isinstance(tags, dict):
        for k, v in tags.items():
            base[k] = bool(v)
    return base


def load_stwm_item_scores(manifest_items: list[dict[str, Any]]) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    manifest_ids = {x.get("item_id") for x in manifest_items}
    manifest_tags = {x.get("item_id"): subset_flags(x) for x in manifest_items}
    belief = load_json(STWM_SOURCES["belief_final_eval"])
    bucket: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for panel_name, panel in (belief.get("panels") or {}).items():
        for row in panel.get("per_item_results") or []:
            item_id = row.get("protocol_item_id")
            method = STWM_METHOD_MAP.get(row.get("method_name"))
            if item_id in manifest_ids and method:
                bucket[method][item_id].append(row)

    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for method, by_item in bucket.items():
        for item_id, rows in by_item.items():
            top1s = [float(r.get("query_future_top1_acc") or 0.0) for r in rows]
            top5s = [float(r.get("top5_hit") or 0.0) for r in rows]
            mrrs = [float(r.get("mrr") or 0.0) for r in rows]
            top1_candidates = Counter([str(r.get("top1_candidate_id")) for r in rows if r.get("top1_candidate_id") is not None])
            gt = str(item_id).split("::")[-1]
            out[method][item_id] = {
                "item_id": item_id,
                "success": True,
                "top1_correct": mean(top1s),
                "top5_hit": mean(top5s),
                "mrr": mean(mrrs),
                "false_confuser": 1.0 - mean(top1s),
                "false_reacquisition": 1.0 - mean(top1s),
                "predicted_candidate_id": top1_candidates.most_common(1)[0][0] if top1_candidates else None,
                "gt_candidate_id": gt,
                "subset_tags": normalize_tags(rows[0].get("subset_tags"), item_id, manifest_tags),
                "row_count": len(rows),
                "source": "stwm_belief_final_eval_20260424",
            }
    meta = {
        "source": rel(STWM_SOURCES["belief_final_eval"]),
        "manifest_items": len(manifest_items),
        "matched_items_by_method": {k: len(v) for k, v in out.items()},
        "aggregation": "mean over official rows/seeds/panels per protocol_item_id",
    }
    return out, meta


def stwm_aggregate(rows_by_item: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows_by_item.values())
    if not rows:
        return {"count": 0, "top1": None, "top5": None, "MRR": None, "false_confuser_rate": None, "false_reacquisition_rate": None}
    return {
        "count": len(rows),
        "top1": mean([float(r.get("top1_correct") or 0.0) for r in rows]),
        "top5": mean([float(r.get("top5_hit") or 0.0) for r in rows]),
        "MRR": mean([float(r.get("mrr") or 0.0) for r in rows]),
        "false_confuser_rate": mean([float(r.get("false_confuser") or 0.0) for r in rows]),
        "false_reacquisition_rate": mean([float(r.get("false_reacquisition") or 0.0) for r in rows]),
        "runtime_per_item": None,
        "source": "official matched per-item mean",
    }


def stwm_subset_aggregate(rows_by_item: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out = {}
    rows = list(rows_by_item.values())
    for subset in ["occlusion_reappearance", "long_gap_persistence", "crossing_ambiguity", "appearance_change", "OOD_hard"]:
        selected = [r for r in rows if subset_flags(r).get(subset)]
        out[subset] = {
            "count": len(selected),
            "top1": mean([float(r.get("top1_correct") or 0.0) for r in selected]) if selected else None,
            "MRR": mean([float(r.get("mrr") or 0.0) for r in selected]) if selected else None,
            "false_confuser_rate": mean([float(r.get("false_confuser") or 0.0) for r in selected]) if selected else None,
            "false_reacquisition_rate": mean([float(r.get("false_reacquisition") or 0.0) for r in selected]) if selected else None,
        }
        out[f"{subset}_top1"] = out[subset]["top1"]
    return out


def aggregate_variant_from_reacq(method: str) -> dict[str, Any] | None:
    reacq = load_json(STWM_SOURCES["reacquisition_v2_eval"])
    variant = (reacq.get("variants") or {}).get(method)
    if not variant:
        return None
    return {
        "count": variant.get("count"),
        "top1": variant.get("top1"),
        "top5": variant.get("top5"),
        "MRR": variant.get("MRR"),
        "false_confuser_rate": variant.get("false_confuser_rate"),
        "false_reacquisition_rate": variant.get("false_reacquisition_rate"),
        "runtime_per_item": None,
        "source": "stwm_reacquisition_v2_eval_20260425 aggregate; not item-aligned to external manifest",
    }


def external_rows_by_item(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {r.get("item_id"): r for r in report.get("per_item_results", []) if r.get("success")}


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric in ["top1", "reacquisition_top1"]:
        return float(row.get("top1_correct") or 0.0)
    if metric == "MRR":
        return float(row.get("mrr") or 0.0)
    if metric == "false_confuser_rate":
        return -float(row.get("false_confuser") if row.get("false_confuser") is not None else (not bool(row.get("top1_correct"))))
    if metric == "false_reacquisition_rate":
        return -float(row.get("false_reacquisition") if row.get("false_reacquisition") is not None else (not bool(row.get("top1_correct"))))
    if metric.endswith("_top1"):
        return float(row.get("top1_correct") or 0.0)
    return float(row.get(metric) or 0.0)


def bootstrap_compare(stwm_rows: dict[str, dict[str, Any]], ext_rows: dict[str, dict[str, Any]], metric: str, subset: str | None = None) -> dict[str, Any]:
    ids = sorted(set(stwm_rows) & set(ext_rows))
    if subset:
        ids = [i for i in ids if subset_flags(ext_rows[i]).get(subset) or subset_flags(stwm_rows[i]).get(subset)]
    deltas = [metric_value(stwm_rows[i], metric) - metric_value(ext_rows[i], metric) for i in ids]
    if not deltas:
        return {"count": 0, "mean_delta": None, "ci95_low": None, "ci95_high": None, "zero_excluded": False, "bootstrap_win_rate": None}
    rng = random.Random(BOOTSTRAP_SEED + abs(hash((metric, subset))) % 100000)
    n = len(deltas)
    boot = []
    for _ in range(BOOTSTRAP_N):
        boot.append(mean([deltas[rng.randrange(n)] for _ in range(n)]))
    boot.sort()
    lo = boot[int(0.025 * (BOOTSTRAP_N - 1))]
    hi = boot[int(0.975 * (BOOTSTRAP_N - 1))]
    md = mean(deltas)
    return {
        "count": n,
        "mean_delta": md,
        "ci95_low": lo,
        "ci95_high": hi,
        "zero_excluded": bool(lo > 0 or hi < 0),
        "bootstrap_win_rate": sum(1 for x in deltas if x > 0) / n,
    }


def write_summary_and_bootstrap(reports: dict[str, dict[str, Any]], manifest_items: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    stwm_rows, stwm_meta = load_stwm_item_scores(manifest_items)
    per_method_overall = {}
    per_method_per_subset = {}
    for name, report in reports.items():
        per_method_overall[name] = report["summary"]
        per_method_per_subset[name] = report.get("per_subset_results") or {}
    for method in ["stwm_trace_belief_assoc", "calibration-only", "cropenc", "legacysem"]:
        per_method_overall[method] = stwm_aggregate(stwm_rows.get(method, {}))
        per_method_per_subset[method] = stwm_subset_aggregate(stwm_rows.get(method, {}))
    frozen = aggregate_variant_from_reacq("frozen_external_teacher_only")
    if frozen:
        per_method_overall["frozen_external_teacher_only"] = frozen
        reacq = load_json(STWM_SOURCES["reacquisition_v2_eval"])
        per_method_per_subset["frozen_external_teacher_only"] = (reacq.get("variants") or {}).get("frozen_external_teacher_only", {}).get("group_breakdown", {})

    ext_names = [n for n, r in reports.items() if r.get("completed") and r["summary"].get("top1") is not None]
    strongest = max(ext_names, key=lambda n: per_method_overall[n]["top1"]) if ext_names else None
    best_false = min(ext_names, key=lambda n: per_method_overall[n]["false_confuser_rate"]) if ext_names else None
    best_long = max(ext_names, key=lambda n: (per_method_per_subset[n].get("long_gap_persistence") or {}).get("top1") or -1) if ext_names else None
    best_ood = max(ext_names, key=lambda n: (per_method_per_subset[n].get("OOD_hard") or {}).get("top1") or -1) if ext_names else None
    summary = {
        "created_at": now(),
        "per_method_overall": per_method_overall,
        "per_method_per_subset": per_method_per_subset,
        "strongest_external_baseline": strongest,
        "best_external_by_overall": strongest,
        "best_external_by_false_confuser": best_false,
        "best_external_by_long_gap": best_long,
        "best_external_by_ood": best_ood,
        "stwm_reference_alignment": stwm_meta,
        "external_reports": {k: rel(v) for k, v in BASELINE_REPORT_PATHS.items()},
    }
    write_json(REPORTS / "stwm_external_baseline_full_eval_summary_20260426.json", summary)
    lines = [
        "| method | top1 | top5 | MRR | false confuser | long-gap top1 | OOD top1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, vals in per_method_overall.items():
        long_top1 = (per_method_per_subset.get(method, {}).get("long_gap_persistence") or {}).get("top1")
        ood_top1 = (per_method_per_subset.get(method, {}).get("OOD_hard") or {}).get("top1")
        lines.append(f"| {DISPLAY_NAMES.get(method, method)} | {fmt(vals.get('top1'))} | {fmt(vals.get('top5'))} | {fmt(vals.get('MRR'))} | {fmt(vals.get('false_confuser_rate'))} | {fmt(long_top1)} | {fmt(ood_top1)} |")
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_FULL_EVAL_SUMMARY_20260426.md", "STWM External Baseline Full Eval Summary 20260426", lines)

    bootstrap = {"created_at": now(), "comparisons": {}, "bootstrap_n": BOOTSTRAP_N, "positive_delta_policy": "positive means STWM better; false-rate metrics are negated before delta"}
    stwm = stwm_rows.get("stwm_trace_belief_assoc", {})
    metrics = [
        ("top1", None),
        ("MRR", None),
        ("false_confuser_rate", None),
        ("false_reacquisition_rate", None),
        ("reacquisition_top1", None),
        ("long_gap_persistence_top1", "long_gap_persistence"),
        ("occlusion_reappearance_top1", "occlusion_reappearance"),
        ("OOD_hard_top1", "OOD_hard"),
    ]
    for ext in ext_names:
        ext_rows = external_rows_by_item(reports[ext])
        comp = {}
        for metric, subset in metrics:
            comp[metric] = bootstrap_compare(stwm, ext_rows, metric, subset)
        bootstrap["comparisons"][f"STWM_vs_{ext}"] = comp
    if strongest:
        bootstrap["comparisons"]["STWM_vs_strongest_external_baseline"] = bootstrap["comparisons"].get(f"STWM_vs_{strongest}")
        bootstrap["strongest_external_baseline"] = strongest
    write_json(REPORTS / "stwm_external_baseline_full_eval_bootstrap_20260426.json", bootstrap)
    write_md_json(DOCS / "STWM_EXTERNAL_BASELINE_FULL_EVAL_BOOTSTRAP_20260426.md", "STWM External Baseline Full Eval Bootstrap 20260426", bootstrap)
    return summary, bootstrap


def fmt(x: Any) -> str:
    if x is None:
        return "NA"
    if isinstance(x, (int, float)):
        return f"{x:.3f}"
    return str(x)


def make_visual_examples(reports: dict[str, dict[str, Any]], manifest_items: list[dict[str, Any]]) -> dict[str, Any]:
    out_dir = OUTPUTS / "external_baseline_comparison_visuals"
    out_dir.mkdir(parents=True, exist_ok=True)
    stwm_rows, _ = load_stwm_item_scores(manifest_items)
    stwm = stwm_rows.get("stwm_trace_belief_assoc", {})
    by_ext = {name: external_rows_by_item(report) for name, report in reports.items()}
    ids = sorted(set.intersection(*(set(v) for v in by_ext.values())) if by_ext else set())
    selected: list[tuple[str, str]] = []

    def stwm_correct(item_id: str) -> bool | None:
        row = stwm.get(item_id)
        if not row:
            return None
        return float(row.get("top1_correct") or 0.0) >= 0.5

    categories = {
        "STWM_correct_external_wrong": lambda i: stwm_correct(i) is True and any(not by_ext[e][i].get("top1_correct") for e in by_ext),
        "external_correct_STWM_wrong": lambda i: stwm_correct(i) is False and any(by_ext[e][i].get("top1_correct") for e in by_ext),
        "all_methods_correct": lambda i: stwm_correct(i) is True and all(by_ext[e][i].get("top1_correct") for e in by_ext),
        "all_methods_confused": lambda i: stwm_correct(i) is False and all(not by_ext[e][i].get("top1_correct") for e in by_ext),
        "false_confuser_external": lambda i: any(by_ext[e][i].get("false_confuser") for e in by_ext),
    }
    for cat, pred in categories.items():
        for i in ids:
            if len(selected) >= 20:
                break
            if pred(i) and i not in [x[1] for x in selected]:
                selected.append((cat, i))
        if len(selected) >= 20:
            break
    for i in ids:
        if len(selected) >= 20:
            break
        if i not in [x[1] for x in selected]:
            selected.append(("filler_diverse", i))

    examples = []
    for idx, (category, item_id) in enumerate(selected[:20], start=1):
        panels = []
        text_lines = [f"{category} | {item_id}", f"STWM correct={stwm_correct(item_id)}"]
        for name in ["cutie", "sam2", "cotracker"]:
            row = by_ext.get(name, {}).get(item_id)
            if not row:
                continue
            text_lines.append(f"{name}: pred={row.get('predicted_candidate_id')} gt={row.get('gt_candidate_id')} correct={row.get('top1_correct')}")
            overlay = ((row.get("visual_overlays") or {}).get("smoke_contact_sheet") or (row.get("visual_overlays") or {}).get("predicted_vs_gt_overlay"))
            if overlay:
                fp = ROOT / overlay
                if fp.exists():
                    img = cv2.imread(str(fp))
                    if img is not None:
                        scale = min(1.0, 900 / max(img.shape[:2]))
                        if scale < 1:
                            img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
                        panels.append(img)
        if not panels:
            continue
        sheet = np.vstack([pad_to_width(p, max(x.shape[1] for x in panels)) for p in panels])
        header = np.full((90, sheet.shape[1], 3), 245, dtype=np.uint8)
        y = 22
        for line in text_lines[:4]:
            cv2.putText(header, line[:150], (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
            y += 20
        final = np.vstack([header, sheet])
        out_path = out_dir / f"external_baseline_comparison_{idx:02d}_{safe_name(item_id)}.png"
        cv2.imwrite(str(out_path), final)
        examples.append({"category": category, "item_id": item_id, "path": rel(out_path), "stwm_correct": stwm_correct(item_id)})

    report = {
        "created_at": now(),
        "output_dir": rel(out_dir),
        "requested_min_examples": 20,
        "generated_examples": len(examples),
        "examples": examples,
        "selection_categories": list(categories.keys()),
    }
    write_json(REPORTS / "stwm_external_baseline_visual_examples_20260426.json", report)
    lines = ["| # | category | item_id | path |", "|---:|---|---|---|"]
    for idx, ex in enumerate(examples, start=1):
        lines.append(f"| {idx} | {ex['category']} | `{ex['item_id']}` | `{ex['path']}` |")
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_VISUAL_EXAMPLES_20260426.md", "STWM External Baseline Visual Examples 20260426", lines)
    return report


def pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
    if img.shape[1] >= width:
        return img
    pad = np.full((img.shape[0], width - img.shape[1], 3), 255, dtype=np.uint8)
    return np.hstack([img, pad])


def decide(summary: dict[str, Any], bootstrap: dict[str, Any], reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    strongest = summary.get("strongest_external_baseline")
    stwm_overall = summary["per_method_overall"].get("stwm_trace_belief_assoc", {})

    def improved_vs(ext: str | None) -> bool | None:
        if not ext:
            return None
        comp = bootstrap.get("comparisons", {}).get(f"STWM_vs_{ext}")
        if not comp:
            return None
        top = comp.get("top1", {})
        mrr = comp.get("MRR", {})
        return bool((top.get("mean_delta") or 0) > 0 and (mrr.get("mean_delta") or 0) >= 0)

    improved = {name: improved_vs(name) for name in ["cutie", "sam2", "cotracker"]}
    improved_strongest = improved_vs(strongest)
    false_strong = None
    hard_strong = None
    if strongest:
        comp = bootstrap.get("comparisons", {}).get(f"STWM_vs_{strongest}") or {}
        false_strong = (comp.get("false_confuser_rate", {}).get("mean_delta") or 0) > 0
        hard_deltas = [
            comp.get("long_gap_persistence_top1", {}).get("mean_delta"),
            comp.get("occlusion_reappearance_top1", {}).get("mean_delta"),
        ]
        hard_strong = any(x is not None and x > 0 for x in hard_deltas)

    completed = {name: bool(report.get("completed")) for name, report in reports.items()}
    main = []
    appendix = []
    for name, ok in completed.items():
        if not ok:
            appendix.append(name)
        elif strongest == name or improved.get(name) is False:
            main.append(name)
        else:
            appendix.append(name)
    if any(completed.values()) and (improved_strongest or false_strong or hard_strong):
        next_step = "add_external_baselines_to_main_paper"
    elif any(completed.values()):
        next_step = "add_external_baselines_to_appendix_only"
    else:
        next_step = "external_baselines_not_reliable_do_not_use"

    decision = {
        "created_at": now(),
        "cutie_completed": completed.get("cutie", False),
        "sam2_completed": completed.get("sam2", False),
        "cotracker_completed": completed.get("cotracker", False),
        "strongest_external_baseline": strongest,
        "stwm_improved_vs_cutie": improved.get("cutie"),
        "stwm_improved_vs_sam2": improved.get("sam2"),
        "stwm_improved_vs_cotracker": improved.get("cotracker"),
        "stwm_improved_vs_strongest_external": improved_strongest,
        "stwm_false_confuser_improved_vs_strongest_external": false_strong,
        "stwm_long_gap_or_occlusion_improved_vs_strongest_external": hard_strong,
        "recommended_main_paper_external_baselines": main,
        "recommended_appendix_external_baselines": appendix,
        "next_step_choice": next_step,
        "notes": [
            "External full eval is deterministic frozen inference on the materialized manifest; smoke scores were not reused.",
            "STWM official comparison uses existing official reports only; no official STWM result was modified.",
            "If CoTracker overall is stronger but STWM is stronger on selected hard subsets, report that decomposition honestly.",
        ],
    }
    write_json(REPORTS / "stwm_external_baseline_full_eval_decision_20260426.json", decision)
    write_md_json(DOCS / "STWM_EXTERNAL_BASELINE_FULL_EVAL_DECISION_20260426.md", "STWM External Baseline Full Eval Decision 20260426", decision)
    return decision


def main() -> None:
    audit = source_audit()
    if not audit["audit_passed"]:
        raise SystemExit("full eval source audit failed; see reports/stwm_external_baseline_full_eval_source_audit_20260426.json")

    manifest = load_json(MANIFEST)
    items = manifest.get("items") or []
    smoke_summary = load_json(SMOKE_SUMMARY)
    plan = item_plan(items, smoke_summary)
    smoke_pass = {
        "cutie": bool(audit.get("cutie_smoke_pass")),
        "sam2": bool(audit.get("sam2_smoke_pass")),
        "cotracker": bool(audit.get("cotracker_smoke_pass")),
    }

    requested = os.environ.get("STWM_EXTERNAL_BASELINES")
    selected = [x.strip() for x in requested.split(",") if x.strip()] if requested else ["cutie", "sam2", "cotracker"]
    summary_only = os.environ.get("STWM_EXTERNAL_SUMMARY_ONLY") == "1"

    reports: dict[str, dict[str, Any]] = {}
    if not summary_only:
        runners = {
            "cutie": CutieSmokeRunner,
            "sam2": SAM2SmokeRunner,
            "cotracker": CoTrackerSmokeRunner,
        }
        for name in selected:
            if name not in runners:
                raise SystemExit(f"unknown baseline requested: {name}")
            reports[name] = run_external_baseline(name, runners[name](), items, smoke_pass[name])

    for name in ["cutie", "sam2", "cotracker"]:
        if name not in reports and BASELINE_REPORT_PATHS[name].exists():
            reports[name] = load_json(BASELINE_REPORT_PATHS[name])

    if set(reports) != {"cutie", "sam2", "cotracker"}:
        missing = sorted(set(["cutie", "sam2", "cotracker"]) - set(reports))
        incomplete = {
            "created_at": now(),
            "exact_blocking_reason": f"missing baseline full eval reports: {missing}",
            "available_reports": sorted(reports),
        }
        write_json(REPORTS / "stwm_external_baseline_full_eval_summary_20260426.json", incomplete)
        write_md_json(DOCS / "STWM_EXTERNAL_BASELINE_FULL_EVAL_SUMMARY_20260426.md", "STWM External Baseline Full Eval Summary 20260426", incomplete)
        return

    summary, bootstrap = write_summary_and_bootstrap(reports, items)
    make_visual_examples(reports, items)
    decide(summary, bootstrap, reports)


if __name__ == "__main__":
    main()
