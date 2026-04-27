from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stwm.tools.external_baselines.common_io import DOCS, REPORTS, read_report_or_none, write_json, write_markdown  # noqa: E402


def main() -> None:
    clone = read_report_or_none(REPORTS / "stwm_external_baseline_clone_audit_20260426.json") or {}
    env = read_report_or_none(REPORTS / "stwm_external_baseline_env_audit_20260426.json") or {}
    item = read_report_or_none(REPORTS / "stwm_external_baseline_item_audit_20260426.json") or {}
    smoke = read_report_or_none(REPORTS / "stwm_external_baseline_smoke_20260426.json") or {}
    full = read_report_or_none(REPORTS / "stwm_external_baseline_eval_20260426.json") or {}

    full_results = full.get("per_baseline_results", {})
    completed = {k: v for k, v in full_results.items() if v.get("full_eval_completed")}
    bootstrap_comparisons = {}
    for name in ["cutie", "sam2", "cotracker"]:
        bootstrap_comparisons[f"STWM_vs_{name}"] = {
            "available": name in completed,
            "metrics": {},
            "exact_blocking_reason": None
            if name in completed
            else (full_results.get(name, {}) or {}).get("exact_blocking_reason", "full_eval_not_completed"),
        }
    bootstrap = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "comparisons": bootstrap_comparisons,
        "STWM_vs_best_external_baseline": {
            "available": bool(completed),
            "metrics": {},
            "exact_blocking_reason": None if completed else "no_external_baseline_completed_full_eval",
        },
    }
    write_json(REPORTS / "stwm_external_baseline_bootstrap_20260426.json", bootstrap)
    lines = [
        "| comparison | available | blocking_reason |",
        "|---|---:|---|",
    ]
    for name, c in bootstrap_comparisons.items():
        lines.append(f"| {name} | `{c['available']}` | {c.get('exact_blocking_reason') or ''} |")
    lines.append(
        f"| STWM_vs_best_external_baseline | `{bootstrap['STWM_vs_best_external_baseline']['available']}` | {bootstrap['STWM_vs_best_external_baseline'].get('exact_blocking_reason') or ''} |"
    )
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_BOOTSTRAP_20260426.md", "STWM External Baseline Bootstrap 20260426", lines)

    env_entries = env.get("entries", {})
    smoke_results = smoke.get("results", {})
    clone_entries = clone.get("entries", {})
    ready = {name: bool(full_results.get(name, {}).get("full_eval_completed")) for name in ["cutie", "sam2", "cotracker"]}
    best_external = None
    decision = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "successful_clone": {name: bool(clone_entries.get(name, {}).get("clone_success")) for name in ["cutie", "sam2", "cotracker"]},
        "successful_import": {name: bool(env_entries.get(name, {}).get("import_ok")) for name in ["cutie", "sam2", "cotracker"]},
        "successful_smoke_test": {name: bool(smoke_results.get(name, {}).get("smoke_pass")) for name in ["cutie", "sam2", "cotracker"]},
        "completed_full_eval": ready,
        "strongest_external_baseline": best_external,
        "cutie_ready": ready["cutie"],
        "sam2_ready": ready["sam2"],
        "cotracker_ready": ready["cotracker"],
        "best_external_baseline": best_external,
        "stwm_improved_vs_cutie": None,
        "stwm_improved_vs_sam2": None,
        "stwm_improved_vs_cotracker": None,
        "stwm_improved_vs_best_external": None,
        "external_baseline_main_paper_ready": False,
        "recommended_main_paper_baselines": [],
        "recommended_appendix_baselines": [],
        "related_work_only_baselines": [
            name for name in ["cutie", "sam2", "cotracker"] if not ready[name]
        ],
        "blocking_summary": {
            "env": {k: v.get("exact_blocking_reason") for k, v in env_entries.items()},
            "item": item.get("exact_blocking_reason"),
            "smoke": {k: v.get("exact_blocking_reason") for k, v in smoke_results.items()},
            "full_eval": {k: v.get("exact_blocking_reason") for k, v in full_results.items()},
        },
        "next_step_choice": "fix_failed_baseline_install"
        if item.get("runnable_items", 0) > 0
        else "do_not_use_external_baselines",
        "exact_blocking_reason": "External repos cloned and importable in the isolated stwm conda env, but no baseline can enter full eval because the current STWM reports do not expose raw frame/video paths plus observed prompts and future candidate masks/boxes needed for frozen VOS/tracking adaptation.",
    }
    write_json(REPORTS / "stwm_external_baseline_final_decision_20260426.json", decision)
    lines = [
        "| baseline | cloned | import_ok | smoke_pass | full_eval |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["cutie", "sam2", "cotracker"]:
        lines.append(
            f"| {name} | `{decision['successful_clone'][name]}` | `{decision['successful_import'][name]}` | `{decision['successful_smoke_test'][name]}` | `{decision['completed_full_eval'][name]}` |"
        )
    lines.extend(
        [
            "",
            f"- best_external_baseline: `{best_external}`",
            f"- stwm_improved_vs_best_external: `{decision['stwm_improved_vs_best_external']}`",
            f"- next_step_choice: `{decision['next_step_choice']}`",
            f"- exact_blocking_reason: `{decision['exact_blocking_reason']}`",
        ]
    )
    write_markdown(DOCS / "STWM_EXTERNAL_BASELINE_FINAL_DECISION_20260426.md", "STWM External Baseline Final Decision 20260426", lines)


if __name__ == "__main__":
    main()
