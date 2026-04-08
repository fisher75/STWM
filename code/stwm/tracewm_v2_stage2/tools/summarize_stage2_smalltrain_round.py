#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    p = ArgumentParser(description="Summarize Stage2 small-train core vs core+burst runs")
    p.add_argument("--core-run-json", required=True)
    p.add_argument("--core-plus-burst-run-json", required=True)
    p.add_argument("--runs-json", required=True)
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--results-md", required=True)
    return p.parse_args()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _f(x: Any, default: float = 1e9) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    fm = run.get("final_metrics", {}) if isinstance(run.get("final_metrics", {}), dict) else {}
    return {
        "teacher_forced_coord_loss": _f(fm.get("teacher_forced_coord_loss"), 1e9),
        "free_rollout_coord_mean_l2": _f(fm.get("free_rollout_coord_mean_l2"), 1e9),
        "free_rollout_endpoint_l2": _f(fm.get("free_rollout_endpoint_l2"), 1e9),
        "tapvid_style_eval": fm.get("tapvid_style_eval", {}),
        "tapvid3d_limited_eval": fm.get("tapvid3d_limited_eval", {}),
        "semantic_branch_metrics": fm.get("semantic_branch_metrics", {}),
    }


def _boundary_ok(run: Dict[str, Any]) -> bool:
    b = run.get("freeze_trainable_boundary", {}) if isinstance(run.get("freeze_trainable_boundary", {}), dict) else {}
    return bool(
        bool(b.get("boundary_ok", False))
        and int(b.get("stage1_trainable_parameter_count", 0)) == 0
        and (not bool(b.get("stage1_grad_detected_after_backward", True)))
    )


def _run_stable(run: Dict[str, Any]) -> bool:
    m = _metrics(run)
    stable_metrics = (
        m["teacher_forced_coord_loss"] < 1e8
        and m["free_rollout_coord_mean_l2"] < 1e8
        and m["free_rollout_endpoint_l2"] < 1e8
    )
    return bool(bool(run.get("run_stable", False)) and stable_metrics)


def _better_than(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ma = _metrics(a)
    mb = _metrics(b)
    key_a = (ma["free_rollout_endpoint_l2"], ma["free_rollout_coord_mean_l2"], ma["teacher_forced_coord_loss"])
    key_b = (mb["free_rollout_endpoint_l2"], mb["free_rollout_coord_mean_l2"], mb["teacher_forced_coord_loss"])
    return bool(key_a < key_b)


def _write_md(path: Path, runs_payload: Dict[str, Any], comparison: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    core = runs_payload.get("runs", {}).get("stage2_smalltrain_core", {})
    burst = runs_payload.get("runs", {}).get("stage2_smalltrain_core_plus_burst", {})

    core_m = _metrics(core) if isinstance(core, dict) else {}
    burst_m = _metrics(burst) if isinstance(burst, dict) else {}

    lines = [
        "# Stage2 Small-Train Results",
        "",
        f"- generated_at_utc: {comparison.get('generated_at_utc', '')}",
        f"- smalltrain_status: {comparison.get('smalltrain_status', '')}",
        f"- next_step_choice: {comparison.get('next_step_choice', '')}",
        "",
        "## Run Metrics",
        "| run | teacher_forced_coord_loss | free_rollout_coord_mean_l2 | free_rollout_endpoint_l2 | parameter_count_frozen | parameter_count_trainable |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| stage2_smalltrain_core | {float(core_m.get('teacher_forced_coord_loss', 1e9)):.6f} "
            f"| {float(core_m.get('free_rollout_coord_mean_l2', 1e9)):.6f} "
            f"| {float(core_m.get('free_rollout_endpoint_l2', 1e9)):.6f} "
            f"| {int((core or {}).get('parameter_count_frozen', 0))} "
            f"| {int((core or {}).get('parameter_count_trainable', 0))} |"
        ),
        (
            f"| stage2_smalltrain_core_plus_burst | {float(burst_m.get('teacher_forced_coord_loss', 1e9)):.6f} "
            f"| {float(burst_m.get('free_rollout_coord_mean_l2', 1e9)):.6f} "
            f"| {float(burst_m.get('free_rollout_endpoint_l2', 1e9)):.6f} "
            f"| {int((burst or {}).get('parameter_count_frozen', 0))} "
            f"| {int((burst or {}).get('parameter_count_trainable', 0))} |"
        ),
        "",
        "## Mandatory Answers",
        f"- core_only_stable: {bool(comparison.get('core_only_stable', False))}",
        f"- core_plus_burst_better_than_core_only: {bool(comparison.get('core_plus_burst_better_than_core_only', False))}",
        f"- stage1_frozen_boundary_kept_correct: {bool(comparison.get('stage1_frozen_boundary_kept_correct', False))}",
        f"- smalltrain_status: {comparison.get('smalltrain_status', '')}",
        f"- next_step_choice: {comparison.get('next_step_choice', '')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    core = _read_json(args.core_run_json)
    burst = _read_json(args.core_plus_burst_run_json)

    runs_payload = {
        "generated_at_utc": now_iso(),
        "round": "stage2_smalltrain_20260408",
        "runs": {
            "stage2_smalltrain_core": core,
            "stage2_smalltrain_core_plus_burst": burst,
        },
        "selection_policy": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "available_eval_metric",
            "total_loss_usage": "reference_only",
        },
    }

    core_stable = _run_stable(core)
    burst_better = _better_than(burst, core)
    boundary_ok = bool(_boundary_ok(core) and _boundary_ok(burst))

    if core_stable and boundary_ok:
        smalltrain_status = "smalltrain_successful"
        if burst_better:
            next_step_choice = "continue_stage2_training"
        else:
            next_step_choice = "keep_core_only_and_continue"
    else:
        smalltrain_status = "needs_bootstrap_fix"
        next_step_choice = "refine_stage2_smalltrain"

    comparison = {
        "generated_at_utc": now_iso(),
        "round": "stage2_smalltrain_20260408",
        "core_only_stable": bool(core_stable),
        "core_plus_burst_better_than_core_only": bool(burst_better),
        "stage1_frozen_boundary_kept_correct": bool(boundary_ok),
        "smalltrain_status": str(smalltrain_status),
        "allowed_next_step_choice": [
            "continue_stage2_training",
            "keep_core_only_and_continue",
            "refine_stage2_smalltrain",
        ],
        "next_step_choice": str(next_step_choice),
        "core_only_final_metrics": _metrics(core),
        "core_plus_burst_final_metrics": _metrics(burst),
        "core_only_checkpoint_dir": str((core.get("checkpoint_inventory", {}) if isinstance(core.get("checkpoint_inventory", {}), dict) else {}).get("checkpoint_dir", "")),
        "core_plus_burst_checkpoint_dir": str((burst.get("checkpoint_inventory", {}) if isinstance(burst.get("checkpoint_inventory", {}), dict) else {}).get("checkpoint_dir", "")),
    }

    runs_json = Path(args.runs_json)
    comparison_json = Path(args.comparison_json)
    results_md = Path(args.results_md)
    _write_json(runs_json, runs_payload)
    _write_json(comparison_json, comparison)
    _write_md(results_md, runs_payload, comparison)

    print(f"[stage2-smalltrain-summary] runs_json={runs_json}")
    print(f"[stage2-smalltrain-summary] comparison_json={comparison_json}")
    print(f"[stage2-smalltrain-summary] results_md={results_md}")
    print(f"[stage2-smalltrain-summary] smalltrain_status={comparison['smalltrain_status']}")
    print(f"[stage2-smalltrain-summary] next_step_choice={comparison['next_step_choice']}")


if __name__ == "__main__":
    main()