#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    p = ArgumentParser(description="Summarize Stage2 core-only mainline training round")
    p.add_argument("--input-run-json", required=True)
    p.add_argument("--progress-json", required=True)
    p.add_argument("--final-json", required=True)
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


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _f(x: Any, default: float = 1e9) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _rank_key(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        _f(metrics.get("free_rollout_endpoint_l2"), 1e9),
        _f(metrics.get("free_rollout_coord_mean_l2"), 1e9),
        _f(metrics.get("teacher_forced_coord_loss"), 1e9),
    )


def _metric_triplet(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {
        "teacher_forced_coord_loss": _f(metrics.get("teacher_forced_coord_loss"), 1e9),
        "free_rollout_coord_mean_l2": _f(metrics.get("free_rollout_coord_mean_l2"), 1e9),
        "free_rollout_endpoint_l2": _f(metrics.get("free_rollout_endpoint_l2"), 1e9),
        "total_loss_reference": _f(metrics.get("total_loss_reference"), 1e9),
    }


def _boundary_ok(run: Dict[str, Any]) -> bool:
    freeze = run.get("freeze_trainable_boundary", {}) if isinstance(run.get("freeze_trainable_boundary", {}), dict) else {}
    return bool(
        bool(run.get("boundary_ok", False))
        and bool(freeze.get("boundary_ok", False))
        and int(freeze.get("stage1_trainable_parameter_count", 0) or 0) == 0
        and (not bool(freeze.get("stage1_grad_detected_after_backward", True)))
    )


def _curve_still_improving(run: Dict[str, Any]) -> bool:
    history = run.get("eval_history", []) if isinstance(run.get("eval_history", []), list) else []
    valid = [x for x in history if isinstance(x, dict) and isinstance(x.get("metrics", {}), dict)]
    if len(valid) < 2:
        best_step = int((run.get("best_checkpoint_metric", {}) if isinstance(run.get("best_checkpoint_metric", {}), dict) else {}).get("global_step", -1) or -1)
        latest_step = int((run.get("latest_checkpoint_metric", {}) if isinstance(run.get("latest_checkpoint_metric", {}), dict) else {}).get("global_step", -1) or -1)
        return bool(best_step >= 0 and best_step == latest_step)

    prev_key = _rank_key(valid[-2].get("metrics", {}))
    last_key = _rank_key(valid[-1].get("metrics", {}))
    if last_key < prev_key:
        return True

    best_step = int((run.get("best_checkpoint_metric", {}) if isinstance(run.get("best_checkpoint_metric", {}), dict) else {}).get("global_step", -1) or -1)
    latest_step = int((run.get("latest_checkpoint_metric", {}) if isinstance(run.get("latest_checkpoint_metric", {}), dict) else {}).get("global_step", -1) or -1)
    return bool(best_step >= 0 and best_step == latest_step)


def _checkpoint_exists(path_value: Any) -> bool:
    p = Path(str(path_value or "")).expanduser()
    return bool(p.exists())


def _next_step_choice(stable: bool, boundary_ok: bool, improving: bool) -> str:
    if (not stable) or (not boundary_ok):
        return "do_one_targeted_stage2_fix"
    if improving:
        return "continue_stage2_training_core_only"
    return "freeze_stage2_core_mainline"


def _write_md(path: str | Path, final_payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    fm = final_payload.get("final_metrics", {}) if isinstance(final_payload.get("final_metrics", {}), dict) else {}
    best = final_payload.get("best_checkpoint_metric", {}) if isinstance(final_payload.get("best_checkpoint_metric", {}), dict) else {}
    latest = final_payload.get("latest_checkpoint_metric", {}) if isinstance(final_payload.get("latest_checkpoint_metric", {}), dict) else {}

    lines = [
        "# Stage2 Core Mainline Train Results",
        "",
        f"- generated_at_utc: {final_payload.get('generated_at_utc', '')}",
        f"- run_name: {final_payload.get('run_name', '')}",
        f"- current_mainline_semantic_source: {final_payload.get('current_mainline_semantic_source', '')}",
        f"- frozen_boundary_kept_correct: {bool(final_payload.get('frozen_boundary_kept_correct', False))}",
        f"- current_stage2_mainline_stable: {bool(final_payload.get('current_stage2_mainline_stable', False))}",
        f"- whether_curve_is_still_improving: {bool(final_payload.get('whether_curve_is_still_improving', False))}",
        f"- next_step_choice: {final_payload.get('next_step_choice', '')}",
        "",
        "## Dataset Binding",
        f"- datasets_bound_for_train: {final_payload.get('datasets_bound_for_train', [])}",
        f"- datasets_bound_for_eval: {final_payload.get('datasets_bound_for_eval', [])}",
        "",
        "## Metrics",
        f"- teacher_forced_coord_loss: {float(fm.get('teacher_forced_coord_loss', 1e9)):.6f}",
        f"- free_rollout_coord_mean_l2: {float(fm.get('free_rollout_coord_mean_l2', 1e9)):.6f}",
        f"- free_rollout_endpoint_l2: {float(fm.get('free_rollout_endpoint_l2', 1e9)):.6f}",
        "",
        "## Checkpoint Metrics",
        f"- best_checkpoint_metric.global_step: {int(best.get('global_step', -1) or -1)}",
        f"- latest_checkpoint_metric.global_step: {int(latest.get('global_step', -1) or -1)}",
        "",
        "## Training Progress",
        f"- optimizer_steps: {int((final_payload.get('training_progress', {}) if isinstance(final_payload.get('training_progress', {}), dict) else {}).get('optimizer_steps', 0) or 0)}",
        f"- effective_batch: {int((final_payload.get('training_progress', {}) if isinstance(final_payload.get('training_progress', {}), dict) else {}).get('effective_batch', 0) or 0)}",
        f"- epochs_completed: {float((final_payload.get('training_progress', {}) if isinstance(final_payload.get('training_progress', {}), dict) else {}).get('epochs_completed', 0.0) or 0.0):.6f}",
        f"- eval_interval: {int((final_payload.get('training_progress', {}) if isinstance(final_payload.get('training_progress', {}), dict) else {}).get('eval_interval', 0) or 0)}",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run = _read_json(args.input_run_json)

    source = str(run.get("current_mainline_semantic_source", ""))
    boundary_ok = _boundary_ok(run)
    run_stable = bool(run.get("run_stable", False))
    current_stage2_mainline_stable = bool(run_stable and boundary_ok and source == "crop_visual_encoder")
    improving = _curve_still_improving(run)
    next_step_choice = _next_step_choice(current_stage2_mainline_stable, boundary_ok, improving)

    final_metrics = _metric_triplet(run.get("final_metrics", {}) if isinstance(run.get("final_metrics", {}), dict) else {})
    best_metric = run.get("best_checkpoint_metric", {}) if isinstance(run.get("best_checkpoint_metric", {}), dict) else {}
    latest_metric = run.get("latest_checkpoint_metric", {}) if isinstance(run.get("latest_checkpoint_metric", {}), dict) else {}
    checkpoint_inventory = run.get("checkpoint_inventory", {}) if isinstance(run.get("checkpoint_inventory", {}), dict) else {}

    datasets_train = [str(x) for x in run.get("datasets_bound_for_train", [])] if isinstance(run.get("datasets_bound_for_train", []), list) else []
    datasets_eval = [str(x) for x in run.get("datasets_bound_for_eval", [])] if isinstance(run.get("datasets_bound_for_eval", []), list) else []

    training_progress = run.get("training_progress", {}) if isinstance(run.get("training_progress", {}), dict) else {}

    progress_payload = {
        "generated_at_utc": now_iso(),
        "round": "stage2_core_mainline_train_20260408",
        "run_name": str(run.get("run_name", "")),
        "status": "completed",
        "training_budget": run.get("training_budget", {}),
        "training_progress": {
            "optimizer_steps": int(training_progress.get("optimizer_steps", 0) or 0),
            "effective_batch": int(training_progress.get("effective_batch", 0) or 0),
            "epochs_completed": float(training_progress.get("epochs_completed", 0.0) or 0.0),
            "eval_interval": int(training_progress.get("eval_interval", 0) or 0),
            "save_every_n_steps": int(training_progress.get("save_every_n_steps", 0) or 0),
        },
        "checkpoints": {
            "checkpoint_dir": str(checkpoint_inventory.get("checkpoint_dir", "")),
            "best": str(checkpoint_inventory.get("best", "")),
            "latest": str(checkpoint_inventory.get("latest", "")),
            "best_exists": _checkpoint_exists(checkpoint_inventory.get("best", "")),
            "latest_exists": _checkpoint_exists(checkpoint_inventory.get("latest", "")),
            "step_checkpoints": checkpoint_inventory.get("step_checkpoints", []),
            "resume_from": str(checkpoint_inventory.get("resume_from", "")),
            "auto_resume_latest": bool(checkpoint_inventory.get("auto_resume_latest", False)),
        },
        "final_metrics": final_metrics,
        "best_checkpoint_metric": best_metric,
        "latest_checkpoint_metric": latest_metric,
        "frozen_parameter_count": int(run.get("frozen_parameter_count", 0) or 0),
        "trainable_parameter_count": int(run.get("trainable_parameter_count", 0) or 0),
        "boundary_ok": bool(boundary_ok),
        "current_mainline_semantic_source": source,
        "datasets_bound_for_train": datasets_train,
        "datasets_bound_for_eval": datasets_eval,
        "comparison_sorting": run.get("comparison_sorting", {}),
    }

    final_payload = {
        "generated_at_utc": now_iso(),
        "round": "stage2_core_mainline_train_20260408",
        "run_name": str(run.get("run_name", "")),
        "current_mainline_semantic_source": source,
        "frozen_boundary_kept_correct": bool(boundary_ok),
        "current_stage2_mainline_stable": bool(current_stage2_mainline_stable),
        "whether_curve_is_still_improving": bool(improving),
        "allowed_next_step_choice": [
            "continue_stage2_training_core_only",
            "freeze_stage2_core_mainline",
            "do_one_targeted_stage2_fix",
        ],
        "next_step_choice": str(next_step_choice),
        "comparison_sorting": {
            "primary": "free_rollout_endpoint_l2",
            "secondary": "free_rollout_coord_mean_l2",
            "tertiary": "teacher_forced_coord_loss",
            "total_loss_usage": "reference_only",
        },
        "final_metrics": final_metrics,
        "best_checkpoint_metric": best_metric,
        "latest_checkpoint_metric": latest_metric,
        "frozen_parameter_count": int(run.get("frozen_parameter_count", 0) or 0),
        "trainable_parameter_count": int(run.get("trainable_parameter_count", 0) or 0),
        "boundary_ok": bool(boundary_ok),
        "stage1_trainable_parameter_count": int(
            (run.get("freeze_trainable_boundary", {}) if isinstance(run.get("freeze_trainable_boundary", {}), dict) else {}).get("stage1_trainable_parameter_count", 0)
            or 0
        ),
        "datasets_bound_for_train": datasets_train,
        "datasets_bound_for_eval": datasets_eval,
        "training_progress": progress_payload["training_progress"],
        "checkpoint_inventory": progress_payload["checkpoints"],
    }

    _write_json(args.progress_json, progress_payload)
    _write_json(args.final_json, final_payload)
    _write_md(args.results_md, final_payload)

    print(f"[stage2-core-mainline-summary] progress_json={args.progress_json}")
    print(f"[stage2-core-mainline-summary] final_json={args.final_json}")
    print(f"[stage2-core-mainline-summary] results_md={args.results_md}")
    print(f"[stage2-core-mainline-summary] current_mainline_semantic_source={final_payload['current_mainline_semantic_source']}")
    print(f"[stage2-core-mainline-summary] frozen_boundary_kept_correct={final_payload['frozen_boundary_kept_correct']}")
    print(f"[stage2-core-mainline-summary] next_step_choice={final_payload['next_step_choice']}")


if __name__ == "__main__":
    main()
