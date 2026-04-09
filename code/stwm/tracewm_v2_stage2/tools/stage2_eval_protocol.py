#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json


SORT_PRIMARY = "free_rollout_endpoint_l2"
SORT_SECONDARY = "free_rollout_coord_mean_l2"
SORT_TERTIARY = "teacher_forced_coord_loss"
SORT_TOTAL_LOSS_USAGE = "reference_only"

MAINLINE_CORE = "stage2_core_cropenc"
MAINLINE_BURST = "stage2_core_plus_burst_cropenc"
MAINLINE_INVALID = "invalid_comparison"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _metrics_from_payload(payload: Dict[str, Any]) -> Dict[str, float]:
    return {
        "teacher_forced_coord_loss": _f(payload.get("teacher_forced_coord_loss"), 1e9),
        "free_rollout_coord_mean_l2": _f(payload.get("free_rollout_coord_mean_l2"), 1e9),
        "free_rollout_endpoint_l2": _f(payload.get("free_rollout_endpoint_l2"), 1e9),
        "total_loss_reference": _f(payload.get("total_loss_reference"), 1e9),
    }


def _metrics_from_run(run: Dict[str, Any]) -> Dict[str, float]:
    fm = run.get("final_metrics", {}) if isinstance(run.get("final_metrics", {}), dict) else {}
    return _metrics_from_payload(fm)


def _metric_rank_key(metrics: Dict[str, float]) -> List[float]:
    return [
        float(metrics.get("free_rollout_endpoint_l2", 1e9)),
        float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
        float(metrics.get("teacher_forced_coord_loss", 1e9)),
    ]


def _extract_best_checkpoint_metric(run: Dict[str, Any], fallback: Dict[str, float]) -> Dict[str, Any]:
    best = run.get("best_metric_so_far", {}) if isinstance(run.get("best_metric_so_far", {}), dict) else {}
    best_metrics = best.get("metrics", {}) if isinstance(best.get("metrics", {}), dict) else {}
    metric_triplet = _metrics_from_payload(best_metrics) if best_metrics else dict(fallback)
    return {
        "global_step": int(best.get("global_step", -1) or -1),
        "metrics": metric_triplet,
        "rank_key": _metric_rank_key(metric_triplet),
    }


def _extract_latest_checkpoint_metric(run: Dict[str, Any], fallback: Dict[str, float]) -> Dict[str, Any]:
    history = run.get("eval_history", []) if isinstance(run.get("eval_history", []), list) else []
    latest_event = history[-1] if history and isinstance(history[-1], dict) else {}
    latest_metrics = latest_event.get("metrics", {}) if isinstance(latest_event.get("metrics", {}), dict) else {}
    metric_triplet = _metrics_from_payload(latest_metrics) if latest_metrics else dict(fallback)
    return {
        "global_step": int(latest_event.get("global_step", -1) or -1),
        "metrics": metric_triplet,
        "rank_key": _metric_rank_key(metric_triplet),
    }


def _split_counts_used(run: Dict[str, Any], split: str) -> Dict[str, int]:
    dataset_summary = run.get("dataset_summary", {}) if isinstance(run.get("dataset_summary", {}), dict) else {}
    split_payload = dataset_summary.get(split, {}) if isinstance(dataset_summary.get(split, {}), dict) else {}
    out: Dict[str, int] = {}
    for name, meta in split_payload.items():
        if not isinstance(meta, dict):
            continue
        out[str(name)] = int(meta.get("sample_count", 0) or 0)
    return out


def _boundary_ok(run: Dict[str, Any]) -> bool:
    b = run.get("freeze_trainable_boundary", {}) if isinstance(run.get("freeze_trainable_boundary", {}), dict) else {}
    return bool(
        bool(b.get("boundary_ok", False))
        and int(b.get("stage1_trainable_parameter_count", 0) or 0) == 0
        and (not bool(b.get("stage1_grad_detected_after_backward", True)))
    )


def _selection_policy() -> Dict[str, str]:
    return {
        "primary": SORT_PRIMARY,
        "secondary": SORT_SECONDARY,
        "tertiary": SORT_TERTIARY,
        "total_loss_usage": SORT_TOTAL_LOSS_USAGE,
    }


def _crop_encoder_ran_through(summary: Dict[str, Any]) -> bool:
    source_ok = str(summary.get("current_mainline_semantic_source", "")) == "crop_visual_encoder"
    sb = summary.get("semantic_branch_metrics", {}) if isinstance(summary.get("semantic_branch_metrics", {}), dict) else {}
    nonempty = _f(sb.get("train_semantic_input_nonempty_ratio"), 0.0)
    final_sb = summary.get("final_semantic_branch_metrics", {}) if isinstance(summary.get("final_semantic_branch_metrics", {}), dict) else {}
    eval_nonempty = _f(final_sb.get("semantic_input_nonempty_ratio"), 0.0)
    return bool(source_ok and (nonempty > 0.0 or eval_nonempty > 0.0))


def _run_datasets_bound(summary: Dict[str, Any]) -> List[str]:
    binding = summary.get("stage2_data_binding", {}) if isinstance(summary.get("stage2_data_binding", {}), dict) else {}
    runs = binding.get("run_datasets", []) if isinstance(binding.get("run_datasets", []), list) else []
    return [str(x) for x in runs]


def _training_budget(summary: Dict[str, Any]) -> Dict[str, int]:
    b = summary.get("training_budget", {}) if isinstance(summary.get("training_budget", {}), dict) else {}
    return {
        "train_steps_target": int(b.get("train_steps_target", 0) or 0),
        "batch_size": int(b.get("batch_size", 0) or 0),
        "eval_interval": int(b.get("eval_interval", 0) or 0),
        "eval_max_batches": int(b.get("eval_max_batches", 0) or 0),
        "save_every_n_steps": int(b.get("save_every_n_steps", 0) or 0),
    }


def build_eval_fix_run_summary(run: Dict[str, Any], run_label: str = "") -> Dict[str, Any]:
    final_metrics = _metrics_from_run(run)
    best_checkpoint_metric = _extract_best_checkpoint_metric(run, fallback=final_metrics)
    latest_checkpoint_metric = _extract_latest_checkpoint_metric(run, fallback=final_metrics)
    train_counts = _split_counts_used(run, "train")
    val_counts = _split_counts_used(run, "val")

    stage2_binding = run.get("stage2_data_binding", {}) if isinstance(run.get("stage2_data_binding", {}), dict) else {}
    checkpoint_inventory = run.get("checkpoint_inventory", {}) if isinstance(run.get("checkpoint_inventory", {}), dict) else {}
    freeze = run.get("freeze_trainable_boundary", {}) if isinstance(run.get("freeze_trainable_boundary", {}), dict) else {}
    final_raw = run.get("final_metrics", {}) if isinstance(run.get("final_metrics", {}), dict) else {}
    final_semantic_branch = final_raw.get("semantic_branch_metrics", {}) if isinstance(final_raw.get("semantic_branch_metrics", {}), dict) else {}

    out = {
        "generated_at_utc": now_iso(),
        "eval_fix_protocol_version": "stage2_eval_fix_20260408",
        "run_name": str(run.get("run_name", run_label or "")),
        "source_round": str(run.get("run_name", "")),
        "objective": str(run.get("objective", "")),
        "current_mainline_semantic_source": str(run.get("current_mainline_semantic_source", "")),
        "legacy_semantic_source": str(run.get("legacy_semantic_source", "")),
        "comparison_sorting": _selection_policy(),
        "teacher_forced_coord_loss": float(final_metrics["teacher_forced_coord_loss"]),
        "free_rollout_coord_mean_l2": float(final_metrics["free_rollout_coord_mean_l2"]),
        "free_rollout_endpoint_l2": float(final_metrics["free_rollout_endpoint_l2"]),
        "total_loss_reference": float(final_metrics["total_loss_reference"]),
        "best_checkpoint_metric": best_checkpoint_metric,
        "latest_checkpoint_metric": latest_checkpoint_metric,
        "train_split_counts_used": train_counts,
        "val_split_counts_used": val_counts,
        "train_split_total_count_used": int(sum(train_counts.values())),
        "val_split_total_count_used": int(sum(val_counts.values())),
        "frozen_parameter_count": int(run.get("parameter_count_frozen", 0) or 0),
        "trainable_parameter_count": int(run.get("parameter_count_trainable", 0) or 0),
        "boundary_ok": bool(_boundary_ok(run)),
        "freeze_trainable_boundary": {
            "stage1_trainable_parameter_count": int(freeze.get("stage1_trainable_parameter_count", 0) or 0),
            "semantic_trainable_parameter_count": int(freeze.get("semantic_trainable_parameter_count", 0) or 0),
            "stage1_grad_detected_after_backward": bool(freeze.get("stage1_grad_detected_after_backward", False)),
            "boundary_ok": bool(_boundary_ok(run)),
        },
        "stage2_data_binding": {
            "core": stage2_binding.get("core", []),
            "optional_extension": stage2_binding.get("optional_extension", []),
            "excluded": stage2_binding.get("excluded", []),
            "run_datasets": _run_datasets_bound(run),
        },
        "training_budget": _training_budget(run),
        "checkpoint_inventory": {
            "checkpoint_dir": str(checkpoint_inventory.get("checkpoint_dir", "")),
            "best": str(checkpoint_inventory.get("best", "")),
            "latest": str(checkpoint_inventory.get("latest", "")),
            "resume_from": str(checkpoint_inventory.get("resume_from", "")),
            "auto_resume_latest": bool(checkpoint_inventory.get("auto_resume_latest", False)),
        },
        "semantic_branch_metrics": run.get("semantic_branch_metrics", {}),
        "final_semantic_branch_metrics": final_semantic_branch,
        "run_stable": bool(run.get("run_stable", False)),
        "source_summary_generated_at_utc": str(run.get("generated_at_utc", "")),
    }
    return out


def _winner(core_value: float, burst_value: float) -> str:
    if core_value < burst_value:
        return MAINLINE_CORE
    if burst_value < core_value:
        return MAINLINE_BURST
    return "tie"


def _is_same_budget(core: Dict[str, Any], burst: Dict[str, Any]) -> bool:
    return bool(_training_budget(core) == _training_budget(burst))


def _is_same_frozen_policy(core: Dict[str, Any], burst: Dict[str, Any]) -> bool:
    core_boundary = bool(core.get("boundary_ok", False))
    burst_boundary = bool(burst.get("boundary_ok", False))
    core_stage1_trainable = int(
        (core.get("freeze_trainable_boundary", {}) if isinstance(core.get("freeze_trainable_boundary", {}), dict) else {}).get("stage1_trainable_parameter_count", 0)
        or 0
    )
    burst_stage1_trainable = int(
        (burst.get("freeze_trainable_boundary", {}) if isinstance(burst.get("freeze_trainable_boundary", {}), dict) else {}).get("stage1_trainable_parameter_count", 0)
        or 0
    )
    return bool(
        core_boundary
        and burst_boundary
        and core_stage1_trainable == 0
        and burst_stage1_trainable == 0
        and int(core.get("frozen_parameter_count", 0) or 0) == int(burst.get("frozen_parameter_count", 0) or 0)
    )


def _is_same_eval_protocol(core: Dict[str, Any], burst: Dict[str, Any]) -> bool:
    expected = _selection_policy()
    core_policy = core.get("comparison_sorting", {}) if isinstance(core.get("comparison_sorting", {}), dict) else {}
    burst_policy = burst.get("comparison_sorting", {}) if isinstance(burst.get("comparison_sorting", {}), dict) else {}
    return bool(core_policy == expected and burst_policy == expected)


def _why_burst_not_better(core: Dict[str, Any], burst: Dict[str, Any]) -> str:
    core_endpoint = _f(core.get("free_rollout_endpoint_l2"), 1e9)
    burst_endpoint = _f(burst.get("free_rollout_endpoint_l2"), 1e9)
    core_mean = _f(core.get("free_rollout_coord_mean_l2"), 1e9)
    burst_mean = _f(burst.get("free_rollout_coord_mean_l2"), 1e9)
    core_teacher = _f(core.get("teacher_forced_coord_loss"), 1e9)
    burst_teacher = _f(burst.get("teacher_forced_coord_loss"), 1e9)

    return (
        "burst is not better because endpoint/mean/teacher losses are not lower than core-only: "
        f"endpoint_delta(burst-core)={burst_endpoint - core_endpoint:.6f}, "
        f"coord_mean_delta(burst-core)={burst_mean - core_mean:.6f}, "
        f"teacher_delta(burst-core)={burst_teacher - core_teacher:.6f}"
    )


def build_eval_fix_comparison(core: Dict[str, Any], burst: Dict[str, Any], round_name: str) -> Dict[str, Any]:
    core_key = (
        _f(core.get("free_rollout_endpoint_l2"), 1e9),
        _f(core.get("free_rollout_coord_mean_l2"), 1e9),
        _f(core.get("teacher_forced_coord_loss"), 1e9),
    )
    burst_key = (
        _f(burst.get("free_rollout_endpoint_l2"), 1e9),
        _f(burst.get("free_rollout_coord_mean_l2"), 1e9),
        _f(burst.get("teacher_forced_coord_loss"), 1e9),
    )

    primary_winner = _winner(core_key[0], burst_key[0])
    secondary_winner = _winner(core_key[1], burst_key[1])
    tertiary_winner = _winner(core_key[2], burst_key[2])

    same_budget = _is_same_budget(core, burst)
    same_frozen_policy = _is_same_frozen_policy(core, burst)
    same_eval_protocol = _is_same_eval_protocol(core, burst)

    invalid_reasons: List[str] = []
    if not same_budget:
        invalid_reasons.append("budget_mismatch")
    if not same_frozen_policy:
        invalid_reasons.append("frozen_policy_mismatch")
    if not same_eval_protocol:
        invalid_reasons.append("eval_protocol_mismatch")

    invalid_comparison = bool(len(invalid_reasons) > 0)

    if invalid_comparison:
        final_recommended_mainline = MAINLINE_INVALID
    elif core_key <= burst_key:
        final_recommended_mainline = MAINLINE_CORE
    else:
        final_recommended_mainline = MAINLINE_BURST

    frozen_boundary_kept_correct = bool(core.get("boundary_ok", False) and burst.get("boundary_ok", False))
    core_better = bool(core_key < burst_key)
    burst_better = bool(burst_key < core_key)

    if invalid_comparison:
        can_continue_stage2_training = False
        next_step_choice = "fix_comparison_first"
    elif final_recommended_mainline == MAINLINE_CORE and frozen_boundary_kept_correct:
        can_continue_stage2_training = True
        next_step_choice = "continue_stage2_training_core_only"
    else:
        can_continue_stage2_training = False
        next_step_choice = "do_one_targeted_burst_fix"

    core_source = str(core.get("current_mainline_semantic_source", ""))
    burst_source = str(burst.get("current_mainline_semantic_source", ""))
    if core_source == burst_source:
        source_mode = core_source
    else:
        source_mode = "mixed_semantic_sources"

    why_burst_not_better = ""
    if final_recommended_mainline == MAINLINE_CORE and not invalid_comparison:
        why_burst_not_better = _why_burst_not_better(core, burst)

    return {
        "generated_at_utc": now_iso(),
        "round": str(round_name),
        "eval_fix_protocol_version": "stage2_eval_fix_20260408",
        "comparison_sorting": _selection_policy(),
        "current_mainline_semantic_source": source_mode,
        "legacy_semantic_source": str(core.get("legacy_semantic_source", "")),
        "crop_based_semantic_encoder_ran_through": bool(_crop_encoder_ran_through(core) and _crop_encoder_ran_through(burst)),
        "semantic_source_upgraded_from_handcrafted_to_crop_encoder": bool(
            source_mode == "crop_visual_encoder"
            and str(core.get("legacy_semantic_source", "")) == "hand_crafted_stats"
            and str(burst.get("legacy_semantic_source", "")) == "hand_crafted_stats"
        ),
        "datasets_bound_for_core": _run_datasets_bound(core),
        "datasets_bound_for_core_plus_burst": _run_datasets_bound(burst),
        "whether_same_budget": bool(same_budget),
        "whether_same_frozen_policy": bool(same_frozen_policy),
        "whether_same_eval_protocol": bool(same_eval_protocol),
        "invalid_comparison": bool(invalid_comparison),
        "invalid_reasons": invalid_reasons,
        "frozen_boundary_kept_correct": bool(frozen_boundary_kept_correct),
        "core_only_better_than_core_plus_burst": bool(core_better),
        "core_plus_burst_better_than_core_only": bool(burst_better),
        "primary_winner": primary_winner,
        "secondary_winner": secondary_winner,
        "tertiary_winner": tertiary_winner,
        "final_recommended_mainline": final_recommended_mainline,
        "why_burst_not_better": why_burst_not_better,
        "can_continue_stage2_training": bool(can_continue_stage2_training),
        "allowed_next_step_choice": [
            "continue_stage2_training_core_only",
            "do_one_targeted_burst_fix",
            "fix_comparison_first",
        ],
        "next_step_choice": str(next_step_choice),
        "core_final_metrics": {
            "teacher_forced_coord_loss": _f(core.get("teacher_forced_coord_loss"), 1e9),
            "free_rollout_coord_mean_l2": _f(core.get("free_rollout_coord_mean_l2"), 1e9),
            "free_rollout_endpoint_l2": _f(core.get("free_rollout_endpoint_l2"), 1e9),
            "total_loss_reference": _f(core.get("total_loss_reference"), 1e9),
        },
        "core_plus_burst_final_metrics": {
            "teacher_forced_coord_loss": _f(burst.get("teacher_forced_coord_loss"), 1e9),
            "free_rollout_coord_mean_l2": _f(burst.get("free_rollout_coord_mean_l2"), 1e9),
            "free_rollout_endpoint_l2": _f(burst.get("free_rollout_endpoint_l2"), 1e9),
            "total_loss_reference": _f(burst.get("total_loss_reference"), 1e9),
        },
        "core_best_checkpoint_metric": core.get("best_checkpoint_metric", {}),
        "core_latest_checkpoint_metric": core.get("latest_checkpoint_metric", {}),
        "core_plus_burst_best_checkpoint_metric": burst.get("best_checkpoint_metric", {}),
        "core_plus_burst_latest_checkpoint_metric": burst.get("latest_checkpoint_metric", {}),
        "core_checkpoint_dir": str((core.get("checkpoint_inventory", {}) if isinstance(core.get("checkpoint_inventory", {}), dict) else {}).get("checkpoint_dir", "")),
        "core_plus_burst_checkpoint_dir": str((burst.get("checkpoint_inventory", {}) if isinstance(burst.get("checkpoint_inventory", {}), dict) else {}).get("checkpoint_dir", "")),
    }


def write_results_markdown(path: str | Path, comparison: Dict[str, Any], title: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    core = comparison.get("core_final_metrics", {}) if isinstance(comparison.get("core_final_metrics", {}), dict) else {}
    burst = (
        comparison.get("core_plus_burst_final_metrics", {})
        if isinstance(comparison.get("core_plus_burst_final_metrics", {}), dict)
        else {}
    )

    lines = [
        f"# {title}",
        "",
        f"- generated_at_utc: {comparison.get('generated_at_utc', '')}",
        f"- current_mainline_semantic_source: {comparison.get('current_mainline_semantic_source', '')}",
        f"- core_only_better_than_core_plus_burst: {bool(comparison.get('core_only_better_than_core_plus_burst', False))}",
        f"- frozen_boundary_kept_correct: {bool(comparison.get('frozen_boundary_kept_correct', False))}",
        f"- final_recommended_mainline: {comparison.get('final_recommended_mainline', '')}",
        f"- can_continue_stage2_training: {bool(comparison.get('can_continue_stage2_training', False))}",
        f"- next_step_choice: {comparison.get('next_step_choice', '')}",
        f"- invalid_comparison: {bool(comparison.get('invalid_comparison', False))}",
        "",
        "## Sorting",
        f"- primary: {SORT_PRIMARY}",
        f"- secondary: {SORT_SECONDARY}",
        f"- tertiary: {SORT_TERTIARY}",
        f"- total_loss_usage: {SORT_TOTAL_LOSS_USAGE}",
        "",
        "## Comparability",
        f"- datasets_bound_for_core: {comparison.get('datasets_bound_for_core', [])}",
        f"- datasets_bound_for_core_plus_burst: {comparison.get('datasets_bound_for_core_plus_burst', [])}",
        f"- whether_same_budget: {bool(comparison.get('whether_same_budget', False))}",
        f"- whether_same_frozen_policy: {bool(comparison.get('whether_same_frozen_policy', False))}",
        f"- whether_same_eval_protocol: {bool(comparison.get('whether_same_eval_protocol', False))}",
        "",
        "## Winners",
        f"- primary_winner: {comparison.get('primary_winner', '')}",
        f"- secondary_winner: {comparison.get('secondary_winner', '')}",
        f"- tertiary_winner: {comparison.get('tertiary_winner', '')}",
        f"- why_burst_not_better: {comparison.get('why_burst_not_better', '')}",
        "",
        "## Core Metrics",
        f"- teacher_forced_coord_loss: {float(core.get('teacher_forced_coord_loss', 1e9)):.6f}",
        f"- free_rollout_coord_mean_l2: {float(core.get('free_rollout_coord_mean_l2', 1e9)):.6f}",
        f"- free_rollout_endpoint_l2: {float(core.get('free_rollout_endpoint_l2', 1e9)):.6f}",
        "",
        "## Core+Burst Metrics",
        f"- teacher_forced_coord_loss: {float(burst.get('teacher_forced_coord_loss', 1e9)):.6f}",
        f"- free_rollout_coord_mean_l2: {float(burst.get('free_rollout_coord_mean_l2', 1e9)):.6f}",
        f"- free_rollout_endpoint_l2: {float(burst.get('free_rollout_endpoint_l2', 1e9)):.6f}",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 eval protocol normalizer and comparison generator")
    sub = p.add_subparsers(dest="command", required=True)

    pn = sub.add_parser("normalize-run-summary", help="normalize a run summary to Stage2 eval-fix schema")
    pn.add_argument("--input-run-json", required=True)
    pn.add_argument("--output-run-json", required=True)
    pn.add_argument("--run-label", default="")

    pc = sub.add_parser("compare-runs", help="build strict comparison json and markdown")
    pc.add_argument("--core-run-json", required=True)
    pc.add_argument("--core-plus-burst-run-json", required=True)
    pc.add_argument("--comparison-json", required=True)
    pc.add_argument("--results-md", required=True)
    pc.add_argument("--round-name", default="stage2_eval_fix_20260408")
    pc.add_argument("--results-title", default="Stage2 Eval-Fix Results")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if str(args.command) == "normalize-run-summary":
        run = _read_json(args.input_run_json)
        normalized = build_eval_fix_run_summary(run, run_label=str(args.run_label))
        _write_json(args.output_run_json, normalized)
        print(f"[stage2-eval-fix] normalized_run_json={args.output_run_json}")
        print(f"[stage2-eval-fix] run_name={normalized.get('run_name', '')}")
        print(f"[stage2-eval-fix] boundary_ok={bool(normalized.get('boundary_ok', False))}")
        return

    if str(args.command) == "compare-runs":
        core = _read_json(args.core_run_json)
        burst = _read_json(args.core_plus_burst_run_json)

        # Accept either original trainer outputs or normalized eval-fix summaries.
        if str(core.get("eval_fix_protocol_version", "")) != "stage2_eval_fix_20260408":
            core = build_eval_fix_run_summary(core, run_label="stage2_smalltrain_cropenc_core")
        if str(burst.get("eval_fix_protocol_version", "")) != "stage2_eval_fix_20260408":
            burst = build_eval_fix_run_summary(burst, run_label="stage2_smalltrain_cropenc_core_plus_burst")

        comparison = build_eval_fix_comparison(core, burst, round_name=str(args.round_name))
        _write_json(args.comparison_json, comparison)
        write_results_markdown(args.results_md, comparison, title=str(args.results_title))

        print(f"[stage2-eval-fix] comparison_json={args.comparison_json}")
        print(f"[stage2-eval-fix] results_md={args.results_md}")
        print(f"[stage2-eval-fix] final_recommended_mainline={comparison.get('final_recommended_mainline', '')}")
        print(f"[stage2-eval-fix] can_continue_stage2_training={bool(comparison.get('can_continue_stage2_training', False))}")
        print(f"[stage2-eval-fix] next_step_choice={comparison.get('next_step_choice', '')}")
        return

    raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
