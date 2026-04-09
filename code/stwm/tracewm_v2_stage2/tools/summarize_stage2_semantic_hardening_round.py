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
    p = ArgumentParser(description="Summarize Stage2 semantic hardening core vs core+burst runs")
    p.add_argument("--core-run-json", required=True)
    p.add_argument("--core-plus-burst-run-json", required=True)
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


def _source_fields(run: Dict[str, Any]) -> Dict[str, str]:
    return {
        "current_mainline_semantic_source": str(run.get("current_mainline_semantic_source", "")),
        "legacy_semantic_source": str(run.get("legacy_semantic_source", "")),
    }


def _crop_encoder_ran(run: Dict[str, Any]) -> bool:
    src = _source_fields(run)
    m = _metrics(run)
    sb = m.get("semantic_branch_metrics", {}) if isinstance(m.get("semantic_branch_metrics", {}), dict) else {}
    nonempty = _f(sb.get("semantic_input_nonempty_ratio"), 0.0)
    return bool(src["current_mainline_semantic_source"] == "crop_visual_encoder" and nonempty > 0.0)


def _tap_eval_missing(run: Dict[str, Any]) -> bool:
    m = _metrics(run)
    tap = m.get("tapvid_style_eval", {}) if isinstance(m.get("tapvid_style_eval", {}), dict) else {}
    tap3d = m.get("tapvid3d_limited_eval", {}) if isinstance(m.get("tapvid3d_limited_eval", {}), dict) else {}
    return bool((not bool(tap.get("compatible", False))) or (not bool(tap3d.get("compatible", False))))


def _write_md(path: Path, comparison: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    core = comparison.get("core_final_metrics", {}) if isinstance(comparison.get("core_final_metrics", {}), dict) else {}
    burst = comparison.get("core_plus_burst_final_metrics", {}) if isinstance(comparison.get("core_plus_burst_final_metrics", {}), dict) else {}

    lines = [
        "# Stage2 Semantic Hardening Results",
        "",
        f"- generated_at_utc: {comparison.get('generated_at_utc', '')}",
        f"- crop_based_semantic_encoder_ran_through: {bool(comparison.get('crop_based_semantic_encoder_ran_through', False))}",
        f"- frozen_boundary_kept_correct: {bool(comparison.get('frozen_boundary_kept_correct', False))}",
        f"- core_plus_burst_better_than_core_only: {bool(comparison.get('core_plus_burst_better_than_core_only', False))}",
        f"- current_mainline_semantic_source = {comparison.get('current_mainline_semantic_source', '')}",
        f"- legacy_semantic_source = {comparison.get('legacy_semantic_source', '')}",
        f"- next_step_choice: {comparison.get('next_step_choice', '')}",
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    core = _read_json(args.core_run_json)
    burst = _read_json(args.core_plus_burst_run_json)

    core_src = _source_fields(core)
    burst_src = _source_fields(burst)

    crop_encoder_ran = bool(_crop_encoder_ran(core) and _crop_encoder_ran(burst))
    boundary_ok = bool(_boundary_ok(core) and _boundary_ok(burst))
    burst_better = bool(_better_than(burst, core))
    core_stable = bool(_run_stable(core))
    source_upgraded = bool(
        core_src["current_mainline_semantic_source"] == "crop_visual_encoder"
        and burst_src["current_mainline_semantic_source"] == "crop_visual_encoder"
        and core_src["legacy_semantic_source"] == "hand_crafted_stats"
        and burst_src["legacy_semantic_source"] == "hand_crafted_stats"
    )

    if (not crop_encoder_ran) or (not boundary_ok) or (not source_upgraded) or (not core_stable):
        next_step_choice = "refine_stage2_semantic_source"
    elif _tap_eval_missing(core) or _tap_eval_missing(burst):
        next_step_choice = "do_one_more_stage2_eval_fix"
    else:
        next_step_choice = "continue_stage2_training"

    comparison = {
        "generated_at_utc": now_iso(),
        "round": "stage2_semantic_hardening_20260408",
        "crop_based_semantic_encoder_ran_through": bool(crop_encoder_ran),
        "frozen_boundary_kept_correct": bool(boundary_ok),
        "core_plus_burst_better_than_core_only": bool(burst_better),
        "semantic_source_upgraded_from_handcrafted_to_crop_encoder": bool(source_upgraded),
        "current_mainline_semantic_source": "crop_visual_encoder",
        "legacy_semantic_source": "hand_crafted_stats",
        "allowed_next_step_choice": [
            "continue_stage2_training",
            "do_one_more_stage2_eval_fix",
            "refine_stage2_semantic_source",
        ],
        "next_step_choice": str(next_step_choice),
        "core_final_metrics": _metrics(core),
        "core_plus_burst_final_metrics": _metrics(burst),
        "core_checkpoint_dir": str((core.get("checkpoint_inventory", {}) if isinstance(core.get("checkpoint_inventory", {}), dict) else {}).get("checkpoint_dir", "")),
        "core_plus_burst_checkpoint_dir": str((burst.get("checkpoint_inventory", {}) if isinstance(burst.get("checkpoint_inventory", {}), dict) else {}).get("checkpoint_dir", "")),
    }

    comparison_json = Path(args.comparison_json)
    results_md = Path(args.results_md)
    _write_json(comparison_json, comparison)
    _write_md(results_md, comparison)

    print(f"[stage2-semhard-summary] comparison_json={comparison_json}")
    print(f"[stage2-semhard-summary] results_md={results_md}")
    print(f"[stage2-semhard-summary] crop_based_semantic_encoder_ran_through={comparison['crop_based_semantic_encoder_ran_through']}")
    print(f"[stage2-semhard-summary] next_step_choice={comparison['next_step_choice']}")


if __name__ == "__main__":
    main()