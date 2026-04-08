#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    fm = summary.get("final_metrics", {}) if isinstance(summary, dict) else {}
    tapvid = fm.get("tapvid", {}) if isinstance(fm, dict) else {}
    tapvid3d = fm.get("tapvid3d", {}) if isinstance(fm, dict) else {}

    val_teacher = float(fm.get("val_teacher_forced_loss", 0.0) or 0.0)
    val_free = float(fm.get("val_free_rollout_loss", 0.0) or 0.0)

    return {
        "val_teacher_forced_loss": val_teacher,
        "val_free_rollout_loss": val_free,
        "val_total_loss": float(fm.get("val_total_loss", 0.0) or 0.0),
        "tf_free_gap": val_free - val_teacher,
        "tapvid_free_endpoint_l2": float(tapvid.get("free_rollout_endpoint_l2", 0.0) or 0.0),
        "tapvid3d_limited_free_endpoint_l2": float(tapvid3d.get("free_rollout_endpoint_l2", 0.0) or 0.0),
    }


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Summarize Stage1 fix-round runs")
    p.add_argument("--diag-json", required=True)
    p.add_argument("--iter1-joint-summary", required=True)
    p.add_argument("--iter1-point-summary", required=True)
    p.add_argument("--iter1-kubric-summary", required=True)
    p.add_argument("--fix-balanced-summary", required=True)
    p.add_argument("--fix-lossnorm-summary", required=True)
    p.add_argument("--fix-sourcecond-summary", required=True)
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--results-md", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()

    diag = load_json(args.diag_json)
    iter1_joint = load_json(args.iter1_joint_summary)
    iter1_point = load_json(args.iter1_point_summary)
    iter1_kubric = load_json(args.iter1_kubric_summary)

    fix_bal = load_json(args.fix_balanced_summary)
    fix_loss = load_json(args.fix_lossnorm_summary)
    fix_src = load_json(args.fix_sourcecond_summary)

    baseline_single = {
        "pointodyssey_only": {
            "summary_path": str(Path(args.iter1_point_summary)),
            **extract_metrics(iter1_point),
            "checkpoint_best": str(iter1_point.get("checkpoint_best", "")),
        },
        "kubric_only": {
            "summary_path": str(Path(args.iter1_kubric_summary)),
            **extract_metrics(iter1_kubric),
            "checkpoint_best": str(iter1_kubric.get("checkpoint_best", "")),
        },
    }

    baseline_joint = {
        "summary_path": str(Path(args.iter1_joint_summary)),
        **extract_metrics(iter1_joint),
        "checkpoint_best": str(iter1_joint.get("checkpoint_best", "")),
    }

    fix_runs = {
        "tracewm_stage1_fix_joint_balanced_sampler": {
            "summary_path": str(Path(args.fix_balanced_summary)),
            **extract_metrics(fix_bal),
            "checkpoint_best": str(fix_bal.get("checkpoint_best", "")),
            "output_dir": str(Path(fix_bal.get("checkpoint_dir", "")).parent),
        },
        "tracewm_stage1_fix_joint_loss_normalized": {
            "summary_path": str(Path(args.fix_lossnorm_summary)),
            **extract_metrics(fix_loss),
            "checkpoint_best": str(fix_loss.get("checkpoint_best", "")),
            "output_dir": str(Path(fix_loss.get("checkpoint_dir", "")).parent),
        },
        "tracewm_stage1_fix_joint_source_conditioned": {
            "summary_path": str(Path(args.fix_sourcecond_summary)),
            **extract_metrics(fix_src),
            "checkpoint_best": str(fix_src.get("checkpoint_best", "")),
            "output_dir": str(Path(fix_src.get("checkpoint_dir", "")).parent),
        },
    }

    base_joint = baseline_joint

    # Effectiveness score relative to iter1 joint baseline.
    # Positive score means better than iter1 joint across tracked metrics.
    scored = {}
    for k, v in fix_runs.items():
        imp_val = (base_joint["val_total_loss"] - v["val_total_loss"]) / max(abs(base_joint["val_total_loss"]), 1e-12)
        imp_tap = (base_joint["tapvid_free_endpoint_l2"] - v["tapvid_free_endpoint_l2"]) / max(abs(base_joint["tapvid_free_endpoint_l2"]), 1e-12)
        imp_t3d = (base_joint["tapvid3d_limited_free_endpoint_l2"] - v["tapvid3d_limited_free_endpoint_l2"]) / max(abs(base_joint["tapvid3d_limited_free_endpoint_l2"]), 1e-12)
        score = imp_val + 0.5 * imp_tap + 0.5 * imp_t3d
        scored[k] = {
            "improvement_val_total": imp_val,
            "improvement_tapvid": imp_tap,
            "improvement_tapvid3d_limited": imp_t3d,
            "score": score,
        }

    most_effective_fix = max(scored.keys(), key=lambda k: scored[k]["score"])

    # Best single baseline from iter1 for strict surpass check.
    best_single_name = min(baseline_single.keys(), key=lambda k: baseline_single[k]["val_total_loss"])
    best_single = baseline_single[best_single_name]

    surpass_runs = []
    for k, v in fix_runs.items():
        if (
            v["val_total_loss"] <= best_single["val_total_loss"]
            and v["tapvid_free_endpoint_l2"] <= best_single["tapvid_free_endpoint_l2"]
            and v["tapvid3d_limited_free_endpoint_l2"] <= best_single["tapvid3d_limited_free_endpoint_l2"]
        ):
            surpass_runs.append(k)

    any_fix_surpasses_best_single = len(surpass_runs) > 0

    best_tapvid_fix = min(fix_runs.keys(), key=lambda k: fix_runs[k]["tapvid_free_endpoint_l2"])
    best_tapvid3d_fix = min(fix_runs.keys(), key=lambda k: fix_runs[k]["tapvid3d_limited_free_endpoint_l2"])

    base_gap_abs = abs(base_joint["tf_free_gap"])
    gap_improvement = {}
    for k, v in fix_runs.items():
        g = abs(v["tf_free_gap"])
        gap_improvement[k] = {
            "abs_gap": g,
            "improved_vs_iter1_joint": g < base_gap_abs,
        }

    if any_fix_surpasses_best_single:
        recommendation = "continue_stage1_trace_only_expand_training"
    elif scored[most_effective_fix]["score"] > 0:
        recommendation = "continue_stage1_model_fix"
    else:
        recommendation = "stop_joint_keep_best_single_for_stage2_prep"

    comparison = {
        "generated_at_utc": now_iso(),
        "round": "stage1_model_fix",
        "task": "trace_only_future_trace_state_generation",
        "diagnosis_json": str(Path(args.diag_json)),
        "baseline_single": baseline_single,
        "baseline_joint_iter1": baseline_joint,
        "fix_runs": fix_runs,
        "effectiveness_scores": scored,
        "answers": {
            "q1_most_effective_fix": {
                "winner": most_effective_fix,
                "winner_score": scored[most_effective_fix],
            },
            "q2_any_fix_surpasses_best_single": {
                "best_single_name": best_single_name,
                "best_single_metrics": best_single,
                "any_fix_surpasses": any_fix_surpasses_best_single,
                "surpass_runs": surpass_runs,
            },
            "q3_best_for_tapvid": {
                "winner": best_tapvid_fix,
                "tapvid_free_endpoint_l2": {k: v["tapvid_free_endpoint_l2"] for k, v in fix_runs.items()},
            },
            "q4_best_for_tapvid3d_limited": {
                "winner": best_tapvid3d_fix,
                "tapvid3d_limited_free_endpoint_l2": {k: v["tapvid3d_limited_free_endpoint_l2"] for k, v in fix_runs.items()},
            },
            "q5_tf_free_gap_improvement": {
                "iter1_joint_abs_gap": base_gap_abs,
                "fix_abs_gap": gap_improvement,
            },
            "q6_next_recommendation": {
                "recommendation": recommendation,
                "allowed_choices": [
                    "continue_stage1_trace_only_expand_training",
                    "continue_stage1_model_fix",
                    "stop_joint_keep_best_single_for_stage2_prep",
                ],
            },
        },
        "next_step_choice": recommendation,
    }

    out_json = Path(args.comparison_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# TraceWM Stage 1 Fix Results (2026-04-08)",
        "",
        f"- generated_at_utc: {comparison['generated_at_utc']}",
        f"- diagnosis_json: {comparison['diagnosis_json']}",
        f"- comparison_json: {out_json}",
        "",
        "## Fix Run Metrics",
        "",
        "| run | val_total_loss | tf_free_gap | tapvid_free_endpoint_l2 | tapvid3d_limited_free_endpoint_l2 | effectiveness_score |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in [
        "tracewm_stage1_fix_joint_balanced_sampler",
        "tracewm_stage1_fix_joint_loss_normalized",
        "tracewm_stage1_fix_joint_source_conditioned",
    ]:
        r = fix_runs[name]
        s = scored[name]
        md_lines.append(
            f"| {name} | {r['val_total_loss']:.6f} | {r['tf_free_gap']:.6f} | {r['tapvid_free_endpoint_l2']:.6f} | {r['tapvid3d_limited_free_endpoint_l2']:.6f} | {s['score']:.6f} |"
        )

    md_lines.extend([
        "",
        "## Required Answers",
        "",
        f"1. Most effective fix: {comparison['answers']['q1_most_effective_fix']['winner']}",
        f"2. Any fix surpasses best single: {comparison['answers']['q2_any_fix_surpasses_best_single']['any_fix_surpasses']}",
        f"3. Best for TAP-Vid: {comparison['answers']['q3_best_for_tapvid']['winner']}",
        f"4. Best for TAPVid-3D limited: {comparison['answers']['q4_best_for_tapvid3d_limited']['winner']}",
        f"5. TF/free gap improvement: {comparison['answers']['q5_tf_free_gap_improvement']['fix_abs_gap']}",
        f"6. Next recommendation: {comparison['answers']['q6_next_recommendation']['recommendation']}",
    ])

    out_md = Path(args.results_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[fix-summary] wrote comparison: {out_json}")
    print(f"[fix-summary] wrote results doc: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
