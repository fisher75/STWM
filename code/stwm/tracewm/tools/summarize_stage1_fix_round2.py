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


def strict_surpass_best_single(run_metrics: Dict[str, float], best_single: Dict[str, float]) -> bool:
    return bool(
        run_metrics["val_total_loss"] <= best_single["val_total_loss"]
        and run_metrics["tapvid_free_endpoint_l2"] <= best_single["tapvid_free_endpoint_l2"]
        and run_metrics["tapvid3d_limited_free_endpoint_l2"] <= best_single["tapvid3d_limited_free_endpoint_l2"]
    )


def score_against_reference(run_metrics: Dict[str, float], ref_metrics: Dict[str, float]) -> Dict[str, float]:
    imp_val = (ref_metrics["val_total_loss"] - run_metrics["val_total_loss"]) / max(abs(ref_metrics["val_total_loss"]), 1e-12)
    imp_tap = (ref_metrics["tapvid_free_endpoint_l2"] - run_metrics["tapvid_free_endpoint_l2"]) / max(abs(ref_metrics["tapvid_free_endpoint_l2"]), 1e-12)
    imp_t3d = (ref_metrics["tapvid3d_limited_free_endpoint_l2"] - run_metrics["tapvid3d_limited_free_endpoint_l2"]) / max(
        abs(ref_metrics["tapvid3d_limited_free_endpoint_l2"]),
        1e-12,
    )
    score = imp_val + 0.5 * imp_tap + 0.5 * imp_t3d
    return {
        "improvement_val_total": imp_val,
        "improvement_tapvid": imp_tap,
        "improvement_tapvid3d_limited": imp_t3d,
        "score": score,
    }


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Summarize Stage1 model-fix round2 runs")
    p.add_argument("--round2-doc", required=True)
    p.add_argument("--iter1-point-summary", required=True)
    p.add_argument("--iter1-kubric-summary", required=True)
    p.add_argument("--run-nowarmup-summary", required=True)
    p.add_argument("--run-point-warmup-summary", required=True)
    p.add_argument("--run-kubric-warmup-summary", required=True)
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--results-md", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()

    round2_doc = str(Path(args.round2_doc))
    iter1_point = load_json(args.iter1_point_summary)
    iter1_kubric = load_json(args.iter1_kubric_summary)

    run_nowarmup = load_json(args.run_nowarmup_summary)
    run_point_warmup = load_json(args.run_point_warmup_summary)
    run_kubric_warmup = load_json(args.run_kubric_warmup_summary)

    baseline_single = {
        "tracewm_stage1_iter1_pointodyssey_only": {
            "summary_path": str(Path(args.iter1_point_summary)),
            **extract_metrics(iter1_point),
            "checkpoint_best": str(iter1_point.get("checkpoint_best", "")),
        },
        "tracewm_stage1_iter1_kubric_only": {
            "summary_path": str(Path(args.iter1_kubric_summary)),
            **extract_metrics(iter1_kubric),
            "checkpoint_best": str(iter1_kubric.get("checkpoint_best", "")),
        },
    }

    fix2_runs = {
        "tracewm_stage1_fix2_joint_balanced_lossnorm": {
            "summary_path": str(Path(args.run_nowarmup_summary)),
            **extract_metrics(run_nowarmup),
            "checkpoint_best": str(run_nowarmup.get("checkpoint_best", "")),
            "checkpoint_latest": str(run_nowarmup.get("checkpoint_latest", "")),
            "output_dir": str(Path(run_nowarmup.get("checkpoint_dir", "")).parent),
        },
        "tracewm_stage1_fix2_point_warmup_then_joint_balanced_lossnorm": {
            "summary_path": str(Path(args.run_point_warmup_summary)),
            **extract_metrics(run_point_warmup),
            "checkpoint_best": str(run_point_warmup.get("checkpoint_best", "")),
            "checkpoint_latest": str(run_point_warmup.get("checkpoint_latest", "")),
            "output_dir": str(Path(run_point_warmup.get("checkpoint_dir", "")).parent),
        },
        "tracewm_stage1_fix2_kubric_warmup_then_joint_balanced_lossnorm": {
            "summary_path": str(Path(args.run_kubric_warmup_summary)),
            **extract_metrics(run_kubric_warmup),
            "checkpoint_best": str(run_kubric_warmup.get("checkpoint_best", "")),
            "checkpoint_latest": str(run_kubric_warmup.get("checkpoint_latest", "")),
            "output_dir": str(Path(run_kubric_warmup.get("checkpoint_dir", "")).parent),
        },
    }

    best_single_name = min(baseline_single.keys(), key=lambda k: baseline_single[k]["val_total_loss"])
    best_single_metrics = baseline_single[best_single_name]

    no_warmup_key = "tracewm_stage1_fix2_joint_balanced_lossnorm"
    point_warmup_key = "tracewm_stage1_fix2_point_warmup_then_joint_balanced_lossnorm"
    kubric_warmup_key = "tracewm_stage1_fix2_kubric_warmup_then_joint_balanced_lossnorm"

    q1_surpass = strict_surpass_best_single(fix2_runs[no_warmup_key], best_single_metrics)

    score_vs_no_warmup = {
        point_warmup_key: score_against_reference(fix2_runs[point_warmup_key], fix2_runs[no_warmup_key]),
        kubric_warmup_key: score_against_reference(fix2_runs[kubric_warmup_key], fix2_runs[no_warmup_key]),
    }

    q2_point_better = bool(score_vs_no_warmup[point_warmup_key]["score"] > 0.0)
    q3_kubric_better = bool(score_vs_no_warmup[kubric_warmup_key]["score"] > 0.0)

    score_vs_best_single = {
        k: score_against_reference(v, best_single_metrics)
        for k, v in fix2_runs.items()
    }
    best_joint_recipe = max(score_vs_best_single.keys(), key=lambda k: score_vs_best_single[k]["score"])

    surpass_runs = [k for k, v in fix2_runs.items() if strict_surpass_best_single(v, best_single_metrics)]
    any_joint_surpasses_best_single = len(surpass_runs) > 0

    if any_joint_surpasses_best_single:
        q6_no_surpass_recommendation = None
        q7_has_surpass_recommendation = "promote_joint_as_stage1_mainline"
        final_recommendation = "promote_joint_as_stage1_mainline"
    else:
        q7_has_surpass_recommendation = None
        if q2_point_better or q3_kubric_better:
            q6_no_surpass_recommendation = "continue_joint_recipe_fix"
        else:
            q6_no_surpass_recommendation = "stop_joint_and_keep_best_single"
        final_recommendation = q6_no_surpass_recommendation

    comparison = {
        "generated_at_utc": now_iso(),
        "round": "stage1_model_fix_round2",
        "task": "trace_only_future_trace_state_generation",
        "round2_doc": round2_doc,
        "baseline_single_reused": baseline_single,
        "best_single": {
            "name": best_single_name,
            "metrics": best_single_metrics,
        },
        "fix2_runs": fix2_runs,
        "scores": {
            "vs_best_single": score_vs_best_single,
            "warmup_vs_no_warmup": score_vs_no_warmup,
        },
        "answers": {
            "q1_balanced_lossnorm_surpasses_best_single": {
                "run": no_warmup_key,
                "surpasses": q1_surpass,
                "run_metrics": fix2_runs[no_warmup_key],
                "best_single_name": best_single_name,
            },
            "q2_point_warmup_better_than_no_warmup": {
                "better": q2_point_better,
                "strict_all_metrics_better": bool(
                    fix2_runs[point_warmup_key]["val_total_loss"] <= fix2_runs[no_warmup_key]["val_total_loss"]
                    and fix2_runs[point_warmup_key]["tapvid_free_endpoint_l2"] <= fix2_runs[no_warmup_key]["tapvid_free_endpoint_l2"]
                    and fix2_runs[point_warmup_key]["tapvid3d_limited_free_endpoint_l2"] <= fix2_runs[no_warmup_key]["tapvid3d_limited_free_endpoint_l2"]
                ),
                "score_delta_vs_no_warmup": score_vs_no_warmup[point_warmup_key],
            },
            "q3_kubric_warmup_better_than_no_warmup": {
                "better": q3_kubric_better,
                "strict_all_metrics_better": bool(
                    fix2_runs[kubric_warmup_key]["val_total_loss"] <= fix2_runs[no_warmup_key]["val_total_loss"]
                    and fix2_runs[kubric_warmup_key]["tapvid_free_endpoint_l2"] <= fix2_runs[no_warmup_key]["tapvid_free_endpoint_l2"]
                    and fix2_runs[kubric_warmup_key]["tapvid3d_limited_free_endpoint_l2"] <= fix2_runs[no_warmup_key]["tapvid3d_limited_free_endpoint_l2"]
                ),
                "score_delta_vs_no_warmup": score_vs_no_warmup[kubric_warmup_key],
            },
            "q4_best_joint_recipe": {
                "winner": best_joint_recipe,
                "winner_score": score_vs_best_single[best_joint_recipe],
            },
            "q5_any_joint_surpasses_best_single": {
                "value": any_joint_surpasses_best_single,
                "surpass_runs": surpass_runs,
            },
            "q6_if_no_surpass_single_recommendation": {
                "active": not any_joint_surpasses_best_single,
                "recommendation": q6_no_surpass_recommendation,
                "allowed_choices": [
                    "stop_joint_and_keep_best_single",
                    "continue_joint_recipe_fix",
                ],
            },
            "q7_if_has_surpass_single_recommendation": {
                "active": any_joint_surpasses_best_single,
                "recommendation": q7_has_surpass_recommendation,
                "allowed_choices": [
                    "promote_joint_as_stage1_mainline",
                ],
            },
        },
        "final_recommendation": final_recommendation,
        "next_step_choice": final_recommendation,
    }

    out_json = Path(args.comparison_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# TraceWM Stage 1 Fix Round 2 Results (2026-04-08)",
        "",
        f"- generated_at_utc: {comparison['generated_at_utc']}",
        f"- round2_doc: {comparison['round2_doc']}",
        f"- comparison_json: {out_json}",
        "",
        "## Run Metrics",
        "",
        "| run | val_total_loss | tf_free_gap | tapvid_free_endpoint_l2 | tapvid3d_limited_free_endpoint_l2 | score_vs_best_single |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in [
        no_warmup_key,
        point_warmup_key,
        kubric_warmup_key,
    ]:
        r = fix2_runs[name]
        s = score_vs_best_single[name]["score"]
        md_lines.append(
            f"| {name} | {r['val_total_loss']:.6f} | {r['tf_free_gap']:.6f} | {r['tapvid_free_endpoint_l2']:.6f} | {r['tapvid3d_limited_free_endpoint_l2']:.6f} | {s:.6f} |"
        )

    md_lines.extend([
        "",
        "## Required Answers",
        "",
        f"1. balanced+lossnorm itself surpasses best single: {comparison['answers']['q1_balanced_lossnorm_surpasses_best_single']['surpasses']}",
        f"2. point warmup better than no-warmup: {comparison['answers']['q2_point_warmup_better_than_no_warmup']['better']}",
        f"3. kubric warmup better than no-warmup: {comparison['answers']['q3_kubric_warmup_better_than_no_warmup']['better']}",
        f"4. best joint recipe among three: {comparison['answers']['q4_best_joint_recipe']['winner']}",
        f"5. any_joint_surpasses_best_single: {comparison['answers']['q5_any_joint_surpasses_best_single']['value']}",
        f"6. if no surpass recommendation: {comparison['answers']['q6_if_no_surpass_single_recommendation']['recommendation']}",
        f"7. if has surpass recommendation: {comparison['answers']['q7_if_has_surpass_single_recommendation']['recommendation']}",
        "",
        f"- final_recommendation: {comparison['final_recommendation']}",
    ])

    out_md = Path(args.results_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[fix2-summary] wrote comparison: {out_json}")
    print(f"[fix2-summary] wrote results doc: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
