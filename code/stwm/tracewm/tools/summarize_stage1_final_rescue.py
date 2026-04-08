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


def extract_metrics(summary: Dict[str, Any]) -> Dict[str, Any]:
    fm = summary.get("final_metrics", {}) if isinstance(summary, dict) else {}
    tapvid = fm.get("tapvid", {}) if isinstance(fm, dict) else {}
    tapvid3d = fm.get("tapvid3d", {}) if isinstance(fm, dict) else {}
    model_info = summary.get("model_info", {}) if isinstance(summary, dict) else {}
    method_flags = summary.get("method_flags", {}) if isinstance(summary, dict) else {}

    val_teacher = float(fm.get("val_teacher_forced_loss", 0.0) or 0.0)
    val_free = float(fm.get("val_free_rollout_loss", 0.0) or 0.0)

    return {
        "val_teacher_forced_loss": val_teacher,
        "val_free_rollout_loss": val_free,
        "val_total_loss": float(fm.get("val_total_loss", 0.0) or 0.0),
        "tf_free_gap": val_free - val_teacher,
        "tapvid_free_endpoint_l2": float(tapvid.get("free_rollout_endpoint_l2", 0.0) or 0.0),
        "tapvid3d_limited_free_endpoint_l2": float(tapvid3d.get("free_rollout_endpoint_l2", 0.0) or 0.0),
        "train_mode": str(summary.get("train_mode", "")),
        "private_parameter_count": int(model_info.get("private_parameter_count", 0) or 0),
        "shared_parameter_count": int(model_info.get("shared_parameter_count", 0) or 0),
        "method_flags": method_flags,
    }


def strict_surpass_best_single(run_metrics: Dict[str, Any], best_single: Dict[str, Any]) -> bool:
    return bool(
        run_metrics["val_total_loss"] <= best_single["val_total_loss"]
        and run_metrics["tapvid_free_endpoint_l2"] <= best_single["tapvid_free_endpoint_l2"]
        and run_metrics["tapvid3d_limited_free_endpoint_l2"] <= best_single["tapvid3d_limited_free_endpoint_l2"]
    )


def score_against_reference(run_metrics: Dict[str, Any], ref_metrics: Dict[str, Any]) -> Dict[str, float]:
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
    p = ArgumentParser(description="Summarize Stage1 final joint rescue runs")
    p.add_argument("--freeze-doc", required=True)
    p.add_argument("--iter1-point-summary", required=True)
    p.add_argument("--iter1-kubric-summary", required=True)
    p.add_argument("--fix2-best-joint-summary", required=True)
    p.add_argument("--pcgrad-summary", required=True)
    p.add_argument("--gradnorm-summary", required=True)
    p.add_argument("--shared-private-summary", required=True)
    p.add_argument("--shared-private-plus-best-grad-summary", required=True)
    p.add_argument("--selected-best-gradient-method", required=True, choices=["pcgrad", "gradnorm"])
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--results-md", required=True)
    return p


def main() -> int:
    args = build_parser().parse_args()

    freeze_doc = str(Path(args.freeze_doc))

    iter1_point = load_json(args.iter1_point_summary)
    iter1_kubric = load_json(args.iter1_kubric_summary)
    fix2_best_joint = load_json(args.fix2_best_joint_summary)

    run_pcgrad = load_json(args.pcgrad_summary)
    run_gradnorm = load_json(args.gradnorm_summary)
    run_shared_private = load_json(args.shared_private_summary)
    run_combo = load_json(args.shared_private_plus_best_grad_summary)

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

    best_single_name = min(baseline_single.keys(), key=lambda k: baseline_single[k]["val_total_loss"])
    best_single_metrics = baseline_single[best_single_name]

    best_current_joint = {
        "name": "tracewm_stage1_fix2_joint_balanced_lossnorm",
        "summary_path": str(Path(args.fix2_best_joint_summary)),
        **extract_metrics(fix2_best_joint),
        "checkpoint_best": str(fix2_best_joint.get("checkpoint_best", "")),
        "checkpoint_latest": str(fix2_best_joint.get("checkpoint_latest", "")),
    }

    runs = {
        "tracewm_stage1_rescue_pcgrad": {
            "summary_path": str(Path(args.pcgrad_summary)),
            **extract_metrics(run_pcgrad),
            "checkpoint_best": str(run_pcgrad.get("checkpoint_best", "")),
            "checkpoint_latest": str(run_pcgrad.get("checkpoint_latest", "")),
            "output_dir": str(Path(run_pcgrad.get("checkpoint_dir", "")).parent),
        },
        "tracewm_stage1_rescue_gradnorm": {
            "summary_path": str(Path(args.gradnorm_summary)),
            **extract_metrics(run_gradnorm),
            "checkpoint_best": str(run_gradnorm.get("checkpoint_best", "")),
            "checkpoint_latest": str(run_gradnorm.get("checkpoint_latest", "")),
            "output_dir": str(Path(run_gradnorm.get("checkpoint_dir", "")).parent),
        },
        "tracewm_stage1_rescue_shared_private": {
            "summary_path": str(Path(args.shared_private_summary)),
            **extract_metrics(run_shared_private),
            "checkpoint_best": str(run_shared_private.get("checkpoint_best", "")),
            "checkpoint_latest": str(run_shared_private.get("checkpoint_latest", "")),
            "output_dir": str(Path(run_shared_private.get("checkpoint_dir", "")).parent),
        },
        "tracewm_stage1_rescue_shared_private_plus_best_grad": {
            "summary_path": str(Path(args.shared_private_plus_best_grad_summary)),
            **extract_metrics(run_combo),
            "checkpoint_best": str(run_combo.get("checkpoint_best", "")),
            "checkpoint_latest": str(run_combo.get("checkpoint_latest", "")),
            "output_dir": str(Path(run_combo.get("checkpoint_dir", "")).parent),
        },
    }

    scores_vs_best_single = {k: score_against_reference(v, best_single_metrics) for k, v in runs.items()}
    scores_vs_best_current_joint = {k: score_against_reference(v, best_current_joint) for k, v in runs.items()}

    q1_best_rescue_run = max(scores_vs_best_single.keys(), key=lambda k: scores_vs_best_single[k]["score"])

    surpass_runs = [k for k, v in runs.items() if strict_surpass_best_single(v, best_single_metrics)]
    q2_any_surpass = len(surpass_runs) > 0

    q2_gap = None
    if not q2_any_surpass:
        best_run_metrics = runs[q1_best_rescue_run]
        q2_gap = {
            "best_rescue_run": q1_best_rescue_run,
            "val_total_loss_delta": float(best_run_metrics["val_total_loss"] - best_single_metrics["val_total_loss"]),
            "tapvid_free_endpoint_l2_delta": float(best_run_metrics["tapvid_free_endpoint_l2"] - best_single_metrics["tapvid_free_endpoint_l2"]),
            "tapvid3d_limited_free_endpoint_l2_delta": float(
                best_run_metrics["tapvid3d_limited_free_endpoint_l2"] - best_single_metrics["tapvid3d_limited_free_endpoint_l2"]
            ),
        }

    pc_tap = runs["tracewm_stage1_rescue_pcgrad"]["tapvid_free_endpoint_l2"]
    gn_tap = runs["tracewm_stage1_rescue_gradnorm"]["tapvid_free_endpoint_l2"]
    if abs(pc_tap - gn_tap) <= 1e-12:
        q3_best_gradient_method = "none"
    elif pc_tap < gn_tap:
        q3_best_gradient_method = "pcgrad"
    else:
        q3_best_gradient_method = "gradnorm"

    sp_key = "tracewm_stage1_rescue_shared_private"
    combo_key = "tracewm_stage1_rescue_shared_private_plus_best_grad"
    q4_shared_private_helpful = bool(scores_vs_best_current_joint[sp_key]["score"] > 0.0)

    standalone_keys = [
        "tracewm_stage1_rescue_pcgrad",
        "tracewm_stage1_rescue_gradnorm",
        "tracewm_stage1_rescue_shared_private",
    ]
    best_standalone_key = max(standalone_keys, key=lambda k: scores_vs_best_current_joint[k]["score"])
    combo_vs_best_standalone = score_against_reference(runs[combo_key], runs[best_standalone_key])
    q5_combo_helpful = bool(combo_vs_best_standalone["score"] > 0.0)

    q6_best_on_tapvid = min(runs.keys(), key=lambda k: runs[k]["tapvid_free_endpoint_l2"])
    q7_best_on_tapvid3d_limited = min(runs.keys(), key=lambda k: runs[k]["tapvid3d_limited_free_endpoint_l2"])

    if q2_any_surpass:
        if q1_best_rescue_run == combo_key:
            q8_final_joint_decision = "promote_joint_as_stage1_mainline"
        else:
            q8_final_joint_decision = "keep_joint_as_secondary_variant"
    else:
        q8_final_joint_decision = "stop_joint_and_keep_best_single"

    next_step_choice = q8_final_joint_decision

    comparison = {
        "generated_at_utc": now_iso(),
        "round": "stage1_final_joint_rescue",
        "task": "trace_only_future_trace_state_generation",
        "freeze_doc": freeze_doc,
        "baseline_single_reused": baseline_single,
        "best_single": {
            "name": best_single_name,
            "metrics": best_single_metrics,
        },
        "best_current_joint_reused": best_current_joint,
        "selected_best_gradient_method_for_combo": args.selected_best_gradient_method,
        "rescue_runs": runs,
        "scores": {
            "vs_best_single": scores_vs_best_single,
            "vs_best_current_joint": scores_vs_best_current_joint,
            "combo_vs_best_standalone": {
                "best_standalone_run": best_standalone_key,
                "score": combo_vs_best_standalone,
            },
        },
        "answers": {
            "q1_best_rescue_run": {
                "winner": q1_best_rescue_run,
                "winner_score": scores_vs_best_single[q1_best_rescue_run],
            },
            "q2_any_rescue_surpasses_best_single": {
                "value": q2_any_surpass,
                "surpass_runs": surpass_runs,
                "gap_if_false": q2_gap,
            },
            "q3_best_gradient_method": {
                "value": q3_best_gradient_method,
                "tapvid_free_endpoint_l2": {
                    "pcgrad": pc_tap,
                    "gradnorm": gn_tap,
                },
            },
            "q4_shared_private_helpful": {
                "value": q4_shared_private_helpful,
                "score_vs_best_current_joint": scores_vs_best_current_joint[sp_key],
            },
            "q5_shared_private_plus_best_grad_helpful": {
                "value": q5_combo_helpful,
                "best_standalone_run": best_standalone_key,
                "score_vs_best_standalone": combo_vs_best_standalone,
            },
            "q6_best_on_tapvid": {
                "winner": q6_best_on_tapvid,
                "tapvid_free_endpoint_l2": {k: v["tapvid_free_endpoint_l2"] for k, v in runs.items()},
            },
            "q7_best_on_tapvid3d_limited": {
                "winner": q7_best_on_tapvid3d_limited,
                "tapvid3d_limited_free_endpoint_l2": {k: v["tapvid3d_limited_free_endpoint_l2"] for k, v in runs.items()},
            },
            "q8_final_joint_decision": q8_final_joint_decision,
        },
        "next_step_choice": next_step_choice,
    }

    out_json = Path(args.comparison_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# TraceWM Stage 1 Final Rescue Results (2026-04-08)",
        "",
        f"- generated_at_utc: {comparison['generated_at_utc']}",
        f"- freeze_doc: {comparison['freeze_doc']}",
        f"- selected_best_gradient_method_for_combo: {comparison['selected_best_gradient_method_for_combo']}",
        f"- comparison_json: {out_json}",
        "",
        "## Run Metrics",
        "",
        "| run | train_mode | val_total_loss | tapvid_free_endpoint_l2 | tapvid3d_limited_free_endpoint_l2 | private_parameter_count | score_vs_best_single |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for name in [
        "tracewm_stage1_rescue_pcgrad",
        "tracewm_stage1_rescue_gradnorm",
        "tracewm_stage1_rescue_shared_private",
        "tracewm_stage1_rescue_shared_private_plus_best_grad",
    ]:
        r = runs[name]
        s = scores_vs_best_single[name]["score"]
        md_lines.append(
            f"| {name} | {r['train_mode']} | {r['val_total_loss']:.6f} | {r['tapvid_free_endpoint_l2']:.6f} | {r['tapvid3d_limited_free_endpoint_l2']:.6f} | {r['private_parameter_count']} | {s:.6f} |"
        )

    md_lines.extend([
        "",
        "## Required Answers",
        "",
        f"1. q1_best_rescue_run: {comparison['answers']['q1_best_rescue_run']['winner']}",
        f"2. q2_any_rescue_surpasses_best_single: {comparison['answers']['q2_any_rescue_surpasses_best_single']['value']}",
        f"3. q3_best_gradient_method: {comparison['answers']['q3_best_gradient_method']['value']}",
        f"4. q4_shared_private_helpful: {comparison['answers']['q4_shared_private_helpful']['value']}",
        f"5. q5_shared_private_plus_best_grad_helpful: {comparison['answers']['q5_shared_private_plus_best_grad_helpful']['value']}",
        f"6. q6_best_on_tapvid: {comparison['answers']['q6_best_on_tapvid']['winner']}",
        f"7. q7_best_on_tapvid3d_limited: {comparison['answers']['q7_best_on_tapvid3d_limited']['winner']}",
        f"8. q8_final_joint_decision: {comparison['answers']['q8_final_joint_decision']}",
        f"next_step_choice: {comparison['next_step_choice']}",
    ])

    out_md = Path(args.results_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[final_rescue_summary] wrote comparison: {out_json}")
    print(f"[final_rescue_summary] wrote results doc: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
