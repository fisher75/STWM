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


def final_metrics(summary: Dict[str, Any]) -> Dict[str, float]:
    fm = summary.get("final_metrics", {}) if isinstance(summary, dict) else {}
    tapvid = fm.get("tapvid", {}) if isinstance(fm, dict) else {}
    tapvid3d = fm.get("tapvid3d", {}) if isinstance(fm, dict) else {}

    return {
        "train_total_loss": float(fm.get("train_total_loss", 0.0) or 0.0),
        "val_teacher_forced_loss": float(fm.get("val_teacher_forced_loss", 0.0) or 0.0),
        "val_free_rollout_loss": float(fm.get("val_free_rollout_loss", 0.0) or 0.0),
        "val_total_loss": float(fm.get("val_total_loss", 0.0) or 0.0),
        "tf_free_gap": float((fm.get("val_free_rollout_loss", 0.0) or 0.0) - (fm.get("val_teacher_forced_loss", 0.0) or 0.0)),
        "tapvid_free_endpoint_l2": float(tapvid.get("free_rollout_endpoint_l2", 0.0) or 0.0),
        "tapvid3d_free_endpoint_l2_limited": float(tapvid3d.get("free_rollout_endpoint_l2", 0.0) or 0.0),
    }


def stability_score(summary: Dict[str, Any]) -> float:
    hist = summary.get("epoch_history", []) if isinstance(summary, dict) else []
    losses = [float(x.get("train_total_loss", 0.0) or 0.0) for x in hist if isinstance(x, dict)]
    if not losses:
        return 0.0
    tail = losses[-3:] if len(losses) >= 3 else losses
    mean = sum(tail) / len(tail)
    var = sum((x - mean) ** 2 for x in tail) / len(tail)
    return float(var ** 0.5)


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Summarize Stage1 iteration-1 runs")
    p.add_argument("--point-summary", required=True)
    p.add_argument("--kubric-summary", required=True)
    p.add_argument("--joint-summary", required=True)
    p.add_argument("--comparison-json", required=True)
    p.add_argument("--results-md", required=True)
    p.add_argument("--protocol-doc", default="/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_ITERATION1_PROTOCOL_20260408.md")
    p.add_argument("--splits-doc", default="/home/chen034/workspace/stwm/docs/STAGE1_ITER1_SPLITS_20260408.md")
    return p


def main() -> int:
    args = build_parser().parse_args()

    point = load_json(args.point_summary)
    kubric = load_json(args.kubric_summary)
    joint = load_json(args.joint_summary)

    runs = {
        "pointodyssey_only": {
            "summary_path": str(Path(args.point_summary)),
            "stability_score": stability_score(point),
            **final_metrics(point),
            "checkpoint_best": str(point.get("checkpoint_best", "")),
            "output_dir": str(Path(point.get("checkpoint_dir", "")).parent),
        },
        "kubric_only": {
            "summary_path": str(Path(args.kubric_summary)),
            "stability_score": stability_score(kubric),
            **final_metrics(kubric),
            "checkpoint_best": str(kubric.get("checkpoint_best", "")),
            "output_dir": str(Path(kubric.get("checkpoint_dir", "")).parent),
        },
        "joint_po_kubric": {
            "summary_path": str(Path(args.joint_summary)),
            "stability_score": stability_score(joint),
            **final_metrics(joint),
            "checkpoint_best": str(joint.get("checkpoint_best", "")),
            "output_dir": str(Path(joint.get("checkpoint_dir", "")).parent),
        },
    }

    po = runs["pointodyssey_only"]
    ku = runs["kubric_only"]
    jo = runs["joint_po_kubric"]

    if (po["stability_score"], po["val_total_loss"]) <= (ku["stability_score"], ku["val_total_loss"]):
        q1_winner = "pointodyssey_only"
    else:
        q1_winner = "kubric_only"

    best_single_val_total = min(po["val_total_loss"], ku["val_total_loss"])
    best_single_tapvid = min(po["tapvid_free_endpoint_l2"], ku["tapvid_free_endpoint_l2"])
    joint_beats_single = bool(
        jo["val_total_loss"] <= best_single_val_total
        and jo["tapvid_free_endpoint_l2"] <= best_single_tapvid
    )

    gaps = {
        "pointodyssey_only": po["tf_free_gap"],
        "kubric_only": ku["tf_free_gap"],
        "joint_po_kubric": jo["tf_free_gap"],
    }

    tapvid_best = min(runs.keys(), key=lambda k: runs[k]["tapvid_free_endpoint_l2"])
    tapvid3d_best = min(runs.keys(), key=lambda k: runs[k]["tapvid3d_free_endpoint_l2_limited"])

    max_abs_gap = max(abs(g) for g in gaps.values())
    next_round = "continue_expand_stage1_trace_only" if (joint_beats_single and max_abs_gap < 0.05) else "stage1_model_fix_round"

    comparison = {
        "generated_at_utc": now_iso(),
        "task": "trace_only_future_trace_state_generation",
        "iteration_round": "stage1_iter1",
        "protocol_doc": str(Path(args.protocol_doc)),
        "splits_doc": str(Path(args.splits_doc)),
        "runs": runs,
        "answers": {
            "q1_stability_between_single_sources": {
                "winner": q1_winner,
                "metric_basis": {
                    "pointodyssey_only_stability_score": po["stability_score"],
                    "kubric_only_stability_score": ku["stability_score"],
                    "pointodyssey_only_val_total_loss": po["val_total_loss"],
                    "kubric_only_val_total_loss": ku["val_total_loss"],
                },
            },
            "q2_joint_vs_single": {
                "joint_beats_best_single": joint_beats_single,
                "metric_basis": {
                    "joint_val_total_loss": jo["val_total_loss"],
                    "best_single_val_total_loss": best_single_val_total,
                    "joint_tapvid_free_endpoint_l2": jo["tapvid_free_endpoint_l2"],
                    "best_single_tapvid_free_endpoint_l2": best_single_tapvid,
                },
            },
            "q3_teacher_forced_vs_free_gap": gaps,
            "q4_best_on_tapvid": {
                "winner": tapvid_best,
                "tapvid_free_endpoint_l2": {k: v["tapvid_free_endpoint_l2"] for k, v in runs.items()},
            },
            "q5_best_on_tapvid3d_limited": {
                "winner": tapvid3d_best,
                "tapvid3d_free_endpoint_l2_limited": {k: v["tapvid3d_free_endpoint_l2_limited"] for k, v in runs.items()},
            },
            "q6_next_round_decision": {
                "decision": next_round,
                "basis": {
                    "joint_beats_single": joint_beats_single,
                    "max_abs_tf_free_gap": max_abs_gap,
                },
            },
        },
        "next_step_choice": next_round,
    }

    comparison_path = Path(args.comparison_json)
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# TraceWM Stage 1 Iteration-1 Results (2026-04-08)",
        "",
        f"- generated_at_utc: {comparison['generated_at_utc']}",
        f"- protocol_doc: {comparison['protocol_doc']}",
        f"- splits_doc: {comparison['splits_doc']}",
        f"- comparison_json: {comparison_path}",
        "",
        "## Run Metrics",
        "",
        "| run | stability_score | val_total_loss | tf_free_gap | tapvid_free_endpoint_l2 | tapvid3d_free_endpoint_l2_limited |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name in ["pointodyssey_only", "kubric_only", "joint_po_kubric"]:
        r = runs[name]
        md_lines.append(
            f"| {name} | {r['stability_score']:.6f} | {r['val_total_loss']:.6f} | {r['tf_free_gap']:.6f} | {r['tapvid_free_endpoint_l2']:.6f} | {r['tapvid3d_free_endpoint_l2_limited']:.6f} |"
        )

    md_lines.extend([
        "",
        "## Required Answers",
        "",
        f"1. Stability winner (PointOdyssey-only vs Kubric-only): {comparison['answers']['q1_stability_between_single_sources']['winner']}",
        f"2. Joint better than best single: {comparison['answers']['q2_joint_vs_single']['joint_beats_best_single']}",
        f"3. Teacher-forced vs free gap: {comparison['answers']['q3_teacher_forced_vs_free_gap']}",
        f"4. Best on TAP-Vid: {comparison['answers']['q4_best_on_tapvid']['winner']}",
        f"5. Best on TAPVid-3D limited eval: {comparison['answers']['q5_best_on_tapvid3d_limited']['winner']}",
        f"6. Next-round decision: {comparison['answers']['q6_next_round_decision']['decision']}",
    ])

    md_path = Path(args.results_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[iter1-summary] wrote comparison: {comparison_path}")
    print(f"[iter1-summary] wrote results doc: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
