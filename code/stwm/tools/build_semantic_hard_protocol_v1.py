from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json
from typing import Any


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build reproducible semantic-hard protocol manifest (seed42 v1)")
    parser.add_argument(
        "--source-manifest",
        default="/home/chen034/workspace/stwm/manifests/protocol_v2/protocol_val_main_v1.json",
    )
    parser.add_argument(
        "--baseline-eval-summaries",
        nargs="+",
        default=[
            (
                "/home/chen034/workspace/stwm/outputs/training/"
                "stwm_v4_2_220m_protocol_frozen_frontend_qstr_mainline_seed42_v1/"
                "seed_42/trace_sem_baseline_seed42_qstr_control_v1/checkpoints/protocol_eval/"
                "protocol_val_main_step_002000.json"
            ),
            (
                "/home/chen034/workspace/stwm/outputs/training/"
                "stwm_v4_2_220m_protocol_frozen_frontend_qtsa_mainline_seed42_v1/"
                "seed_42/trace_sem_baseline_seed42_qtsa_control_v1/checkpoints/protocol_eval/"
                "protocol_val_main_step_002000.json"
            ),
        ],
    )
    parser.add_argument("--target-size", type=int, default=96)
    parser.add_argument(
        "--output-manifest",
        default="/home/chen034/workspace/stwm/manifests/protocol_v2/semantic_hard_seed42_v1.json",
    )
    parser.add_argument(
        "--output-report",
        default="/home/chen034/workspace/stwm/reports/stwm_semantic_hard_protocol_v1.json",
    )
    parser.add_argument(
        "--output-clip-ids",
        default="/home/chen034/workspace/stwm/manifests/protocol_v2/semantic_hard_seed42_v1_clip_ids.json",
    )
    return parser


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if x != x:
        return float(default)
    if x in {float("inf"), float("-inf")}:
        return float(default)
    return float(x)


def _mean(vals: list[float]) -> float:
    if not vals:
        return 0.0
    return float(sum(vals) / max(1, len(vals)))


def _std(vals: list[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    m = _mean(vals)
    var = sum((x - m) ** 2 for x in vals) / float(len(vals))
    return float(var ** 0.5)


def _rank01(metric: dict[str, float], higher_is_harder: bool) -> dict[str, float]:
    if not metric:
        return {}
    items = sorted(metric.items(), key=lambda kv: (kv[1], kv[0]))
    n = max(1, len(items) - 1)
    out: dict[str, float] = {}
    for i, (clip_id, _) in enumerate(items):
        p = float(i) / float(n)
        out[clip_id] = p if higher_is_harder else (1.0 - p)
    return out


def _aggregate_eval_per_clip(eval_payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = eval_payload.get("per_clip", [])
    clip_rows: dict[str, list[dict[str, Any]]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                continue
            clip_rows.setdefault(clip_id, []).append(row)

    agg: dict[str, dict[str, float]] = {}
    for clip_id, per_rows in clip_rows.items():
        qerr = [_safe_float(r.get("query_localization_error"), 0.0) for r in per_rows]
        top1 = [_safe_float(r.get("query_top1_acc"), 0.0) for r in per_rows]
        hit = [_safe_float(r.get("query_hit_rate"), 0.0) for r in per_rows]
        same = [_safe_float(r.get("query_same_class_candidates"), 1.0) for r in per_rows]
        agg[clip_id] = {
            "query_localization_error": _mean(qerr),
            "query_top1_acc": _mean(top1),
            "query_hit_rate": _mean(hit),
            "query_same_class_candidates": _mean(same),
        }
    return agg


def main() -> None:
    args = build_parser().parse_args()

    source_manifest_path = Path(args.source_manifest)
    source_items = json.loads(source_manifest_path.read_text())
    if not isinstance(source_items, list):
        raise RuntimeError(f"invalid source manifest payload: {source_manifest_path}")

    item_map: dict[str, dict[str, Any]] = {}
    for item in source_items:
        if not isinstance(item, dict):
            continue
        clip_id = str(item.get("clip_id", "")).strip()
        if clip_id:
            item_map[clip_id] = item

    per_run_metrics: dict[str, dict[str, dict[str, float]]] = {}
    resolved_eval_paths: list[str] = []
    for p_raw in args.baseline_eval_summaries:
        p = Path(str(p_raw))
        if not p.exists():
            continue
        payload = json.loads(p.read_text())
        run_name = str(payload.get("run_name", p.stem))
        per_run_metrics[run_name] = _aggregate_eval_per_clip(payload)
        resolved_eval_paths.append(str(p))

    if not per_run_metrics:
        raise RuntimeError("no valid baseline eval summaries found")

    clip_stats: dict[str, dict[str, Any]] = {}
    for clip_id in sorted(item_map.keys()):
        qerr_runs: list[float] = []
        top1_runs: list[float] = []
        hit_runs: list[float] = []
        same_runs: list[float] = []
        supporting_runs: list[str] = []

        for run_name, clip_metric_map in per_run_metrics.items():
            row = clip_metric_map.get(clip_id)
            if row is None:
                continue
            supporting_runs.append(run_name)
            qerr_runs.append(_safe_float(row.get("query_localization_error"), 0.0))
            top1_runs.append(_safe_float(row.get("query_top1_acc"), 0.0))
            hit_runs.append(_safe_float(row.get("query_hit_rate"), 0.0))
            same_runs.append(_safe_float(row.get("query_same_class_candidates"), 1.0))

        if not supporting_runs:
            continue

        avg_qerr = _mean(qerr_runs)
        avg_top1 = _mean(top1_runs)
        avg_hit = _mean(hit_runs)
        avg_same = _mean(same_runs)

        clip_stats[clip_id] = {
            "avg_query_localization_error": avg_qerr,
            "avg_query_top1_acc": avg_top1,
            "avg_query_hit_rate": avg_hit,
            "avg_query_same_class_candidates": avg_same,
            "query_localization_error_std": _std(qerr_runs),
            "query_top1_acc_std": _std(top1_runs),
            "supporting_runs": supporting_runs,
        }

    if not clip_stats:
        raise RuntimeError("no overlapping clip-level metrics found between source manifest and baseline evals")

    metric_qerr = {k: float(v["avg_query_localization_error"]) for k, v in clip_stats.items()}
    metric_fail = {k: float(1.0 - v["avg_query_top1_acc"]) for k, v in clip_stats.items()}
    metric_miss = {k: float(1.0 - v["avg_query_hit_rate"]) for k, v in clip_stats.items()}
    metric_instab = {
        k: float(v["query_localization_error_std"] + v["query_top1_acc_std"]) for k, v in clip_stats.items()
    }
    metric_ambig = {k: float(v["avg_query_same_class_candidates"]) for k, v in clip_stats.items()}

    rank_qerr = _rank01(metric_qerr, higher_is_harder=True)
    rank_fail = _rank01(metric_fail, higher_is_harder=True)
    rank_miss = _rank01(metric_miss, higher_is_harder=True)
    rank_instab = _rank01(metric_instab, higher_is_harder=True)
    rank_ambig = _rank01(metric_ambig, higher_is_harder=True)

    weighted: list[dict[str, Any]] = []
    for clip_id, stats in clip_stats.items():
        score = (
            0.45 * rank_qerr.get(clip_id, 0.0)
            + 0.35 * rank_fail.get(clip_id, 0.0)
            + 0.10 * rank_miss.get(clip_id, 0.0)
            + 0.08 * rank_instab.get(clip_id, 0.0)
            + 0.02 * rank_ambig.get(clip_id, 0.0)
        )
        weighted.append(
            {
                "clip_id": clip_id,
                "difficulty_score": float(score),
                "components": {
                    "qerr_rank": float(rank_qerr.get(clip_id, 0.0)),
                    "fail_rank": float(rank_fail.get(clip_id, 0.0)),
                    "miss_rank": float(rank_miss.get(clip_id, 0.0)),
                    "instability_rank": float(rank_instab.get(clip_id, 0.0)),
                    "ambiguity_rank": float(rank_ambig.get(clip_id, 0.0)),
                },
                "stats": stats,
            }
        )

    weighted.sort(key=lambda x: (-float(x["difficulty_score"]), str(x["clip_id"])))
    target_size = max(1, int(args.target_size))
    selected = weighted[:target_size]
    selected_clip_ids = [str(x["clip_id"]) for x in selected]

    output_manifest: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        clip_id = str(row["clip_id"])
        src = dict(item_map[clip_id])
        md = dict(src.get("metadata", {}))
        md["semantic_hard_protocol_v1"] = {
            "rank": int(rank),
            "difficulty_score": float(row["difficulty_score"]),
            "selection_rule_version": "semantic_hard_v1",
            "components": row["components"],
            "stats": row["stats"],
            "source_eval_summaries": resolved_eval_paths,
        }
        src["metadata"] = md
        output_manifest.append(src)

    source_manifest_sha1 = hashlib.sha1(source_manifest_path.read_bytes()).hexdigest()
    selected_avg_top1 = _mean([float(x["stats"]["avg_query_top1_acc"]) for x in selected]) if selected else 0.0
    selected_avg_qerr = _mean([float(x["stats"]["avg_query_localization_error"]) for x in selected]) if selected else 0.0
    full_avg_top1 = _mean([float(x["avg_query_top1_acc"]) for x in clip_stats.values()])
    full_avg_qerr = _mean([float(x["avg_query_localization_error"]) for x in clip_stats.values()])

    report = {
        "selection_rule_version": "semantic_hard_v1",
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha1": source_manifest_sha1,
        "baseline_eval_summaries": resolved_eval_paths,
        "candidate_clip_count": int(len(clip_stats)),
        "target_size": int(target_size),
        "selected_count": int(len(selected)),
        "selected_clip_ids": selected_clip_ids,
        "difficulty_weights": {
            "query_localization_error": 0.45,
            "query_top1_fail": 0.35,
            "query_hit_miss": 0.10,
            "cross_run_instability": 0.08,
            "same_class_ambiguity": 0.02,
        },
        "coverage": {
            "selected_avg_query_top1_acc": float(selected_avg_top1),
            "selected_avg_query_localization_error": float(selected_avg_qerr),
            "full_avg_query_top1_acc": float(full_avg_top1),
            "full_avg_query_localization_error": float(full_avg_qerr),
            "selected_is_harder_than_full": bool(
                (selected_avg_top1 <= full_avg_top1) and (selected_avg_qerr >= full_avg_qerr)
            ),
        },
        "selected_clips": selected,
    }

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps(output_manifest, indent=2))

    out_report = Path(args.output_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2))

    out_clip_ids = Path(args.output_clip_ids)
    out_clip_ids.parent.mkdir(parents=True, exist_ok=True)
    out_clip_ids.write_text(json.dumps(selected_clip_ids, indent=2))

    print(
        json.dumps(
            {
                "output_manifest": str(out_manifest),
                "output_report": str(out_report),
                "output_clip_ids": str(out_clip_ids),
                "selected_count": int(len(selected)),
                "selected_is_harder_than_full": bool(report["coverage"]["selected_is_harder_than_full"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
