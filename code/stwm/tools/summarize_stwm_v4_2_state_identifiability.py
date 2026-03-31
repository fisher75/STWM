from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import statistics
from typing import Any


QUERY_TYPES = [
    "same_category_distractor",
    "spatial_disambiguation",
    "relation_conditioned_query",
    "future_conditioned_reappearance_aware",
]

CORE_METRICS = [
    "trajectory_l1",
    "query_localization_error",
    "query_traj_gap",
    "reconnect_success_rate",
    "reappearance_event_ratio",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize STWM V4.2 state-identifiability evaluation")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_state_identifiability")
    parser.add_argument("--manifest", default="/home/chen034/workspace/stwm/manifests/minisplits/stwm_v4_2_state_identifiability_v1.json")
    parser.add_argument("--seeds", default="42,123")
    parser.add_argument("--runs", default="full_v4_2,wo_semantics_v4_2,wo_object_bias_v4_2")
    parser.add_argument("--summary-name", default="mini_val_summary.json")
    parser.add_argument("--log-name", default="train_log.jsonl")
    parser.add_argument(
        "--output-json",
        default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.json",
    )
    parser.add_argument(
        "--output-md",
        default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_state_identifiability/comparison_state_identifiability.md",
    )
    return parser


def _mean_std(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    mean = float(statistics.mean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "count": len(values)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _extract_summary_metrics(summary: dict[str, Any]) -> dict[str, float]:
    losses = summary.get("average_losses", {})
    diagnostics = summary.get("diagnostics", {})
    return {
        "trajectory_l1": float(losses.get("trajectory_l1", 0.0)),
        "query_localization_error": float(losses.get("query_localization_error", 0.0)),
        "query_traj_gap": float(losses.get("query_traj_gap", 0.0)),
        "reconnect_success_rate": float(diagnostics.get("reconnect_success_rate", 0.0)),
        "reappearance_event_ratio": float(diagnostics.get("reappearance_event_ratio", 0.0)),
    }


def _load_clip_query_types(manifest_path: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    items = _load_json(manifest_path)
    clip_types: dict[str, list[str]] = {}
    type_clip_count = {q: 0 for q in QUERY_TYPES}

    for item in items:
        clip_id = str(item.get("clip_id", ""))
        if not clip_id:
            continue
        md = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        proto = md.get("state_identifiability_protocol", {}) if isinstance(md.get("state_identifiability_protocol"), dict) else {}
        qtypes = [str(x) for x in proto.get("query_types", []) if str(x)]
        qtypes = [x for x in qtypes if x in QUERY_TYPES]
        clip_types[clip_id] = qtypes
        for q in qtypes:
            type_clip_count[q] = int(type_clip_count[q] + 1)

    return clip_types, type_clip_count


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    n = len(rows)
    if n <= 0:
        return {
            "num_rows": 0,
            "trajectory_l1": 0.0,
            "query_localization_error": 0.0,
            "query_traj_gap": 0.0,
            "reconnect_success_rate": 0.0,
            "reappearance_event_ratio": 0.0,
        }

    traj = [float(r.get("trajectory_l1", 0.0)) for r in rows]
    query = [float(r.get("query_localization_error", 0.0)) for r in rows]
    qgap = [float(r.get("query_traj_gap", 0.0)) for r in rows]
    reconnect = [float(r.get("reconnect_success", 0.0)) for r in rows]
    reappear = [float(r.get("has_reappearance_event", 0.0)) for r in rows]
    return {
        "num_rows": int(n),
        "trajectory_l1": float(sum(traj) / max(1, len(traj))),
        "query_localization_error": float(sum(query) / max(1, len(query))),
        "query_traj_gap": float(sum(qgap) / max(1, len(qgap))),
        "reconnect_success_rate": float(sum(reconnect) / max(1, len(reconnect))),
        "reappearance_event_ratio": float(sum(reappear) / max(1, len(reappear))),
    }


def _rows_for_query_type(rows: list[dict[str, Any]], clip_query_types: dict[str, list[str]], query_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        clip_id = str(row.get("clip_id", ""))
        if not clip_id:
            continue
        types = clip_query_types.get(clip_id, [])
        if query_type in types:
            out.append(row)
    return out


def _metric_lower_better(name: str) -> bool:
    return name not in {"reconnect_success_rate", "reappearance_event_ratio"}


def _fmt(mean_std: dict[str, float | int], signed: bool = False) -> str:
    mean = float(mean_std.get("mean", 0.0))
    std = float(mean_std.get("std", 0.0))
    if signed:
        return f"{mean:+.6f} +- {std:.6f}"
    return f"{mean:.6f} +- {std:.6f}"


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    manifest_path = Path(args.manifest)
    seeds = [x.strip() for x in str(args.seeds).split(",") if x.strip()]
    runs = [x.strip() for x in str(args.runs).split(",") if x.strip()]

    clip_query_types, type_clip_count = _load_clip_query_types(manifest_path)

    per_seed: dict[str, dict[str, Any]] = {run: {} for run in runs}
    summary_paths: dict[str, dict[str, str]] = {run: {} for run in runs}

    for run in runs:
        for seed in seeds:
            run_dir = runs_root / f"seed_{seed}" / run
            summary_path = run_dir / str(args.summary_name)
            log_path = run_dir / str(args.log_name)
            if not summary_path.exists() or not log_path.exists():
                continue

            summary = _load_json(summary_path)
            rows = _read_rows(log_path)
            overall = _extract_summary_metrics(summary)

            per_type: dict[str, dict[str, float | int]] = {}
            for query_type in QUERY_TYPES:
                type_rows = _rows_for_query_type(rows, clip_query_types, query_type)
                per_type[query_type] = _metrics_from_rows(type_rows)

            per_seed[run][seed] = {
                "overall": overall,
                "per_type": per_type,
                "num_rows": len(rows),
                "summary_path": str(summary_path),
                "log_path": str(log_path),
            }
            summary_paths[run][seed] = str(summary_path)

    aggregate: dict[str, Any] = {run: {"overall": {}, "per_type": {}} for run in runs}
    for run in runs:
        for metric in CORE_METRICS:
            vals = [float(per_seed[run][seed]["overall"][metric]) for seed in seeds if seed in per_seed[run]]
            aggregate[run]["overall"][metric] = _mean_std(vals)

        for query_type in QUERY_TYPES:
            aggregate[run]["per_type"][query_type] = {}
            for metric in CORE_METRICS:
                vals = [
                    float(per_seed[run][seed]["per_type"][query_type][metric])
                    for seed in seeds
                    if seed in per_seed[run] and int(per_seed[run][seed]["per_type"][query_type]["num_rows"]) > 0
                ]
                aggregate[run]["per_type"][query_type][metric] = _mean_std(vals)
            row_counts = [
                int(per_seed[run][seed]["per_type"][query_type]["num_rows"])
                for seed in seeds
                if seed in per_seed[run]
            ]
            aggregate[run]["per_type"][query_type]["num_rows_mean"] = float(sum(row_counts) / max(1, len(row_counts))) if row_counts else 0.0

    delta_vs_full: dict[str, Any] = {}
    sign_consistency: dict[str, Any] = {}
    if "full_v4_2" in runs:
        for run in runs:
            if run == "full_v4_2":
                continue

            delta_vs_full[run] = {"overall": {}, "per_type": {}}
            sign_consistency[run] = {"overall": {}, "per_type": {}}

            for metric in CORE_METRICS:
                deltas: list[float] = []
                full_better = 0
                for seed in seeds:
                    if seed not in per_seed.get("full_v4_2", {}) or seed not in per_seed.get(run, {}):
                        continue
                    d = float(per_seed[run][seed]["overall"][metric]) - float(per_seed["full_v4_2"][seed]["overall"][metric])
                    deltas.append(d)
                    if _metric_lower_better(metric):
                        if d > 0.0:
                            full_better += 1
                    else:
                        if d < 0.0:
                            full_better += 1
                delta_vs_full[run]["overall"][metric] = _mean_std(deltas)
                sign_consistency[run]["overall"][metric] = int(full_better)

            for query_type in QUERY_TYPES:
                delta_vs_full[run]["per_type"][query_type] = {}
                sign_consistency[run]["per_type"][query_type] = {}
                for metric in CORE_METRICS:
                    deltas: list[float] = []
                    full_better = 0
                    for seed in seeds:
                        if seed not in per_seed.get("full_v4_2", {}) or seed not in per_seed.get(run, {}):
                            continue
                        a = per_seed["full_v4_2"][seed]["per_type"][query_type]
                        b = per_seed[run][seed]["per_type"][query_type]
                        if int(a.get("num_rows", 0)) <= 0 or int(b.get("num_rows", 0)) <= 0:
                            continue
                        d = float(b[metric]) - float(a[metric])
                        deltas.append(d)
                        if _metric_lower_better(metric):
                            if d > 0.0:
                                full_better += 1
                        else:
                            if d < 0.0:
                                full_better += 1
                    delta_vs_full[run]["per_type"][query_type][metric] = _mean_std(deltas)
                    sign_consistency[run]["per_type"][query_type][metric] = int(full_better)

    out = {
        "runs_root": str(runs_root),
        "manifest": str(manifest_path),
        "seeds": seeds,
        "runs": runs,
        "query_types": QUERY_TYPES,
        "core_metrics": CORE_METRICS,
        "query_type_clip_count": type_clip_count,
        "summary_paths": summary_paths,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "delta_vs_full": delta_vs_full,
        "sign_consistency": sign_consistency,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2))

    lines: list[str] = []
    lines.append("# STWM V4.2 State-Identifiability Summary")
    lines.append("")
    lines.append(f"Runs root: `{runs_root}`")
    lines.append(f"Seeds: `{', '.join(seeds)}`")
    lines.append("")

    lines.append("## Overall Aggregate (mean +- std)")
    lines.append("")
    lines.append("| run | trajectory_l1 | query_localization_error | query_traj_gap | reconnect_success_rate | reappearance_event_ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for run in runs:
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                run,
                _fmt(aggregate[run]["overall"]["trajectory_l1"]),
                _fmt(aggregate[run]["overall"]["query_localization_error"]),
                _fmt(aggregate[run]["overall"]["query_traj_gap"]),
                _fmt(aggregate[run]["overall"]["reconnect_success_rate"]),
                _fmt(aggregate[run]["overall"]["reappearance_event_ratio"]),
            )
        )

    if delta_vs_full:
        lines.append("")
        lines.append("## Overall Delta vs full_v4_2 (run - full)")
        lines.append("")
        lines.append("| run | d_trajectory_l1 | d_query_localization_error | d_query_traj_gap | d_reconnect_success_rate | d_reappearance_event_ratio |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for run in runs:
            if run == "full_v4_2":
                continue
            d = delta_vs_full[run]["overall"]
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    run,
                    _fmt(d["trajectory_l1"], signed=True),
                    _fmt(d["query_localization_error"], signed=True),
                    _fmt(d["query_traj_gap"], signed=True),
                    _fmt(d["reconnect_success_rate"], signed=True),
                    _fmt(d["reappearance_event_ratio"], signed=True),
                )
            )

    lines.append("")
    lines.append("## Per-Type Delta vs full_v4_2 (run - full)")
    lines.append("")
    lines.append("| query_type | clip_count | run | d_traj | d_query | d_query_gap | d_reconnect_success | full_better_query_count |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for query_type in QUERY_TYPES:
        clip_count = int(type_clip_count.get(query_type, 0))
        for run in runs:
            if run == "full_v4_2":
                continue
            d = delta_vs_full.get(run, {}).get("per_type", {}).get(query_type, {})
            sc = sign_consistency.get(run, {}).get("per_type", {}).get(query_type, {})
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    query_type,
                    clip_count,
                    run,
                    _fmt(d.get("trajectory_l1", {"mean": 0.0, "std": 0.0}), signed=True),
                    _fmt(d.get("query_localization_error", {"mean": 0.0, "std": 0.0}), signed=True),
                    _fmt(d.get("query_traj_gap", {"mean": 0.0, "std": 0.0}), signed=True),
                    _fmt(d.get("reconnect_success_rate", {"mean": 0.0, "std": 0.0}), signed=True),
                    int(sc.get("query_localization_error", 0)),
                )
            )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "runs": runs, "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
