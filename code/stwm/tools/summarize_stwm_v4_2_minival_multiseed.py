from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import statistics
from typing import Any


CORE_METRICS = [
    "trajectory_l1",
    "query_localization_error",
    "semantic_loss",
    "reid_loss",
    "query_traj_gap",
    "memory_gate_mean",
    "reconnect_success_rate",
    "reappearance_event_ratio",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize STWM V4.2 mini-val multi-seed runs")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--runs", default="full_v4_2,wo_semantics_v4_2,wo_identity_v4_2")
    parser.add_argument("--summary-name", default="mini_val_summary.json")
    parser.add_argument(
        "--output-json",
        default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.json",
    )
    parser.add_argument(
        "--output-md",
        default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed/comparison_multiseed.md",
    )
    return parser


def _mean_std(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    mean = float(statistics.mean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "count": len(values)}


def _extract_metrics(summary: dict[str, Any]) -> dict[str, float]:
    losses = summary.get("average_losses", {})
    diagnostics = summary.get("diagnostics", {})
    return {
        "trajectory_l1": float(losses.get("trajectory_l1", 0.0)),
        "query_localization_error": float(losses.get("query_localization_error", 0.0)),
        "semantic_loss": float(losses.get("semantic", 0.0)),
        "reid_loss": float(losses.get("reid", 0.0)),
        "query_traj_gap": float(losses.get("query_traj_gap", 0.0)),
        "memory_gate_mean": float(diagnostics.get("memory_gate_mean", 0.0)),
        "reconnect_success_rate": float(diagnostics.get("reconnect_success_rate", 0.0)),
        "reappearance_event_ratio": float(diagnostics.get("reappearance_event_ratio", 0.0)),
    }


def _metric_is_lower_better(name: str) -> bool:
    return name not in {
        "memory_gate_mean",
        "reconnect_success_rate",
        "reappearance_event_ratio",
    }


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    seeds = [x.strip() for x in str(args.seeds).split(",") if x.strip()]
    runs = [x.strip() for x in str(args.runs).split(",") if x.strip()]

    per_seed: dict[str, dict[str, dict[str, float]]] = {run: {} for run in runs}
    summary_paths: dict[str, dict[str, str]] = {run: {} for run in runs}

    for run in runs:
        for seed in seeds:
            summary_path = runs_root / f"seed_{seed}" / run / str(args.summary_name)
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text())
            per_seed[run][seed] = _extract_metrics(payload)
            summary_paths[run][seed] = str(summary_path)

    aggregate: dict[str, dict[str, dict[str, float | int]]] = {run: {} for run in runs}
    for run in runs:
        for metric in CORE_METRICS:
            vals = [per_seed[run][seed][metric] for seed in seeds if seed in per_seed[run]]
            aggregate[run][metric] = _mean_std(vals)

    delta_vs_full: dict[str, dict[str, dict[str, float | int]]] = {}
    pairwise_per_seed: dict[str, dict[str, dict[str, float]]] = {}
    pairwise_stability: dict[str, dict[str, int]] = {}

    if "full_v4_2" in runs:
        for run in runs:
            if run == "full_v4_2":
                continue
            delta_vs_full[run] = {}
            pairwise_per_seed[run] = {}
            pairwise_stability[run] = {metric: 0 for metric in CORE_METRICS}
            for metric in CORE_METRICS:
                deltas: list[float] = []
                for seed in seeds:
                    if seed not in per_seed.get("full_v4_2", {}) or seed not in per_seed.get(run, {}):
                        continue
                    d = per_seed[run][seed][metric] - per_seed["full_v4_2"][seed][metric]
                    deltas.append(float(d))
                    pairwise_per_seed[run].setdefault(seed, {})[metric] = float(d)
                    if _metric_is_lower_better(metric):
                        if d > 0.0:
                            pairwise_stability[run][metric] += 1
                    else:
                        if d < 0.0:
                            pairwise_stability[run][metric] += 1
                delta_vs_full[run][metric] = _mean_std(deltas)

    outcome = {
        "runs_root": str(runs_root),
        "seeds": seeds,
        "runs": runs,
        "summary_name": str(args.summary_name),
        "core_metrics": CORE_METRICS,
        "summary_paths": summary_paths,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "delta_vs_full": delta_vs_full,
        "pairwise_delta_per_seed": pairwise_per_seed,
        "pairwise_stability_count": pairwise_stability,
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(outcome, indent=2))

    lines: list[str] = []
    lines.append("# STWM V4.2 Mini-Val Multi-Seed Comparison")
    lines.append("")
    lines.append(f"Runs root: `{runs_root}`")
    lines.append(f"Seeds: `{', '.join(seeds)}`")
    lines.append("")

    lines.append("## Aggregate (mean +- std)")
    lines.append("")
    lines.append(
        "| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for run in runs:
        def fmt(metric: str) -> str:
            m = aggregate[run][metric]
            return f"{float(m['mean']):.6f} +- {float(m['std']):.6f}"

        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                run,
                fmt("trajectory_l1"),
                fmt("query_localization_error"),
                fmt("semantic_loss"),
                fmt("reid_loss"),
                fmt("query_traj_gap"),
                fmt("memory_gate_mean"),
                fmt("reconnect_success_rate"),
                fmt("reappearance_event_ratio"),
            )
        )

    if delta_vs_full:
        lines.append("")
        lines.append("## Delta vs full_v4_2 (mean +- std)")
        lines.append("")
        lines.append(
            "| run | d_trajectory_l1 | d_query_localization_error | d_semantic_loss | d_reid_loss | d_query_traj_gap | d_memory_gate_mean | d_reconnect_success_rate | d_reappearance_event_ratio |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for run in runs:
            if run == "full_v4_2":
                continue

            def fmt_d(metric: str) -> str:
                m = delta_vs_full[run][metric]
                return f"{float(m['mean']):+.6f} +- {float(m['std']):.6f}"

            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    run,
                    fmt_d("trajectory_l1"),
                    fmt_d("query_localization_error"),
                    fmt_d("semantic_loss"),
                    fmt_d("reid_loss"),
                    fmt_d("query_traj_gap"),
                    fmt_d("memory_gate_mean"),
                    fmt_d("reconnect_success_rate"),
                    fmt_d("reappearance_event_ratio"),
                )
            )

    if pairwise_stability:
        lines.append("")
        lines.append("## Full Better Count Across Seeds")
        lines.append("")
        lines.append("Counts indicate in how many seeds (out of 3) full_v4_2 beats the ablation on each metric.")
        lines.append("")
        lines.append(
            "| run | trajectory_l1 | query_localization_error | semantic_loss | reid_loss | query_traj_gap | memory_gate_mean | reconnect_success_rate | reappearance_event_ratio |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for run in runs:
            if run == "full_v4_2":
                continue
            c = pairwise_stability[run]
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    run,
                    c["trajectory_l1"],
                    c["query_localization_error"],
                    c["semantic_loss"],
                    c["reid_loss"],
                    c["query_traj_gap"],
                    c["memory_gate_mean"],
                    c["reconnect_success_rate"],
                    c["reappearance_event_ratio"],
                )
            )

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n")

    print(json.dumps({"output_json": str(output_json), "output_md": str(output_md), "seeds": seeds, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
