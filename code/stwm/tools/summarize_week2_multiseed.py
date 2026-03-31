from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import statistics
from typing import Any


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize week2 mini-val multi-seed runs")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/week2_minival_v2_1")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--runs", default="full,wo_semantics,wo_identity_memory")
    parser.add_argument("--summary-name", default="mini_val_summary_last.json")
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/reports/week2_minival_v2_1_multiseed_summary.json")
    return parser


METRICS = [
    "future_mask_iou",
    "future_trajectory_l1",
    "query_localization_error",
    "query_top1_acc",
    "query_hit_rate",
    "identity_consistency",
    "identity_switch_rate",
    "occlusion_recovery_acc",
]


def _mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    mean = float(statistics.mean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "count": len(values)}


def main() -> None:
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    seeds = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    runs = [r.strip() for r in str(args.runs).split(",") if r.strip()]

    per_seed: dict[str, dict[str, dict[str, float]]] = {run: {} for run in runs}

    for run in runs:
        for seed in seeds:
            summary_path = runs_root / f"seed_{seed}" / run / "eval" / args.summary_name
            if not summary_path.exists():
                continue
            payload = json.loads(summary_path.read_text())
            metrics = payload.get("metrics", {})
            per_seed[run][seed] = {k: float(metrics.get(k, 0.0)) for k in METRICS}

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for run in runs:
        aggregate[run] = {}
        for metric in METRICS:
            vals = [per_seed[run][seed][metric] for seed in seeds if seed in per_seed[run]]
            aggregate[run][metric] = _mean_std(vals)

    pairwise_delta_vs_full: dict[str, dict[str, dict[str, float]]] = {}
    if "full" in runs:
        for run in runs:
            if run == "full":
                continue
            pairwise_delta_vs_full[run] = {}
            for metric in METRICS:
                deltas = []
                for seed in seeds:
                    if seed in per_seed.get("full", {}) and seed in per_seed.get(run, {}):
                        deltas.append(per_seed[run][seed][metric] - per_seed["full"][seed][metric])
                pairwise_delta_vs_full[run][metric] = _mean_std(deltas)

    output = {
        "runs_root": str(runs_root),
        "seeds": seeds,
        "runs": runs,
        "metrics": METRICS,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "delta_vs_full": pairwise_delta_vs_full,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))

    print(json.dumps({"output_json": str(output_path), "runs": runs, "seeds": seeds}, indent=2))


if __name__ == "__main__":
    main()
