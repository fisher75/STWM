from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import random
import statistics
from typing import Any


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Paired analysis for week2 multi-seed runs")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/week2_minival_v2_2")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--baseline", default="full")
    parser.add_argument("--compare-runs", default="wo_semantics,wo_identity_memory")
    parser.add_argument("--summary-name", default="mini_val_summary_last.json")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/reports/week2_minival_v2_2_paired_analysis.json")
    return parser


METRICS = [
    "future_trajectory_l1",
    "query_localization_error",
    "query_top1_acc",
    "query_hit_rate",
    "identity_consistency",
    "identity_switch_rate",
    "occlusion_recovery_acc",
]


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _mean(vals: list[float]) -> float:
    return float(statistics.mean(vals)) if vals else 0.0


def _std(vals: list[float]) -> float:
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0


def _bootstrap_ci(vals: list[float], iters: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), float(vals[0])
    rng = random.Random(12345)
    samples = []
    n = len(vals)
    for _ in range(max(100, int(iters))):
        draw = [vals[rng.randrange(n)] for _ in range(n)]
        samples.append(sum(draw) / n)
    samples.sort()
    lo_idx = int((alpha / 2.0) * len(samples))
    hi_idx = int((1.0 - alpha / 2.0) * len(samples)) - 1
    lo_idx = max(0, min(lo_idx, len(samples) - 1))
    hi_idx = max(0, min(hi_idx, len(samples) - 1))
    return float(samples[lo_idx]), float(samples[hi_idx])


def main() -> None:
    args = build_parser().parse_args()

    runs_root = Path(args.runs_root)
    seeds = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    baseline = str(args.baseline)
    compare_runs = [r.strip() for r in str(args.compare_runs).split(",") if r.strip()]

    # Load per-seed per-run summaries.
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for run in [baseline] + compare_runs:
        summaries[run] = {}
        for seed in seeds:
            p = runs_root / f"seed_{seed}" / run / "eval" / args.summary_name
            if p.exists():
                summaries[run][seed] = _load_summary(p)

    # Per-seed aggregate deltas (run metric - baseline metric).
    per_seed_delta: dict[str, dict[str, dict[str, float]]] = {}
    for run in compare_runs:
        per_seed_delta[run] = {}
        for seed in seeds:
            if seed not in summaries[baseline] or seed not in summaries[run]:
                continue
            b_m = summaries[baseline][seed].get("metrics", {})
            r_m = summaries[run][seed].get("metrics", {})
            per_seed_delta[run][seed] = {m: float(r_m.get(m, 0.0)) - float(b_m.get(m, 0.0)) for m in METRICS}

    # Per-clip paired deltas pooled over seeds.
    per_clip_paired: dict[str, dict[str, Any]] = {}
    for run in compare_runs:
        metric_deltas: dict[str, list[float]] = {m: [] for m in METRICS}
        paired_count = 0
        for seed in seeds:
            if seed not in summaries[baseline] or seed not in summaries[run]:
                continue
            b_pc = {str(x.get("clip_id")): x for x in summaries[baseline][seed].get("per_clip", [])}
            r_pc = {str(x.get("clip_id")): x for x in summaries[run][seed].get("per_clip", [])}
            common = sorted(set(b_pc).intersection(r_pc))
            for clip_id in common:
                paired_count += 1
                for m in METRICS:
                    metric_deltas[m].append(float(r_pc[clip_id].get(m, 0.0)) - float(b_pc[clip_id].get(m, 0.0)))

        stats = {}
        for m, vals in metric_deltas.items():
            lo, hi = _bootstrap_ci(vals, iters=int(args.bootstrap_iters), alpha=0.05)
            stats[m] = {
                "mean": _mean(vals),
                "std": _std(vals),
                "count": len(vals),
                "bootstrap_ci_95": [lo, hi],
            }
        per_clip_paired[run] = {"paired_samples": paired_count, "metric_stats": stats}

    out = {
        "runs_root": str(runs_root),
        "seeds": seeds,
        "baseline": baseline,
        "compare_runs": compare_runs,
        "metrics": METRICS,
        "per_seed_delta": per_seed_delta,
        "per_clip_paired": per_clip_paired,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    print(json.dumps({
        "output_json": str(output_path),
        "baseline": baseline,
        "compare_runs": compare_runs,
        "seeds": seeds,
    }, indent=2))


if __name__ == "__main__":
    main()
