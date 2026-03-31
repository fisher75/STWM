from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import statistics
from typing import Any

import numpy as np


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Posthoc query-trajectory decoupling multi-seed analysis for STWM V4.2")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--runs", default="full_v4_2,wo_semantics_v4_2,wo_identity_v4_2")
    parser.add_argument("--close-threshold", type=float, default=0.002)
    parser.add_argument(
        "--output-json",
        default="/home/chen034/workspace/stwm/reports/stwm_v4_2_query_decoupling_multiseed.json",
    )
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _mean_std(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    mean = float(statistics.mean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "count": len(values)}


def _analyze(rows: list[dict[str, Any]], close_threshold: float) -> dict[str, float | int | bool]:
    traj = np.asarray([float(r.get("trajectory_l1", 0.0)) for r in rows], dtype=np.float64)
    query = np.asarray([float(r.get("query_localization_error", 0.0)) for r in rows], dtype=np.float64)
    if traj.size == 0:
        return {
            "num_rows": 0,
            "pearson_corr": 0.0,
            "close_ratio": 0.0,
            "mean_abs_gap": 0.0,
            "mean_signed_gap": 0.0,
            "decoupling_score": 0.0,
            "exact_equal_ratio": 0.0,
            "proxy_like": False,
        }

    if traj.size > 1 and np.std(traj) > 1e-12 and np.std(query) > 1e-12:
        corr = float(np.corrcoef(traj, query)[0, 1])
    else:
        corr = 1.0

    gap = query - traj
    close_ratio = float(np.mean(np.abs(gap) <= float(close_threshold)))
    mean_abs_gap = float(np.mean(np.abs(gap)))
    mean_signed_gap = float(np.mean(gap))
    exact_equal_ratio = float(np.mean(np.abs(gap) <= 1e-12))

    decoupling_score = float(
        max(0.0, min(1.0, 0.5 * (1.0 - min(1.0, abs(corr))) + 0.5 * (1.0 - close_ratio)))
    )
    proxy_like = bool(abs(corr) > 0.95 and close_ratio > 0.70)

    return {
        "num_rows": int(traj.size),
        "pearson_corr": corr,
        "close_ratio": close_ratio,
        "mean_abs_gap": mean_abs_gap,
        "mean_signed_gap": mean_signed_gap,
        "decoupling_score": decoupling_score,
        "exact_equal_ratio": exact_equal_ratio,
        "proxy_like": proxy_like,
    }


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    seeds = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    runs = [r.strip() for r in str(args.runs).split(",") if r.strip()]

    per_seed: dict[str, dict[str, dict[str, float | int | bool]]] = {run: {} for run in runs}

    for run in runs:
        for seed in seeds:
            rows = _read_rows(runs_root / f"seed_{seed}" / run / "train_log.jsonl")
            per_seed[run][seed] = _analyze(rows, close_threshold=float(args.close_threshold))

    aggregate: dict[str, dict[str, dict[str, float | int]]] = {run: {} for run in runs}
    agg_keys = [
        "pearson_corr",
        "close_ratio",
        "mean_abs_gap",
        "mean_signed_gap",
        "decoupling_score",
        "exact_equal_ratio",
    ]
    for run in runs:
        for key in agg_keys:
            vals = [float(per_seed[run][seed][key]) for seed in seeds if seed in per_seed[run]]
            aggregate[run][key] = _mean_std(vals)
        aggregate[run]["proxy_like_count"] = int(
            sum(1 for seed in seeds if bool(per_seed[run][seed].get("proxy_like", False)))
        )

    comparison_vs_full: dict[str, Any] = {}
    if "full_v4_2" in runs:
        full = per_seed["full_v4_2"]
        for run in runs:
            if run == "full_v4_2":
                continue
            deltas = {
                "delta_corr_abs": [],
                "delta_close_ratio": [],
                "delta_decoupling_score": [],
            }
            per_seed_delta: dict[str, dict[str, float]] = {}
            for seed in seeds:
                if seed not in full or seed not in per_seed[run]:
                    continue
                d_corr = float(abs(per_seed[run][seed]["pearson_corr"]) - abs(full[seed]["pearson_corr"]))
                d_close = float(per_seed[run][seed]["close_ratio"] - full[seed]["close_ratio"])
                d_score = float(per_seed[run][seed]["decoupling_score"] - full[seed]["decoupling_score"])
                deltas["delta_corr_abs"].append(d_corr)
                deltas["delta_close_ratio"].append(d_close)
                deltas["delta_decoupling_score"].append(d_score)
                per_seed_delta[seed] = {
                    "delta_corr_abs": d_corr,
                    "delta_close_ratio": d_close,
                    "delta_decoupling_score": d_score,
                }
            comparison_vs_full[run] = {
                "per_seed_delta": per_seed_delta,
                "aggregate_delta": {
                    key: _mean_std(vals) for key, vals in deltas.items()
                },
            }

    # Old-v1 proxy signature is used as a conservative reference:
    # query metric was effectively tied to trajectory metric.
    old_v1_reference = {
        "proxy_like_signature": {
            "abs_corr_gt": 0.95,
            "close_ratio_gt": 0.70,
            "exact_equal_ratio_approx": 1.0,
        }
    }

    full_stably_better_than_old_v1 = False
    if "full_v4_2" in per_seed:
        conds = []
        for seed in seeds:
            row = per_seed["full_v4_2"].get(seed, {})
            conds.append(
                (not bool(row.get("proxy_like", False)))
                and float(row.get("close_ratio", 1.0)) < 0.70
                and float(row.get("exact_equal_ratio", 1.0)) < 0.95
            )
        full_stably_better_than_old_v1 = bool(conds) and all(conds)

    out: dict[str, Any] = {
        "runs_root": str(runs_root),
        "seeds": seeds,
        "runs": runs,
        "close_threshold": float(args.close_threshold),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "comparison_vs_full": comparison_vs_full,
        "old_v1_reference": old_v1_reference,
        "judgement": {
            "full_stably_better_than_old_v1_proxy_state": full_stably_better_than_old_v1,
            "full_proxy_like_count": int(aggregate.get("full_v4_2", {}).get("proxy_like_count", 0)),
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    print(json.dumps({"output_json": str(output_path), "seeds": seeds, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
