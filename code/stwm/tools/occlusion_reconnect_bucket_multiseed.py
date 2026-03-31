from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import statistics
from typing import Any

import numpy as np


EVENT_METRICS = [
    "event_row_ratio",
    "reconnect_success_rate",
    "reconnect_min_error_mean",
    "trajectory_l1_on_events",
    "query_localization_error_on_events",
    "memory_gate_mean_on_events",
    "reappearance_count_mean",
]


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Posthoc occlusion/reconnect bucket multi-seed analysis for STWM V4.2")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_multiseed")
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--runs", default="full_v4_2,wo_semantics_v4_2,wo_identity_v4_2")
    parser.add_argument("--min-total-event-rows", type=int, default=10)
    parser.add_argument("--min-paired-seeds", type=int, default=2)
    parser.add_argument(
        "--output-json",
        default="/home/chen034/workspace/stwm/reports/stwm_v4_2_occlusion_reconnect_multiseed.json",
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


def _mean(vals: list[float]) -> float:
    return float(np.mean(vals)) if vals else 0.0


def _mean_std(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "count": 0}
    mean = float(statistics.mean(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    return {"mean": mean, "std": std, "count": len(values)}


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    all_n = len(rows)
    event_rows = [r for r in rows if float(r.get("has_reappearance_event", 0.0)) > 0.5]
    event_n = len(event_rows)

    succ = [float(r.get("reconnect_success", 0.0)) for r in event_rows]
    reconnect_err = [float(r.get("reconnect_min_error", 0.0)) for r in event_rows]
    traj = [float(r.get("trajectory_l1", 0.0)) for r in event_rows]
    query = [float(r.get("query_localization_error", 0.0)) for r in event_rows]
    gate = [float(r.get("memory_gate_mean", 0.0)) for r in event_rows]
    events_per_row = [float(r.get("reappearance_count", 0)) for r in event_rows]

    return {
        "total_rows": int(all_n),
        "event_rows": int(event_n),
        "event_row_ratio": float(event_n / all_n) if all_n > 0 else 0.0,
        "reconnect_success_rate": _mean(succ),
        "reconnect_min_error_mean": _mean(reconnect_err),
        "trajectory_l1_on_events": _mean(traj),
        "query_localization_error_on_events": _mean(query),
        "memory_gate_mean_on_events": _mean(gate),
        "reappearance_count_mean": _mean(events_per_row),
    }


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    seeds = [s.strip() for s in str(args.seeds).split(",") if s.strip()]
    runs = [r.strip() for r in str(args.runs).split(",") if r.strip()]

    per_seed: dict[str, dict[str, dict[str, Any]]] = {run: {} for run in runs}
    for run in runs:
        for seed in seeds:
            rows = _read_rows(runs_root / f"seed_{seed}" / run / "train_log.jsonl")
            per_seed[run][seed] = _stats(rows)

    aggregate: dict[str, dict[str, Any]] = {run: {} for run in runs}
    for run in runs:
        total_event_rows = int(sum(int(per_seed[run][seed]["event_rows"]) for seed in seeds if seed in per_seed[run]))
        aggregate[run]["total_event_rows"] = total_event_rows
        for metric in EVENT_METRICS:
            vals = [float(per_seed[run][seed][metric]) for seed in seeds if seed in per_seed[run]]
            aggregate[run][metric] = _mean_std(vals)

    comparison_vs_full: dict[str, Any] = {}
    if "full_v4_2" in runs:
        for run in runs:
            if run == "full_v4_2":
                continue
            per_seed_delta: dict[str, dict[str, float]] = {}
            delta_vals: dict[str, list[float]] = {
                "delta_event_row_ratio": [],
                "delta_reconnect_success_rate": [],
                "delta_reconnect_min_error_mean": [],
                "delta_query_error_on_events": [],
                "delta_traj_l1_on_events": [],
            }
            paired_seed_count = 0
            for seed in seeds:
                if seed not in per_seed.get("full_v4_2", {}) or seed not in per_seed.get(run, {}):
                    continue
                full_s = per_seed["full_v4_2"][seed]
                cur_s = per_seed[run][seed]
                if int(full_s["event_rows"]) <= 0 and int(cur_s["event_rows"]) <= 0:
                    continue
                paired_seed_count += 1
                d_event = float(cur_s["event_row_ratio"] - full_s["event_row_ratio"])
                d_succ = float(cur_s["reconnect_success_rate"] - full_s["reconnect_success_rate"])
                d_recon_err = float(cur_s["reconnect_min_error_mean"] - full_s["reconnect_min_error_mean"])
                d_query = float(cur_s["query_localization_error_on_events"] - full_s["query_localization_error_on_events"])
                d_traj = float(cur_s["trajectory_l1_on_events"] - full_s["trajectory_l1_on_events"])
                per_seed_delta[seed] = {
                    "delta_event_row_ratio": d_event,
                    "delta_reconnect_success_rate": d_succ,
                    "delta_reconnect_min_error_mean": d_recon_err,
                    "delta_query_error_on_events": d_query,
                    "delta_traj_l1_on_events": d_traj,
                }
                delta_vals["delta_event_row_ratio"].append(d_event)
                delta_vals["delta_reconnect_success_rate"].append(d_succ)
                delta_vals["delta_reconnect_min_error_mean"].append(d_recon_err)
                delta_vals["delta_query_error_on_events"].append(d_query)
                delta_vals["delta_traj_l1_on_events"].append(d_traj)

            comparison_vs_full[run] = {
                "paired_seed_count": paired_seed_count,
                "per_seed_delta": per_seed_delta,
                "aggregate_delta": {k: _mean_std(v) for k, v in delta_vals.items()},
            }

    comparison_runs = [run for run in runs if run != "full_v4_2"]
    paired_seed_count_by_run = {
        run: int(comparison_vs_full.get(run, {}).get("paired_seed_count", 0))
        for run in comparison_runs
    }

    statistical_power = {
        "min_total_event_rows": int(args.min_total_event_rows),
        "min_paired_seeds": int(args.min_paired_seeds),
        "per_run_total_event_rows": {
            run: int(aggregate[run]["total_event_rows"]) for run in runs
        },
        "comparison_runs": comparison_runs,
        "paired_seed_count_by_run": paired_seed_count_by_run,
        "full_has_min_event_rows": bool(
            int(aggregate.get("full_v4_2", {}).get("total_event_rows", 0)) >= int(args.min_total_event_rows)
        ),
        # Keep legacy fields for backward compatibility with older reports/docs.
        "full_vs_wo_semantics_paired_seed_count": int(
            comparison_vs_full.get("wo_semantics_v4_2", {}).get("paired_seed_count", 0)
        ),
        "full_vs_wo_identity_paired_seed_count": int(
            comparison_vs_full.get("wo_identity_v4_2", {}).get("paired_seed_count", 0)
        ),
    }

    per_run_pairing_ok = all(
        int(count) >= int(args.min_paired_seeds)
        for count in paired_seed_count_by_run.values()
    ) if paired_seed_count_by_run else True

    statistical_power["sufficient_for_reconnect_claim"] = bool(
        statistical_power["full_has_min_event_rows"] and per_run_pairing_ok
    )

    out: dict[str, Any] = {
        "runs_root": str(runs_root),
        "seeds": seeds,
        "runs": runs,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "comparison_vs_full": comparison_vs_full,
        "statistical_power": statistical_power,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    print(json.dumps({"output_json": str(output_path), "seeds": seeds, "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
