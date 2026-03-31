from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import numpy as np


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Posthoc occlusion/reconnect bucket analysis for STWM V4.2")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_seed42")
    parser.add_argument("--runs", default="full_v4_2,wo_semantics_v4_2,wo_identity_v4_2")
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_occlusion_reconnect_seed42.json")
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
    runs = [r.strip() for r in str(args.runs).split(",") if r.strip()]

    out: dict[str, Any] = {
        "runs_root": str(runs_root),
        "runs": runs,
        "per_run": {},
        "comparison_vs_full": {},
    }

    for run in runs:
        rows = _read_rows(runs_root / run / "train_log.jsonl")
        out["per_run"][run] = _stats(rows)

    full = out["per_run"].get("full_v4_2")
    if isinstance(full, dict):
        for run in runs:
            if run == "full_v4_2":
                continue
            cur = out["per_run"][run]
            out["comparison_vs_full"][run] = {
                "delta_event_row_ratio": float(cur["event_row_ratio"] - full["event_row_ratio"]),
                "delta_reconnect_success_rate": float(cur["reconnect_success_rate"] - full["reconnect_success_rate"]),
                "delta_reconnect_min_error_mean": float(cur["reconnect_min_error_mean"] - full["reconnect_min_error_mean"]),
                "delta_query_error_on_events": float(cur["query_localization_error_on_events"] - full["query_localization_error_on_events"]),
                "delta_traj_l1_on_events": float(cur["trajectory_l1_on_events"] - full["trajectory_l1_on_events"]),
            }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    print(json.dumps({"output_json": str(output_path), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
