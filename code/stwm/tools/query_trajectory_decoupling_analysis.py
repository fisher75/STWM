from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
from typing import Any

import numpy as np


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Posthoc query-trajectory decoupling analysis for STWM V4.2")
    parser.add_argument("--runs-root", default="/home/chen034/workspace/stwm/outputs/training/stwm_v4_2_minival_seed42")
    parser.add_argument("--runs", default="full_v4_2,wo_semantics_v4_2,wo_identity_v4_2")
    parser.add_argument("--close-threshold", type=float, default=0.002)
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/reports/stwm_v4_2_query_decoupling_seed42.json")
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

    # Higher score means less proxy-like coupling.
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
        "proxy_like": proxy_like,
    }


def main() -> None:
    args = build_parser().parse_args()
    runs_root = Path(args.runs_root)
    runs = [r.strip() for r in str(args.runs).split(",") if r.strip()]

    out: dict[str, Any] = {
        "runs_root": str(runs_root),
        "runs": runs,
        "close_threshold": float(args.close_threshold),
        "per_run": {},
        "comparison_vs_full": {},
    }

    for run in runs:
        rows = _read_rows(runs_root / run / "train_log.jsonl")
        out["per_run"][run] = _analyze(rows, close_threshold=float(args.close_threshold))

    full = out["per_run"].get("full_v4_2")
    if isinstance(full, dict):
        for run in runs:
            if run == "full_v4_2":
                continue
            cur = out["per_run"][run]
            out["comparison_vs_full"][run] = {
                "delta_corr_abs": float(abs(cur["pearson_corr"]) - abs(full["pearson_corr"])),
                "delta_close_ratio": float(cur["close_ratio"] - full["close_ratio"]),
                "delta_decoupling_score": float(cur["decoupling_score"] - full["decoupling_score"]),
                "full_less_proxy_like": bool((not full["proxy_like"]) and bool(cur["proxy_like"])),
            }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))

    print(json.dumps({"output_json": str(output_path), "runs": runs}, indent=2))


if __name__ == "__main__":
    main()
