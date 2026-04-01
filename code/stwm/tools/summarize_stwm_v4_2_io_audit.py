from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import csv
import json
import statistics
from typing import Any

import numpy as np


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def read_gpu_util(path: Path) -> list[float]:
    vals: list[float] = []
    if not path.exists():
        return vals
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            raw = row.get("utilization_gpu", "").strip()
            if not raw:
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
    return vals


def read_disk_free(path: Path) -> list[float]:
    vals: list[float] = []
    if not path.exists():
        return vals
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            raw = row.get("disk_free_gb", "").strip()
            if not raw:
                continue
            try:
                vals.append(float(raw))
            except ValueError:
                continue
    return vals


def read_train_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def summarize_lane(lane_dir: Path, min_disk_free_gb: float) -> dict[str, Any]:
    gpu_trace = lane_dir / "gpu_usage_trace.csv"
    disk_trace = lane_dir / "disk_trace.csv"
    train_log = lane_dir / "train_log.jsonl"
    summary_json = lane_dir / "mini_val_summary.json"
    latest_ckpt = lane_dir / "checkpoints" / "latest.pt"
    best_ckpt = lane_dir / "checkpoints" / "best.pt"

    gpu_util = read_gpu_util(gpu_trace)
    disk_free = read_disk_free(disk_trace)
    train_rows = read_train_log(train_log)

    step_times = [float(row.get("step_time_s", 0.0)) for row in train_rows if float(row.get("step_time_s", 0.0)) > 0.0]
    data_ratios = [float(row.get("data_wait_ratio", 0.0)) for row in train_rows if "data_wait_ratio" in row]
    peak_memory = [float(row.get("gpu_peak_memory_gb", 0.0)) for row in train_rows if "gpu_peak_memory_gb" in row]

    util_median = float(statistics.median(gpu_util)) if gpu_util else 0.0
    step_p50 = percentile(step_times, 50.0)
    step_p95 = percentile(step_times, 95.0)
    step_ratio = float(step_p95 / step_p50) if step_p50 > 0 else 0.0

    spike_threshold = 1.5 * step_p50 if step_p50 > 0 else 0.0
    spike_share = 0.0
    if step_times and spike_threshold > 0:
        spike_share = float(sum(1 for x in step_times if x > spike_threshold) / len(step_times))

    no_io_sawtooth = bool(spike_share <= 0.10)
    disk_min = float(min(disk_free)) if disk_free else 0.0

    lane_ok = bool(
        util_median >= 85.0
        and step_ratio <= 1.5
        and no_io_sawtooth
        and disk_min >= min_disk_free_gb
        and latest_ckpt.exists()
        and best_ckpt.exists()
    )

    return {
        "lane_dir": str(lane_dir),
        "files": {
            "gpu_trace": str(gpu_trace),
            "disk_trace": str(disk_trace),
            "train_log": str(train_log),
            "summary": str(summary_json),
            "latest_checkpoint": str(latest_ckpt),
            "best_checkpoint": str(best_ckpt),
        },
        "metrics": {
            "gpu_utilization_median": util_median,
            "gpu_utilization_p95": percentile(gpu_util, 95.0),
            "step_time_p50_s": step_p50,
            "step_time_p95_s": step_p95,
            "step_time_p95_over_p50": step_ratio,
            "step_time_spike_share": spike_share,
            "data_wait_ratio_p50": percentile(data_ratios, 50.0),
            "data_wait_ratio_p95": percentile(data_ratios, 95.0),
            "gpu_peak_memory_gb_max": float(max(peak_memory) if peak_memory else 0.0),
            "disk_free_gb_min": disk_min,
            "steps_logged": len(step_times),
        },
        "checks": {
            "util_median_ge_85": bool(util_median >= 85.0),
            "step_p95_over_p50_le_1_5": bool(step_ratio <= 1.5),
            "no_io_sawtooth": no_io_sawtooth,
            "disk_free_ok": bool(disk_min >= min_disk_free_gb),
            "latest_exists": bool(latest_ckpt.exists()),
            "best_exists": bool(best_ckpt.exists()),
        },
        "lane_ok_for_scale_out": lane_ok,
    }


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize STWM V4.2 Phase1 IO/throughput audit")
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--min-disk-free-gb", type=float, default=50.0)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audit_root = Path(args.audit_root)

    lane_dirs = sorted([p for p in audit_root.iterdir() if p.is_dir() and p.name.startswith("lane")])
    if not lane_dirs:
        raise RuntimeError(f"no lane directories found under {audit_root}")

    lane_summaries = [summarize_lane(lane_dir, float(args.min_disk_free_gb)) for lane_dir in lane_dirs]
    can_expand = all(bool(item["lane_ok_for_scale_out"]) for item in lane_summaries)

    out = {
        "audit_root": str(audit_root),
        "min_disk_free_gb": float(args.min_disk_free_gb),
        "lanes": lane_summaries,
        "decision": {
            "can_expand_to_4_lanes": bool(can_expand),
            "reason": "all lanes passed utilization/stability/io/disk/checkpoint checks" if can_expand else "at least one lane failed scale-out checks",
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
