from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple
import time

from .gpu_lease import is_gpu_leased
from .gpu_telemetry import snapshot_gpu_telemetry


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def _aggregate_samples(samples: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_gpu: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
        "gpu_util": [],
        "mem_util": [],
        "free_mem_gb": [],
        "active_compute_process_count": [],
        "name": "",
    })

    for snap in samples:
        for row in snap.get("gpus", []):
            gpu_id = int(row.get("gpu_id", -1))
            if gpu_id < 0:
                continue
            acc = by_gpu[gpu_id]
            acc["name"] = str(row.get("name", ""))
            acc["gpu_util"].append(float(row.get("gpu_util", 0.0)))
            acc["mem_util"].append(float(row.get("mem_util", 0.0)))
            acc["free_mem_gb"].append(float(row.get("free_mem_gb", 0.0)))
            acc["active_compute_process_count"].append(float(row.get("active_compute_process_count", 0.0)))

    out: Dict[int, Dict[str, Any]] = {}
    for gpu_id, acc in by_gpu.items():
        out[gpu_id] = {
            "gpu_id": int(gpu_id),
            "name": str(acc.get("name", "")),
            "avg_gpu_util": _avg(list(acc["gpu_util"])),
            "avg_mem_util": _avg(list(acc["mem_util"])),
            "free_mem_gb": _avg(list(acc["free_mem_gb"])),
            "active_compute_process_count": int(round(_avg(list(acc["active_compute_process_count"])))),
            "sample_count": int(len(acc["gpu_util"])),
        }
    return out


def _candidate_sort_key(row: Dict[str, Any]) -> Tuple[float, float, int, float]:
    return (
        float(row.get("avg_gpu_util", 0.0)),
        float(row.get("avg_mem_util", 0.0)),
        int(row.get("active_compute_process_count", 0)),
        -float(row.get("free_mem_gb", 0.0)),
    )


def select_single_gpu(
    required_mem_gb: float,
    safety_margin_gb: float,
    sample_count: int = 12,
    interval_sec: float = 2.0,
    lease_path: str | None = None,
) -> Dict[str, Any]:
    required = float(required_mem_gb)
    margin = float(safety_margin_gb)

    snapshots: List[Dict[str, Any]] = []
    for i in range(max(int(sample_count), 1)):
        snapshots.append(snapshot_gpu_telemetry(prefer_nvml=True))
        if i + 1 < int(sample_count):
            time.sleep(max(float(interval_sec), 0.0))

    aggregated = _aggregate_samples(snapshots)
    gpu_ids = sorted(aggregated.keys())

    rows: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []

    for gpu_id in gpu_ids:
        row = dict(aggregated[gpu_id])
        leased = is_gpu_leased(gpu_id=gpu_id, lease_path=lease_path) if lease_path else False
        enough_mem = float(row.get("free_mem_gb", 0.0)) >= (required + margin)

        selected_reason = ""
        if not enough_mem:
            selected_reason = "filtered_insufficient_free_mem"
        elif leased:
            selected_reason = "filtered_local_lease"
        else:
            selected_reason = "candidate"
            candidates.append(row)

        row["selected"] = False
        row["selected_reason"] = selected_reason
        row["required_mem_gb"] = float(required)
        row["safety_margin_gb"] = float(margin)
        row["leased"] = bool(leased)
        rows.append(row)

    candidates = sorted(candidates, key=_candidate_sort_key)

    selected_gpu_id = -1
    if candidates:
        selected_gpu_id = int(candidates[0]["gpu_id"])

    final_rows: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda x: int(x.get("gpu_id", -1))):
        if int(row.get("gpu_id", -1)) == selected_gpu_id:
            row["selected"] = True
            row["selected_reason"] = "best_rank_after_window_sampling"
        elif row.get("selected_reason") == "candidate":
            row["selected_reason"] = "candidate_not_top_rank"
        final_rows.append(row)

    ranking = [int(x.get("gpu_id", -1)) for x in candidates]

    return {
        "generated_at_utc": now_iso(),
        "required_mem_gb": float(required),
        "safety_margin_gb": float(margin),
        "sample_count": int(sample_count),
        "sample_interval_sec": float(interval_sec),
        "selected_gpu_id": int(selected_gpu_id),
        "candidate_ranking": ranking,
        "gpus": final_rows,
    }
