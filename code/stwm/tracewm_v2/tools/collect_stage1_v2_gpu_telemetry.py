#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json
import signal
import time

from stwm.infra.gpu_telemetry import snapshot_gpu_telemetry


_STOP = False


def _handle_stop(_sig: int, _frame: Any) -> None:
    global _STOP
    _STOP = True


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> Any:
    parser = ArgumentParser(description="Collect lightweight GPU telemetry for Stage1-v2 perf run")
    parser.add_argument("--output-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_gpu_telemetry_20260408.json")
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--tag", default="tracewm_stage1_v2_perf_20260408")
    return parser.parse_args()


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / float(len(values)))


def main() -> None:
    args = parse_args()
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    start = time.perf_counter()
    samples: List[Dict[str, Any]] = []

    while True:
        if _STOP:
            break
        if float(args.max_seconds) > 0 and (time.perf_counter() - start) >= float(args.max_seconds):
            break

        samples.append(snapshot_gpu_telemetry(prefer_nvml=True))
        time.sleep(max(float(args.interval_sec), 0.1))

    per_gpu: Dict[int, Dict[str, Any]] = {}
    for snap in samples:
        for row in snap.get("gpus", []):
            gpu_id = int(row.get("gpu_id", -1))
            if gpu_id < 0:
                continue
            rec = per_gpu.setdefault(
                gpu_id,
                {
                    "gpu_id": gpu_id,
                    "gpu_util": [],
                    "mem_util": [],
                    "memory_used_gb": [],
                    "power_draw_w": [],
                },
            )
            rec["gpu_util"].append(float(row.get("gpu_util", 0.0)))
            rec["mem_util"].append(float(row.get("mem_util", 0.0)))
            rec["memory_used_gb"].append(float(row.get("memory_used_gb", 0.0)))
            power = float(row.get("power_draw_w", -1.0))
            if power >= 0:
                rec["power_draw_w"].append(power)

    summary_rows = []
    for gpu_id in sorted(per_gpu.keys()):
        rec = per_gpu[gpu_id]
        summary_rows.append(
            {
                "gpu_id": gpu_id,
                "avg_gpu_util": _mean(rec["gpu_util"]),
                "avg_mem_util": _mean(rec["mem_util"]),
                "avg_memory_used_gb": _mean(rec["memory_used_gb"]),
                "avg_power_draw_w": _mean(rec["power_draw_w"]),
                "sample_count": int(len(rec["gpu_util"])),
            }
        )

    payload = {
        "generated_at_utc": now_iso(),
        "tag": str(args.tag),
        "interval_sec": float(args.interval_sec),
        "sample_count": int(len(samples)),
        "samples": samples,
        "summary": summary_rows,
    }

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stage1-v2-gpu-telemetry] output={out}")
    print(f"[stage1-v2-gpu-telemetry] sample_count={len(samples)}")


if __name__ == "__main__":
    main()
