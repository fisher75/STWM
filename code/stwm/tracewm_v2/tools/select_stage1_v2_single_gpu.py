#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict
import json
import sys

from stwm.infra.gpu_lease import DEFAULT_LEASE_PATH, acquire_lease
from stwm.infra.gpu_selector import select_single_gpu


def parse_args() -> Any:
    parser = ArgumentParser(description="Select single GPU for Stage1-v2 perf round on shared cluster")
    parser.add_argument("--required-mem-gb", type=float, default=40.0)
    parser.add_argument("--safety-margin-gb", type=float, default=8.0)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--sample-interval-sec", type=float, default=2.0)
    parser.add_argument("--lease-path", default=str(DEFAULT_LEASE_PATH))
    parser.add_argument("--lease-owner", default="tracewm_stage1_v2_perf_20260408")
    parser.add_argument("--lease-ttl-sec", type=int, default=8 * 3600)
    parser.add_argument("--report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_gpu_selection_audit_20260408.json")
    parser.add_argument("--report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_GPU_SELECTION_AUDIT_20260408.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    payload = select_single_gpu(
        required_mem_gb=float(args.required_mem_gb),
        safety_margin_gb=float(args.safety_margin_gb),
        sample_count=int(args.sample_count),
        interval_sec=float(args.sample_interval_sec),
        lease_path=str(args.lease_path),
    )

    selected_gpu_id = int(payload.get("selected_gpu_id", -1))
    lease_rec: Dict[str, Any] = {}
    if selected_gpu_id >= 0:
        lease_rec = acquire_lease(
            gpu_id=selected_gpu_id,
            owner=str(args.lease_owner),
            ttl_seconds=int(args.lease_ttl_sec),
            lease_path=str(args.lease_path),
        )

    payload["lease"] = lease_rec
    payload["lease_path"] = str(args.lease_path)

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stage1-v2 GPU Selection Audit",
        "",
        f"- selected_gpu_id: {selected_gpu_id}",
        f"- required_mem_gb: {payload.get('required_mem_gb', 0.0)}",
        f"- safety_margin_gb: {payload.get('safety_margin_gb', 0.0)}",
        f"- sample_count: {payload.get('sample_count', 0)}",
        f"- sample_interval_sec: {payload.get('sample_interval_sec', 0.0)}",
        f"- lease_id: {lease_rec.get('lease_id', '')}",
        "",
        "| gpu_id | free_mem_gb | avg_gpu_util | avg_mem_util | active_compute_process_count | selected | selected_reason |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]

    gpus = payload.get("gpus", []) if isinstance(payload.get("gpus", []), list) else []
    for row in sorted(gpus, key=lambda x: int(x.get("gpu_id", -1))):
        lines.append(
            "| {gpu} | {free:.2f} | {gpuu:.2f} | {memu:.2f} | {proc} | {sel} | {reason} |".format(
                gpu=int(row.get("gpu_id", -1)),
                free=float(row.get("free_mem_gb", 0.0)),
                gpuu=float(row.get("avg_gpu_util", 0.0)),
                memu=float(row.get("avg_mem_util", 0.0)),
                proc=int(row.get("active_compute_process_count", 0)),
                sel=bool(row.get("selected", False)),
                reason=str(row.get("selected_reason", "")),
            )
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-gpu-select] report_json={report_json}")
    print(f"[stage1-v2-gpu-select] report_md={report_md}")
    print(f"[stage1-v2-gpu-select] selected_gpu_id={selected_gpu_id}")
    print(f"[stage1-v2-gpu-select] lease_id={lease_rec.get('lease_id', '')}")

    if selected_gpu_id < 0:
        sys.exit(23)


if __name__ == "__main__":
    main()
