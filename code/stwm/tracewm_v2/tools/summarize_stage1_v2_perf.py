#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _classify_bottleneck(debug_stats: Dict[str, Any], dl_profile: Dict[str, Any]) -> str:
    step = debug_stats.get("timing_stats", {}).get("step_time", {})
    batch_wait = debug_stats.get("timing_stats", {}).get("batch_wait_time", {})
    h2d = debug_stats.get("timing_stats", {}).get("h2d_time", {})
    forward = debug_stats.get("timing_stats", {}).get("forward_time", {})
    backward = debug_stats.get("timing_stats", {}).get("backward_time", {})

    step_mean = float(step.get("mean", 0.0))
    if step_mean <= 0:
        return "mixed_bottleneck"

    wait_ratio = float(batch_wait.get("mean", 0.0)) / step_mean
    h2d_ratio = float(h2d.get("mean", 0.0)) / step_mean
    gpu_ratio = (float(forward.get("mean", 0.0)) + float(backward.get("mean", 0.0))) / step_mean

    rows = dl_profile.get("rows", []) if isinstance(dl_profile.get("rows", []), list) else []
    workers0 = [r for r in rows if int(r.get("config", {}).get("num_workers", 0)) == 0 and r.get("status") == "pass"]
    workersp = [r for r in rows if int(r.get("config", {}).get("num_workers", 0)) > 0 and r.get("status") == "pass"]
    best0 = max([float(r.get("batches_per_sec", 0.0)) for r in workers0], default=0.0)
    bestp = max([float(r.get("batches_per_sec", 0.0)) for r in workersp], default=0.0)
    dl_speedup = float(bestp / best0) if best0 > 0 else 1.0

    if wait_ratio > 0.35:
        if dl_speedup < 1.15:
            return "io_bound"
        return "cpu_dataloader_bound"
    if h2d_ratio > 0.25:
        return "h2d_bound"
    if gpu_ratio > 0.70:
        return "gpu_bound"
    return "mixed_bottleneck"


def parse_args() -> Any:
    parser = ArgumentParser(description="Summarize Stage1-v2 perf round")
    parser.add_argument("--preflight-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_preflight_20260408.json")
    parser.add_argument("--gpu-selection-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_gpu_selection_audit_20260408.json")
    parser.add_argument("--dataloader-profile-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_dataloader_profile_20260408.json")
    parser.add_argument("--gpu-telemetry-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_gpu_telemetry_20260408.json")
    parser.add_argument("--debug-timing-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_debug_small_20260408.json")
    parser.add_argument("--prototype-timing-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_prototype_220m_20260408.json")
    parser.add_argument("--nsys-status-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_nsys_profile_20260408.json")
    parser.add_argument("--perf-step-timing-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_perf_step_timing_20260408.json")
    parser.add_argument("--summary-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_perf_summary_20260408.json")
    parser.add_argument("--summary-md", default="/home/chen034/workspace/stwm/docs/TRACEWM_STAGE1_V2_PERF_RESULTS_20260408.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    preflight = _read_json(args.preflight_json)
    gpu_sel = _read_json(args.gpu_selection_json)
    dl_prof = _read_json(args.dataloader_profile_json)
    telemetry = _read_json(args.gpu_telemetry_json)
    debug_timing = _read_json(args.debug_timing_json)
    proto_timing = _read_json(args.prototype_timing_json) if Path(args.prototype_timing_json).exists() else {}
    nsys_status = _read_json(args.nsys_status_json) if Path(args.nsys_status_json).exists() else {"status": "unavailable"}

    combined_timing = {
        "generated_at_utc": now_iso(),
        "runs": {
            "stage1_v2_perf_debug_small": debug_timing,
            "stage1_v2_perf_prototype_220m": proto_timing,
        },
    }
    perf_timing_json = Path(args.perf_step_timing_json)
    perf_timing_json.parent.mkdir(parents=True, exist_ok=True)
    perf_timing_json.write_text(json.dumps(combined_timing, ensure_ascii=False, indent=2), encoding="utf-8")

    primary = _classify_bottleneck(debug_timing, dl_prof)

    selected_gpu = int(gpu_sel.get("selected_gpu_id", -1))
    telemetry_rows = telemetry.get("summary", []) if isinstance(telemetry.get("summary", []), list) else []
    sel_tele = next((r for r in telemetry_rows if int(r.get("gpu_id", -1)) == selected_gpu), {})

    evidence = [
        {
            "field": "preflight_pass",
            "value": bool(preflight.get("preflight_pass", False)),
            "source": str(args.preflight_json),
        },
        {
            "field": "selected_gpu_id",
            "value": selected_gpu,
            "source": str(args.gpu_selection_json),
        },
        {
            "field": "debug_step_time_mean",
            "value": float(debug_timing.get("timing_stats", {}).get("step_time", {}).get("mean", 0.0)),
            "source": str(args.debug_timing_json),
        },
        {
            "field": "debug_batch_wait_mean",
            "value": float(debug_timing.get("timing_stats", {}).get("batch_wait_time", {}).get("mean", 0.0)),
            "source": str(args.debug_timing_json),
        },
        {
            "field": "dataloader_best_batches_per_sec",
            "value": float(dl_prof.get("best_batches_per_sec", 0.0)),
            "source": str(args.dataloader_profile_json),
        },
        {
            "field": "selected_gpu_avg_util",
            "value": float(sel_tele.get("avg_gpu_util", 0.0)),
            "source": str(args.gpu_telemetry_json),
        },
        {
            "field": "selected_gpu_avg_mem_util",
            "value": float(sel_tele.get("avg_mem_util", 0.0)),
            "source": str(args.gpu_telemetry_json),
        },
    ]

    top_actions = [
        {
            "rank": 1,
            "action": "Use dataloader best_config from profile as default perf config",
            "expected_gain": "high",
        },
        {
            "rank": 2,
            "action": "Keep pin_memory=True and non_blocking H2D copies in single-GPU runs",
            "expected_gain": "high",
        },
        {
            "rank": 3,
            "action": "Pin single GPU by averaged window sampling and lease guard",
            "expected_gain": "medium-high",
        },
        {
            "rank": 4,
            "action": "Tune num_workers and prefetch_factor jointly for stable batches/sec",
            "expected_gain": "medium",
        },
        {
            "rank": 5,
            "action": "Use profiler traces to remove top-k expensive kernels or sync points",
            "expected_gain": "medium",
        },
    ]

    recommended_gpu_policy = {
        "mode": "single_gpu_only",
        "selection_rule": [
            "avg_gpu_util lowest",
            "avg_mem_util lowest",
            "active_compute_process_count lowest",
            "free_mem highest",
        ],
        "window": {
            "sample_count": int(gpu_sel.get("sample_count", 0)),
            "sample_interval_sec": float(gpu_sel.get("sample_interval_sec", 0.0)),
        },
        "memory_filter": {
            "required_mem_gb": float(gpu_sel.get("required_mem_gb", 0.0)),
            "safety_margin_gb": float(gpu_sel.get("safety_margin_gb", 0.0)),
        },
        "selected_gpu_id": selected_gpu,
    }

    payload = {
        "generated_at_utc": now_iso(),
        "primary_bottleneck": primary,
        "evidence": evidence,
        "top_5_actions": top_actions,
        "recommended_gpu_policy": recommended_gpu_policy,
        "paths": {
            "preflight": str(args.preflight_json),
            "gpu_selection": str(args.gpu_selection_json),
            "dataloader_profile": str(args.dataloader_profile_json),
            "gpu_telemetry": str(args.gpu_telemetry_json),
            "perf_step_timing": str(args.perf_step_timing_json),
            "nsys_status": str(args.nsys_status_json),
            "nsys_status_value": str(nsys_status.get("status", "unknown")),
        },
    }

    summary_json = Path(args.summary_json)
    summary_md = Path(args.summary_md)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# TRACEWM Stage1-v2 Perf Results",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- primary_bottleneck: {payload['primary_bottleneck']}",
        f"- selected_gpu_id: {recommended_gpu_policy['selected_gpu_id']}",
        "",
        "## Evidence",
    ]
    for ev in evidence:
        lines.append(f"- {ev['field']}: {ev['value']} (source={ev['source']})")

    lines.append("")
    lines.append("## Top 5 Actions")
    for act in top_actions:
        lines.append(f"- {act['rank']}. {act['action']} (expected_gain={act['expected_gain']})")

    lines.append("")
    lines.append("## Recommended GPU Policy")
    lines.append(f"- mode: {recommended_gpu_policy['mode']}")
    lines.append(f"- selected_gpu_id: {recommended_gpu_policy['selected_gpu_id']}")
    lines.append("- selection_rule:")
    for rule in recommended_gpu_policy["selection_rule"]:
        lines.append(f"  - {rule}")

    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-perf-summary] summary_json={summary_json}")
    print(f"[stage1-v2-perf-summary] summary_md={summary_md}")


if __name__ == "__main__":
    main()
