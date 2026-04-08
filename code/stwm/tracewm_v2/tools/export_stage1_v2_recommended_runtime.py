#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def parse_args() -> Any:
    parser = ArgumentParser(description="Export Stage1-v2 single-GPU recommended runtime")
    parser.add_argument("--gpu-selection-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_gpu_selection_audit_20260408.json")
    parser.add_argument("--dataloader-profile-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_dataloader_profile_20260408.json")
    parser.add_argument("--preflight-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_220m_preflight_20260408.json")
    parser.add_argument("--debug-summary-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_perf_debug_small_20260408.json")
    parser.add_argument("--perf-summary-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_perf_summary_20260408.json")
    parser.add_argument("--report-json", default="/home/chen034/workspace/stwm/reports/stage1_v2_recommended_runtime_20260408.json")
    parser.add_argument("--report-md", default="/home/chen034/workspace/stwm/docs/STAGE1_V2_RECOMMENDED_RUNTIME_20260408.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_sel = _read_json(args.gpu_selection_json)
    dl_prof = _read_json(args.dataloader_profile_json)
    preflight = _read_json(args.preflight_json)
    debug_summary = _read_json(args.debug_summary_json)
    perf_summary = _read_json(args.perf_summary_json)

    best_cfg = dl_prof.get("best_config", {}) if isinstance(dl_prof.get("best_config", {}), dict) else {}

    # Hardening policy pins recommended runtime defaults to the validated best config
    # from the previous perf round. This keeps runtime defaults stable and explicit.
    recommended_num_workers = 8
    recommended_pin_memory = True
    recommended_persistent_workers = True
    recommended_prefetch_factor = 4

    debug_args = debug_summary.get("args", {}) if isinstance(debug_summary.get("args", {}), dict) else {}
    recommended_batch_size_debug_small = int(debug_args.get("batch_size", 2) or 2)
    recommended_batch_size_prototype_220m = int(preflight.get("selected_batch_size", 1) or 1)

    selected_gpu_policy = perf_summary.get("recommended_gpu_policy", {})
    if not isinstance(selected_gpu_policy, dict) or not selected_gpu_policy:
        selected_gpu_policy = {
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
            "selected_gpu_id": int(gpu_sel.get("selected_gpu_id", -1)),
        }

    payload = {
        "generated_at_utc": now_iso(),
        "selected_gpu_policy": selected_gpu_policy,
        "required_mem_gb": float(gpu_sel.get("required_mem_gb", 40.0)),
        "safety_margin_gb": float(gpu_sel.get("safety_margin_gb", 8.0)),
        "recommended_num_workers": int(recommended_num_workers),
        "recommended_pin_memory": bool(recommended_pin_memory),
        "recommended_persistent_workers": bool(recommended_persistent_workers),
        "recommended_prefetch_factor": int(recommended_prefetch_factor),
        "recommended_batch_size_debug_small": int(recommended_batch_size_debug_small),
        "recommended_batch_size_prototype_220m": int(recommended_batch_size_prototype_220m),
        "single_gpu_only": True,
        "profile_best_config_reference": best_cfg,
        "notes": [
            "recommended runtime defaults only",
            "not scientific variables",
            "recommended runtime defaults are pinned to 8/true/true/4 for hardening stability",
        ],
    }

    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Stage1-v2 Recommended Runtime",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        f"- selected_gpu_policy: {json.dumps(payload['selected_gpu_policy'], ensure_ascii=True)}",
        f"- required_mem_gb: {payload['required_mem_gb']}",
        f"- safety_margin_gb: {payload['safety_margin_gb']}",
        f"- recommended_num_workers: {payload['recommended_num_workers']}",
        f"- recommended_pin_memory: {payload['recommended_pin_memory']}",
        f"- recommended_persistent_workers: {payload['recommended_persistent_workers']}",
        f"- recommended_prefetch_factor: {payload['recommended_prefetch_factor']}",
        f"- recommended_batch_size_debug_small: {payload['recommended_batch_size_debug_small']}",
        f"- recommended_batch_size_prototype_220m: {payload['recommended_batch_size_prototype_220m']}",
        f"- single_gpu_only: {payload['single_gpu_only']}",
        "",
        "Recommended defaults are runtime settings only and do not change scientific logic.",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stage1-v2-runtime-export] report_json={report_json}")
    print(f"[stage1-v2-runtime-export] report_md={report_md}")


if __name__ == "__main__":
    main()