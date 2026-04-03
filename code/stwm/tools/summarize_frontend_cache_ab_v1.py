from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import time
from typing import Any


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Summarize frontend cache pilot A/B metrics and decision")
    parser.add_argument("--raw-summary", required=True)
    parser.add_argument("--frontend-summary", required=True)
    parser.add_argument("--monitor-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text())


def main() -> None:
    args = build_parser().parse_args()

    raw = _load_json(args.raw_summary)
    frontend = _load_json(args.frontend_summary)
    monitor = _load_json(args.monitor_report)

    raw_rt = raw.get("runtime", {}) if isinstance(raw, dict) else {}
    fe_rt = frontend.get("runtime", {}) if isinstance(frontend, dict) else {}

    raw_step_p50 = float(raw_rt.get("step_time_p50_s", 0.0))
    fe_step_p50 = float(fe_rt.get("step_time_p50_s", 0.0))
    raw_step_p95 = float(raw_rt.get("step_time_p95_s", 0.0))
    fe_step_p95 = float(fe_rt.get("step_time_p95_s", 0.0))

    raw_data_p50 = float(raw_rt.get("data_time_p50_s", 0.0))
    fe_data_p50 = float(fe_rt.get("data_time_p50_s", 0.0))

    raw_wait_p50 = float(raw_rt.get("data_wait_ratio_p50", 0.0))
    fe_wait_p50 = float(fe_rt.get("data_wait_ratio_p50", 0.0))

    step_ratio = fe_step_p50 / max(1e-9, raw_step_p50)
    data_ratio = fe_data_p50 / max(1e-9, raw_data_p50)
    wait_delta_abs = fe_wait_p50 - raw_wait_p50

    monitor_raw = monitor.get("raw", {}) if isinstance(monitor, dict) else {}
    monitor_fe = monitor.get("frontend_cache", {}) if isinstance(monitor, dict) else {}

    raw_exit_ok = bool(monitor_raw.get("exit_code", 1) == 0)
    fe_exit_ok = bool(monitor_fe.get("exit_code", 1) == 0)
    stability_ok = raw_exit_ok and fe_exit_ok

    cpu_raw = float(monitor_raw.get("cpu_pct_mean", 0.0))
    cpu_fe = float(monitor_fe.get("cpu_pct_mean", 0.0))
    gpu_sm_raw = float(monitor_raw.get("gpu_sm_util_mean", 0.0))
    gpu_sm_fe = float(monitor_fe.get("gpu_sm_util_mean", 0.0))

    go_by_wait = (raw_wait_p50 - fe_wait_p50) >= 0.15
    go_by_step = (1.0 - step_ratio) >= 0.20
    go = stability_ok and (go_by_wait or go_by_step)

    decision = {
        "go": bool(go),
        "reason": "meets_throughput_threshold_and_stability" if go else "threshold_or_stability_not_met",
        "criteria": {
            "wait_ratio_abs_drop_ge_0p15": bool(go_by_wait),
            "step_time_reduction_ge_20pct": bool(go_by_step),
            "stability_ok": bool(stability_ok),
        },
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "raw_summary": str(args.raw_summary),
            "frontend_summary": str(args.frontend_summary),
            "monitor_report": str(args.monitor_report),
        },
        "ab_12step": {
            "raw": {
                "step_time_p50_s": raw_step_p50,
                "step_time_p95_s": raw_step_p95,
                "data_time_p50_s": raw_data_p50,
                "data_wait_ratio_p50": raw_wait_p50,
                "gpu_peak_memory_gb_max": float(raw_rt.get("gpu_peak_memory_gb_max", 0.0)),
            },
            "frontend_cache": {
                "step_time_p50_s": fe_step_p50,
                "step_time_p95_s": fe_step_p95,
                "data_time_p50_s": fe_data_p50,
                "data_wait_ratio_p50": fe_wait_p50,
                "gpu_peak_memory_gb_max": float(fe_rt.get("gpu_peak_memory_gb_max", 0.0)),
            },
            "delta": {
                "step_time_p50_ratio": float(step_ratio),
                "step_time_p50_reduction_pct": float((1.0 - step_ratio) * 100.0),
                "data_time_p50_ratio": float(data_ratio),
                "data_time_p50_reduction_pct": float((1.0 - data_ratio) * 100.0),
                "data_wait_ratio_abs_drop": float(raw_wait_p50 - fe_wait_p50),
                "data_wait_ratio_delta": float(wait_delta_abs),
            },
        },
        "ab_monitor_6step": {
            "raw": {
                "exit_ok": bool(raw_exit_ok),
                "cpu_pct_mean": float(cpu_raw),
                "gpu_sm_util_mean": float(gpu_sm_raw),
                "gpu_used_mem_mib_mean": float(monitor_raw.get("gpu_used_mem_mib_mean", 0.0)),
            },
            "frontend_cache": {
                "exit_ok": bool(fe_exit_ok),
                "cpu_pct_mean": float(cpu_fe),
                "gpu_sm_util_mean": float(gpu_sm_fe),
                "gpu_used_mem_mib_mean": float(monitor_fe.get("gpu_used_mem_mib_mean", 0.0)),
            },
            "delta": {
                "cpu_pct_mean_delta": float(cpu_fe - cpu_raw),
                "gpu_sm_util_mean_delta": float(gpu_sm_fe - gpu_sm_raw),
                "gpu_used_mem_mib_mean_delta": float(
                    float(monitor_fe.get("gpu_used_mem_mib_mean", 0.0))
                    - float(monitor_raw.get("gpu_used_mem_mib_mean", 0.0))
                ),
            },
        },
        "decision": decision,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(str(out_path))


if __name__ == "__main__":
    main()
