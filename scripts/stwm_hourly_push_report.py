#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path("/home/chen034/workspace/stwm")
PROBE_SCRIPT = Path("/tmp/stwm_lane_probe_now.sh")
OUT_DIR = ROOT / "outputs/monitoring/stwm_hourly_push"
STATE_FILE = OUT_DIR / "state.json"
LATEST_FILE = OUT_DIR / "latest.md"
RAW_FILE = OUT_DIR / "latest_probe_raw.txt"


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = proc.stderr.strip() or "unknown error"
        raise RuntimeError(f"Command failed: {' '.join(cmd)}; {err}")
    return proc.stdout


def to_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def to_int(value: str) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def parse_probe_output(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    lanes: dict[str, Any] = {}
    overall: dict[str, Any] = {}

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("overall_eta_hours_low="):
            overall["low_h"] = to_float(line.split("=", 1)[1])
            continue
        if line.startswith("overall_eta_hours_high="):
            overall["high_h"] = to_float(line.split("=", 1)[1])
            continue

        if not line.startswith("lane"):
            continue

        parts = line.split()
        if len(parts) >= 2 and parts[0] == "lane" and parts[1] == "state":
            continue

        lane_name = parts[0]
        state = parts[1] if len(parts) > 1 else "UNKNOWN"

        lane_info: dict[str, Any] = {
            "state": state,
        }

        if state == "RUNNING" and len(parts) >= 23:
            lane_info.update(
                {
                    "scale": parts[2],
                    "seed": parts[3],
                    "gpu": parts[4],
                    "pid": parts[5],
                    "cpu_pct": to_float(parts[6]),
                    "run_name": parts[7],
                    "pending": to_int(parts[8]),
                    "step": to_int(parts[9]),
                    "step_time_last_s": to_float(parts[10]),
                    "step_time_avg200_s": to_float(parts[11]),
                    "data_wait_last": to_float(parts[12]),
                    "data_wait_avg200": to_float(parts[13]),
                    "step_delta20": to_int(parts[14]),
                    "eta5_h": to_float(parts[15]),
                    "eta8_h": to_float(parts[16]),
                    "eta_lane_low_h": to_float(parts[17]),
                    "eta_lane_high_h": to_float(parts[18]),
                    "vram_mib": to_float(parts[19]),
                    "pmon_sm_avg": to_float(parts[20]),
                    "pmon_mem_avg": to_float(parts[21]),
                    "pmon_sm_n": to_int(parts[22]),
                }
            )

        lanes[lane_name] = lane_info

    return lanes, overall


def get_host_metrics() -> dict[str, Any]:
    uptime_out = run_cmd(["uptime"]).strip()
    m = re.search(r"load average:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)", uptime_out)
    load1 = load5 = load15 = None
    if m:
        load1 = to_float(m.group(1))
        load5 = to_float(m.group(2))
        load15 = to_float(m.group(3))

    free_out = run_cmd(["free", "-b"])
    mem_total = mem_used = mem_available = None
    for line in free_out.splitlines():
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) >= 7:
                mem_total = to_int(parts[1])
                mem_used = to_int(parts[2])
                mem_available = to_int(parts[6])
            break

    gpu_out = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    gpus: dict[str, Any] = {}
    for line in gpu_out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, util, mem_used_mib, mem_total_mib = parts
        gpus[idx] = {
            "util_pct": to_float(util),
            "mem_used_mib": to_float(mem_used_mib),
            "mem_total_mib": to_float(mem_total_mib),
        }

    return {
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "mem_total_gib": (mem_total / (1024**3)) if mem_total else None,
        "mem_used_gib": (mem_used / (1024**3)) if mem_used else None,
        "mem_available_gib": (mem_available / (1024**3)) if mem_available else None,
        "gpus": gpus,
    }


def load_prev_state() -> dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def fmt(value: float | int | None, ndigits: int = 2) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{ndigits}f}"


def signed(value: float | None, ndigits: int = 2) -> str:
    if value is None:
        return "NA"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.{ndigits}f}"


def compute_adjusted_eta_drift(cur_eta: float | None, prev_eta: float | None, elapsed_h: float | None) -> float | None:
    if cur_eta is None or prev_eta is None or elapsed_h is None:
        return None
    expected_now = max(prev_eta - elapsed_h, 0.0)
    return cur_eta - expected_now


def build_report(cur: dict[str, Any], prev: dict[str, Any] | None) -> str:
    lines: list[str] = []

    elapsed_h: float | None = None
    if prev and isinstance(prev.get("ts_epoch"), (int, float)):
        elapsed_h = (cur["ts_epoch"] - float(prev["ts_epoch"])) / 3600.0

    lines.append("# STWM Hourly Push")
    lines.append(f"- generated_at: {cur['ts_human']}")
    if elapsed_h is None:
        lines.append("- interval_h: baseline")
        lines.append("- note: first baseline snapshot; drift values will appear next run")
    else:
        lines.append(f"- interval_h: {elapsed_h:.3f}")

    host = cur["host"]
    prev_host = prev.get("host", {}) if prev else {}

    mem_used_delta = None
    load1_delta = None
    if prev:
        if host.get("mem_used_gib") is not None and prev_host.get("mem_used_gib") is not None:
            mem_used_delta = host["mem_used_gib"] - prev_host["mem_used_gib"]
        if host.get("load1") is not None and prev_host.get("load1") is not None:
            load1_delta = host["load1"] - prev_host["load1"]

    lines.append("")
    lines.append("## Resource Change")
    lines.append(
        "- host_mem_used_gib: "
        f"{fmt(host.get('mem_used_gib'))} ({signed(mem_used_delta)} vs prev)"
    )
    lines.append(
        "- host_mem_available_gib: "
        f"{fmt(host.get('mem_available_gib'))}"
    )
    lines.append(
        "- host_load1: "
        f"{fmt(host.get('load1'))} ({signed(load1_delta)} vs prev)"
    )

    lines.append("")
    lines.append("## Running Lanes")

    for lane in ["lane0", "lane1", "lane2", "lane3"]:
        lane_cur = cur["lanes"].get(lane, {"state": "UNKNOWN"})
        state = lane_cur.get("state", "UNKNOWN")

        if state != "RUNNING":
            lines.append(f"- {lane}: {state}")
            continue

        lane_prev = prev.get("lanes", {}).get(lane, {}) if prev else {}

        step_gain = step_per_h = None
        if elapsed_h and lane_prev.get("state") == "RUNNING":
            if lane_cur.get("step") is not None and lane_prev.get("step") is not None:
                step_gain = lane_cur["step"] - lane_prev["step"]
                step_per_h = step_gain / elapsed_h

        eta5_drift = compute_adjusted_eta_drift(
            lane_cur.get("eta5_h"), lane_prev.get("eta5_h"), elapsed_h
        )
        eta8_drift = compute_adjusted_eta_drift(
            lane_cur.get("eta8_h"), lane_prev.get("eta8_h"), elapsed_h
        )

        cpu_delta = vram_delta = pmon_sm_delta = None
        if prev and lane_prev.get("state") == "RUNNING":
            if lane_cur.get("cpu_pct") is not None and lane_prev.get("cpu_pct") is not None:
                cpu_delta = lane_cur["cpu_pct"] - lane_prev["cpu_pct"]
            if lane_cur.get("vram_mib") is not None and lane_prev.get("vram_mib") is not None:
                vram_delta = lane_cur["vram_mib"] - lane_prev["vram_mib"]
            if lane_cur.get("pmon_sm_avg") is not None and lane_prev.get("pmon_sm_avg") is not None:
                pmon_sm_delta = lane_cur["pmon_sm_avg"] - lane_prev["pmon_sm_avg"]

        gpu_idx = lane_cur.get("gpu")
        gpu_cur = host.get("gpus", {}).get(str(gpu_idx), {})
        gpu_prev = prev_host.get("gpus", {}).get(str(gpu_idx), {}) if prev else {}
        gpu_util_delta = gpu_mem_delta = None
        if prev and gpu_cur and gpu_prev:
            if gpu_cur.get("util_pct") is not None and gpu_prev.get("util_pct") is not None:
                gpu_util_delta = gpu_cur["util_pct"] - gpu_prev["util_pct"]
            if gpu_cur.get("mem_used_mib") is not None and gpu_prev.get("mem_used_mib") is not None:
                gpu_mem_delta = gpu_cur["mem_used_mib"] - gpu_prev["mem_used_mib"]

        lines.append(
            f"- {lane}: scale={lane_cur.get('scale')} seed={lane_cur.get('seed')} run={lane_cur.get('run_name')} "
            f"step={fmt(lane_cur.get('step'), 0)} (gain={fmt(step_gain, 0)}, speed={fmt(step_per_h)} step/h, delta20={fmt(lane_cur.get('step_delta20'), 0)})"
        )
        lines.append(
            f"  cpu_pct={fmt(lane_cur.get('cpu_pct'))} ({signed(cpu_delta)}), "
            f"vram_mib={fmt(lane_cur.get('vram_mib'))} ({signed(vram_delta, 0)}), "
            f"pmon_sm_avg={fmt(lane_cur.get('pmon_sm_avg'))} ({signed(pmon_sm_delta)})"
        )
        lines.append(
            f"  gpu{gpu_idx}_util_pct={fmt(gpu_cur.get('util_pct'))} ({signed(gpu_util_delta)}), "
            f"gpu{gpu_idx}_mem_used_mib={fmt(gpu_cur.get('mem_used_mib'))} ({signed(gpu_mem_delta, 0)})"
        )
        lines.append(
            f"  eta5_h={fmt(lane_cur.get('eta5_h'))} (drift={signed(eta5_drift)}), "
            f"eta8_h={fmt(lane_cur.get('eta8_h'))} (drift={signed(eta8_drift)})"
        )

    lines.append("")
    lines.append("## Queue ETA Drift")

    overall_cur = cur.get("overall", {})
    overall_prev = prev.get("overall", {}) if prev else {}
    low_drift = compute_adjusted_eta_drift(
        overall_cur.get("low_h"), overall_prev.get("low_h"), elapsed_h
    )
    high_drift = compute_adjusted_eta_drift(
        overall_cur.get("high_h"), overall_prev.get("high_h"), elapsed_h
    )

    lines.append(
        f"- overall_low_h={fmt(overall_cur.get('low_h'))} (drift={signed(low_drift)})"
    )
    lines.append(
        f"- overall_high_h={fmt(overall_cur.get('high_h'))} (drift={signed(high_drift)})"
    )

    lines.append("")
    lines.append("## Notes")
    lines.append("- Drift definition: current ETA minus expected ETA after subtracting elapsed hours.")
    lines.append("- Positive drift means slower than expected; negative means faster than expected.")

    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PROBE_SCRIPT.exists():
        raise FileNotFoundError(f"Probe script missing: {PROBE_SCRIPT}")

    probe_output = run_cmd(["bash", str(PROBE_SCRIPT)])
    RAW_FILE.write_text(probe_output, encoding="utf-8")

    lanes, overall = parse_probe_output(probe_output)
    now_epoch = int(time.time())
    now_human = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    cur_state: dict[str, Any] = {
        "ts_epoch": now_epoch,
        "ts_human": now_human,
        "lanes": lanes,
        "overall": overall,
        "host": get_host_metrics(),
    }

    prev_state = load_prev_state()
    report = build_report(cur_state, prev_state)

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M")
    history_file = OUT_DIR / f"report_{stamp}.md"
    history_file.write_text(report, encoding="utf-8")
    LATEST_FILE.write_text(report, encoding="utf-8")
    STATE_FILE.write_text(json.dumps(cur_state, ensure_ascii=True, indent=2), encoding="utf-8")

    print(report, end="")
    print(f"history_file={history_file}")
    print(f"latest_file={LATEST_FILE}")


if __name__ == "__main__":
    main()
