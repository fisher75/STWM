from __future__ import annotations

from datetime import datetime, timezone
from shutil import which
from typing import Any, Dict, List
import subprocess


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(raw: str, default: int = 0) -> int:
    try:
        return int(float(raw.strip()))
    except Exception:
        return int(default)


def _safe_float(raw: str, default: float = 0.0) -> float:
    try:
        return float(raw.strip())
    except Exception:
        return float(default)


def _run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()


def _query_nvidia_smi_rows() -> List[Dict[str, Any]]:
    if which("nvidia-smi") is None:
        return []

    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw",
        "--format=csv,noheader,nounits",
    ]
    raw = _run(query)

    rows: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        idx = _safe_int(parts[0], -1)
        total_mb = _safe_float(parts[3], 0.0)
        used_mb = _safe_float(parts[4], 0.0)
        free_mb = _safe_float(parts[5], 0.0)
        rows.append(
            {
                "gpu_id": idx,
                "name": str(parts[1]),
                "uuid": str(parts[2]),
                "memory_total_gb": float(total_mb / 1024.0),
                "memory_used_gb": float(used_mb / 1024.0),
                "free_mem_gb": float(free_mb / 1024.0),
                "gpu_util": float(_safe_float(parts[6], 0.0)),
                "mem_util": float(_safe_float(parts[7], 0.0)),
                "power_draw_w": float(_safe_float(parts[8], -1.0)),
                "active_compute_process_count": 0,
            }
        )
    return rows


def _augment_compute_process_count(rows: List[Dict[str, Any]]) -> None:
    if not rows or which("nvidia-smi") is None:
        return

    by_uuid = {str(r.get("uuid", "")): r for r in rows}
    try:
        raw = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception:
        raw = ""

    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        gpu_uuid = str(parts[0])
        row = by_uuid.get(gpu_uuid)
        if row is None:
            continue
        row["active_compute_process_count"] = int(row["active_compute_process_count"]) + 1


def _snapshot_via_nvml() -> Dict[str, Any] | None:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None

    try:
        pynvml.nvmlInit()
        count = int(pynvml.nvmlDeviceGetCount())
        rows: List[Dict[str, Any]] = []
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            uuid = pynvml.nvmlDeviceGetUUID(handle)

            try:
                proc_list = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                proc_count = int(len(proc_list))
            except Exception:
                proc_count = 0

            try:
                power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            except Exception:
                power_w = -1.0

            rows.append(
                {
                    "gpu_id": int(idx),
                    "name": name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name),
                    "uuid": uuid.decode("utf-8") if isinstance(uuid, (bytes, bytearray)) else str(uuid),
                    "memory_total_gb": float(mem.total / (1024.0 ** 3)),
                    "memory_used_gb": float(mem.used / (1024.0 ** 3)),
                    "free_mem_gb": float(mem.free / (1024.0 ** 3)),
                    "gpu_util": float(util.gpu),
                    "mem_util": float(util.memory),
                    "power_draw_w": float(power_w),
                    "active_compute_process_count": int(proc_count),
                }
            )

        return {
            "backend": "nvml",
            "timestamp_utc": now_iso(),
            "gpus": rows,
        }
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def snapshot_gpu_telemetry(prefer_nvml: bool = True) -> Dict[str, Any]:
    if prefer_nvml:
        nvml = _snapshot_via_nvml()
        if nvml is not None:
            return nvml

    rows = _query_nvidia_smi_rows()
    _augment_compute_process_count(rows)

    backend = "nvidia-smi" if rows else "unavailable"
    return {
        "backend": backend,
        "timestamp_utc": now_iso(),
        "gpus": rows,
    }
