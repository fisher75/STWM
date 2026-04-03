#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import subprocess
from typing import Any


CLASS_POLICIES = {
    "A": {
        "default": {
            "min_free_mem_gib": 28.0,
            "max_gpu_util": 95.0,
            "max_active_apps": 3,
        },
        "fallback": {
            "after_seconds": 10 * 60,
            "min_free_mem_gib": 22.0,
            "max_gpu_util": 97.0,
            "max_active_apps": 4,
        },
    },
    "B": {
        "default": {
            "min_free_mem_gib": 24.0,
            "max_gpu_util": 90.0,
            "max_active_apps": 3,
        },
        "fallback": {
            "after_seconds": 0,
            "min_free_mem_gib": 20.0,
            "max_gpu_util": 95.0,
            "max_active_apps": 4,
        },
    },
    "C": {
        "default": {
            "min_free_mem_gib": 49.0,
            "max_gpu_util": 85.0,
            "max_active_apps": 2,
        },
        "fallback": {
            "after_seconds": 10 * 60,
            "min_free_mem_gib": 40.0,
            "max_gpu_util": 90.0,
            "max_active_apps": 3,
        },
    },
}


def _run(cmd: list[str]) -> str:
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    return out.strip()


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _safe_int(v: str, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _read_gpu_snapshot() -> list[dict[str, Any]]:
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 6:
            continue
        idx = _safe_int(parts[0], -1)
        uuid = str(parts[1])
        mem_total = _safe_int(parts[2], 0)
        mem_used = _safe_int(parts[3], 0)
        util = _safe_float(parts[4], 0.0)
        mem_util = _safe_float(parts[5], 0.0)
        free_mib = max(0, mem_total - mem_used)
        rows.append(
            {
                "index": idx,
                "uuid": uuid,
                "memory_total_mib": mem_total,
                "memory_used_mib": mem_used,
                "free_mem_mib": free_mib,
                "free_mem_gib": float(free_mib) / 1024.0,
                "gpu_util": util,
                "memory_util": mem_util,
                "active_compute_apps": 0,
            }
        )
    return rows


def _augment_compute_apps(rows: list[dict[str, Any]]) -> None:
    by_uuid = {str(r["uuid"]): r for r in rows}
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
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        uuid = str(parts[0])
        if uuid in by_uuid:
            by_uuid[uuid]["active_compute_apps"] = int(by_uuid[uuid]["active_compute_apps"]) + 1


def _read_leases(queue_root: Path) -> dict[int, dict[str, int]]:
    lease_dir = queue_root / "leases"
    lease_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[int, dict[str, int]] = {}
    stale: list[Path] = []
    for lease in sorted(lease_dir.glob("*.json")):
        try:
            payload = json.loads(lease.read_text())
        except Exception:
            stale.append(lease)
            continue

        pid = _safe_int(str(payload.get("pid", "0")), 0)
        if pid > 0:
            if not Path(f"/proc/{pid}").exists():
                stale.append(lease)
                continue

        gpu_index = _safe_int(str(payload.get("gpu_index", "-1")), -1)
        class_type = str(payload.get("class_type", "")).strip().upper()
        if gpu_index < 0 or class_type not in {"A", "B", "C"}:
            stale.append(lease)
            continue

        if gpu_index not in counts:
            counts[gpu_index] = {"A": 0, "B": 0, "C": 0}
        counts[gpu_index][class_type] += 1

    for p in stale:
        try:
            p.unlink()
        except Exception:
            pass

    return counts


def _occupancy_class(active_apps: int) -> str:
    if active_apps <= 0:
        return "empty"
    if active_apps <= 2:
        return "lightly_shared"
    return "busy"


def _occupancy_rank(cls: str) -> int:
    if cls == "empty":
        return 2
    if cls == "lightly_shared":
        return 1
    return 0


def _policy_for(class_type: str, wait_seconds: int) -> dict[str, Any]:
    cfg = CLASS_POLICIES[str(class_type)]
    fallback_after = int(cfg["fallback"]["after_seconds"])
    use_fallback = int(wait_seconds) >= fallback_after
    active = dict(cfg["fallback"] if use_fallback else cfg["default"])
    active.pop("after_seconds", None)
    return {
        "mode": "fallback" if use_fallback else "default",
        "wait_seconds": int(wait_seconds),
        "fallback_after_seconds": int(fallback_after),
        "thresholds": active,
    }


def _threshold_blockers(row: dict[str, Any], thresholds: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    apps = int(row["active_compute_apps"])
    free_mem_gib = float(row["free_mem_gib"])
    util = float(row["gpu_util"])

    min_free = float(thresholds["min_free_mem_gib"])
    max_util = float(thresholds["max_gpu_util"])
    max_apps = int(thresholds["max_active_apps"])

    blockers: list[str] = []
    gap = {
        "free_mem_gib_short": float(max(0.0, min_free - free_mem_gib)),
        "gpu_util_excess": float(max(0.0, util - max_util)),
        "active_apps_excess": float(max(0, apps - max_apps)),
    }
    if free_mem_gib < min_free:
        blockers.append("threshold_free_mem")
    if util > max_util:
        blockers.append("threshold_gpu_util")
    if apps > max_apps:
        blockers.append("threshold_active_apps")

    return blockers, gap


def _lease_blockers(class_type: str, lease_counts: dict[str, int]) -> list[str]:
    a = int(lease_counts.get("A", 0))
    b = int(lease_counts.get("B", 0))
    c = int(lease_counts.get("C", 0))
    total = a + b + c

    out: list[str] = []
    if class_type == "C":
        # Class C keeps conservative isolation for future 1B training.
        if a > 0 or b > 0 or c > 0:
            out.append("lease_conflict_class_c_isolation")
        return out

    if c > 0:
        out.append("lease_conflict_with_class_c")
    if total >= 2:
        out.append("lease_conflict_stwm_card_cap_2")

    return out


def _eligibility(row: dict[str, Any], class_type: str, thresholds: dict[str, Any], lease_counts: dict[str, int]) -> tuple[bool, list[str], dict[str, float]]:
    threshold_blockers, gap = _threshold_blockers(row, thresholds)
    lease_blockers = _lease_blockers(class_type, lease_counts)
    blockers = list(threshold_blockers) + list(lease_blockers)
    return len(blockers) == 0, blockers, gap


def _nearest_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None

    def _score(r: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
        lease_penalty = 15.0 if any(str(x).startswith("lease_") for x in r.get("blockers", [])) else 0.0
        gap = r.get("threshold_gap", {})
        free_gap = float(gap.get("free_mem_gib_short", 0.0))
        util_gap = float(gap.get("gpu_util_excess", 0.0))
        app_gap = float(gap.get("active_apps_excess", 0.0))
        base_penalty = free_gap * 3.0 + util_gap * 0.2 + app_gap * 5.0
        return (
            lease_penalty + base_penalty,
            -float(_occupancy_rank(str(r.get("occupancy_class", "busy")))),
            -float(r.get("free_mem_mib", 0.0)),
            float(r.get("gpu_util", 0.0)),
            float(r.get("active_compute_apps", 0.0)),
            float(r.get("memory_util", 0.0)),
        )

    ordered = sorted(rows, key=_score)
    best = dict(ordered[0])
    return {
        "index": int(best.get("index", -1)),
        "occupancy_class": str(best.get("occupancy_class", "")),
        "free_mem_gib": float(best.get("free_mem_gib", 0.0)),
        "gpu_util": float(best.get("gpu_util", 0.0)),
        "active_compute_apps": int(best.get("active_compute_apps", 0)),
        "blocking_reasons": list(best.get("blockers", [])),
        "threshold_gap": dict(best.get("threshold_gap", {})),
    }


def select_gpu(
    *,
    class_type: str,
    queue_root: Path,
    wait_seconds: int,
) -> dict[str, Any]:
    rows = _read_gpu_snapshot()
    _augment_compute_apps(rows)
    lease_map = _read_leases(queue_root)

    policy = _policy_for(str(class_type), int(wait_seconds))
    thresholds = dict(policy["thresholds"])

    decorated: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for row in rows:
        idx = int(row["index"])
        lease_counts = lease_map.get(idx, {"A": 0, "B": 0, "C": 0})
        row["lease_counts"] = dict(lease_counts)
        row["occupancy_class"] = _occupancy_class(int(row["active_compute_apps"]))

        ok, blockers, gap = _eligibility(
            row,
            class_type=str(class_type),
            thresholds=thresholds,
            lease_counts=lease_counts,
        )

        row["eligible"] = bool(ok)
        row["threshold_gap"] = gap
        row["blockers"] = blockers
        row["eligible_reason"] = "eligible" if ok else ";".join(blockers)
        decorated.append(row)
        if ok:
            candidates.append(row)

    def _sort_key(r: dict[str, Any]) -> tuple[float, float, float, float, float]:
        return (
            float(_occupancy_rank(str(r["occupancy_class"]))),
            float(r["free_mem_mib"]),
            -float(r["gpu_util"]),
            -float(r["active_compute_apps"]),
            -float(r["memory_util"]),
        )

    chosen = None
    if candidates:
        candidates.sort(key=_sort_key, reverse=True)
        chosen = dict(candidates[0])

    nearest = None if chosen is not None else _nearest_candidate(decorated)

    return {
        "status": "ok" if chosen is not None else "no_eligible_gpu",
        "class_type": class_type,
        "policy": policy,
        "snapshot": decorated,
        "chosen": chosen,
        "nearest_candidate": nearest,
    }


def build_parser() -> ArgumentParser:
    p = ArgumentParser(description="Select GPU for protocol_v2 queue by class policy")
    p.add_argument("--class-type", required=True, choices=["A", "B", "C"])
    p.add_argument("--queue-root", required=True)
    p.add_argument("--wait-seconds", type=int, default=0)
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = select_gpu(
        class_type=str(args.class_type),
        queue_root=Path(args.queue_root),
        wait_seconds=max(0, int(args.wait_seconds)),
    )
    print(json.dumps(out, ensure_ascii=True))


if __name__ == "__main__":
    main()
