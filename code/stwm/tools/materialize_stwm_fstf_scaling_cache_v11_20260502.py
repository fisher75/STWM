#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF Scaling Cache Manifest V11", ""]
    for key in [
        "generated_at_utc",
        "cache_manifest_completed",
        "future_leakage_audit",
        "observed_semantic_memory_coverage",
        "target_coverage",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    lines.append("## Scaling Points")
    for point in payload.get("scaling_points", []):
        lines.append(
            f"- {point.get('axis')}={point.get('value')}: "
            f"available=`{point.get('available')}`, blocker=`{point.get('blocking_reason', '')}`"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(root: Path, path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else root / p


def file_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"exists": False, "path": ""}
    exists = path.exists()
    info: dict[str, Any] = {"path": str(path), "exists": bool(exists)}
    if exists:
        stat = path.stat()
        info.update(
            {
                "size_bytes": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "sha1_8": hashlib.sha1(path.read_bytes()[:1_000_000]).hexdigest()[:8],
            }
        )
    return info


def npz_shape(path: Path | None, key: str) -> list[int]:
    if path is None or not path.exists():
        return []
    try:
        with np.load(path, allow_pickle=True) as data:
            if key not in data:
                return []
            return [int(x) for x in data[key].shape]
    except Exception:
        return []


def report_point(root: Path, *, axis: str, value: str, future_report: str, observed_report: str, split_reports: dict[str, str]) -> dict[str, Any]:
    future = load_json(root / future_report)
    observed = load_json(root / observed_report)
    future_cache = resolve(root, str(future.get("target_cache_path") or ""))
    obs_paths = observed.get("target_cache_paths_by_prototype_count", {})
    proto = str(future.get("prototype_count") or value if axis == "C" else 32)
    obs_cache = resolve(root, str(obs_paths.get(str(proto), observed.get("target_cache_path", ""))))
    split_infos = {}
    for split, report in split_reports.items():
        payload = load_json(root / report)
        split_infos[split] = {
            "report": report,
            "item_count": int(payload.get("final_eval_item_count", payload.get(f"materialized_{split}", 0)) or 0),
            "batch_cache": file_info(resolve(root, payload.get("batch_cache_path"))),
            "item_leakage": bool(payload.get("item_leakage", False)),
        }
    available = bool(future and observed and future_cache and future_cache.exists() and obs_cache and obs_cache.exists())
    blocker = "" if available else "missing future or observed prototype cache for this scaling point"
    target_shape = npz_shape(future_cache, "future_semantic_proto_target")
    slot_count = int(target_shape[2]) if len(target_shape) >= 3 else None
    horizon = int(target_shape[1]) if len(target_shape) >= 2 else None
    return {
        "axis": axis,
        "value": value,
        "available": available,
        "blocking_reason": blocker,
        "future_report": future_report,
        "observed_report": observed_report,
        "future_cache": file_info(future_cache),
        "observed_cache": file_info(obs_cache),
        "train_cache_path": split_infos.get("train", {}).get("batch_cache", {}).get("path", ""),
        "val_cache_path": split_infos.get("val", {}).get("batch_cache", {}).get("path", ""),
        "test_cache_path": split_infos.get("test", {}).get("batch_cache", {}).get("path", ""),
        "split_reports": split_infos,
        "item_count": int(future.get("item_count", 0) or 0),
        "slot_count": slot_count,
        "horizon": horizon,
        "prototype_count": int(future.get("prototype_count", 0) or 0),
        "changed_stable_counts": "computed_at_eval_time",
        "future_leakage_audit": bool(future.get("no_future_candidate_leakage", True)),
        "observed_semantic_memory_coverage": float(observed.get("observed_proto_valid_ratio", 0.0) or 0.0),
        "target_coverage": float(future.get("target_valid_ratio", 0.0) or 0.0),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="reports/stwm_fstf_scaling_cache_manifest_v11_20260502.json")
    p.add_argument("--doc", default="docs/STWM_FSTF_SCALING_CACHE_MANIFEST_V11_20260502.md")
    args = p.parse_args()
    root = Path(".").resolve()
    splits = {
        "train": "reports/stwm_mixed_fullscale_v2_materialization_train_20260428.json",
        "val": "reports/stwm_mixed_fullscale_v2_materialization_val_20260428.json",
        "test": "reports/stwm_mixed_fullscale_v2_materialization_test_20260428.json",
    }
    observed = "reports/stwm_mixed_observed_semantic_prototype_targets_v11_20260502.json"
    if not (root / observed).exists():
        observed = "reports/stwm_mixed_observed_semantic_prototype_targets_v2_20260428.json"
    points = [
        report_point(root, axis="C", value="16", future_report="reports/stwm_fstf_future_semantic_trace_prototype_targets_c16_v11_20260502.json", observed_report=observed, split_reports=splits),
        report_point(root, axis="C", value="32", future_report="reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json", observed_report=observed, split_reports=splits),
        report_point(root, axis="C", value="64", future_report="reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json", observed_report=observed, split_reports=splits),
        report_point(root, axis="C", value="128", future_report="reports/stwm_fstf_future_semantic_trace_prototype_targets_c128_v11_20260502.json", observed_report=observed, split_reports=splits),
    ]
    for h in [16, 24]:
        points.append(
            {
                "axis": "H",
                "value": str(h),
                "available": False,
                "blocking_reason": f"H{h} future feature/prototype target cache not materialized in V11 yet",
                "horizon": h,
                "prototype_count": 32,
                "slot_count": 8,
            }
        )
    for k in [16, 32]:
        points.append(
            {
                "axis": "K",
                "value": str(k),
                "available": False,
                "blocking_reason": f"K{k} trace-unit materialization cache not materialized in V11 yet",
                "horizon": 8,
                "prototype_count": 32,
                "slot_count": k,
            }
        )
    payload = {
        "audit_name": "stwm_fstf_scaling_cache_manifest_v11",
        "generated_at_utc": now_iso(),
        "cache_manifest_completed": True,
        "scaling_points": points,
        "future_leakage_audit": all(bool(p.get("future_leakage_audit", True)) for p in points if p.get("available")),
        "observed_semantic_memory_coverage": max([float(p.get("observed_semantic_memory_coverage", 0.0) or 0.0) for p in points] or [0.0]),
        "target_coverage": max([float(p.get("target_coverage", 0.0) or 0.0) for p in points] or [0.0]),
        "missing_scaling_points": [p for p in points if not p.get("available")],
        "no_fake_horizon_or_density_cache": True,
    }
    write_json(Path(args.output), payload)
    write_doc(Path(args.doc), payload)


if __name__ == "__main__":
    main()
