#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json
import random

import numpy as np

from stwm.tracewm_v2.constants import (
    DATE_TAG,
    FEATURE_INDEX,
    TRACE_AUDIT_REPORT_PATH,
    TRACE_CONTRACT_PATH,
)
from stwm.tracewm_v2.tools.cache_build_utils import now_iso
from stwm.tracewm_v2.trace_cache_contract import build_contract_payload, save_contract


REQUIRED_FIELDS = [
    "dataset",
    "split",
    "clip_id",
    "source_ref",
    "track_source",
    "tracks_2d",
    "tracks_3d",
    "valid",
    "visibility",
    "point_ids",
]


def parse_args() -> Any:
    parser = ArgumentParser(description="Audit Stage1 v2 trace cache and write contract manifest")
    parser.add_argument(
        "--point-index",
        default=f"/home/chen034/workspace/data/_manifests/stage1_v2_pointodyssey_cache_index_{DATE_TAG}.json",
    )
    parser.add_argument(
        "--kubric-index",
        default=f"/home/chen034/workspace/data/_manifests/stage1_v2_kubric_cache_index_{DATE_TAG}.json",
    )
    parser.add_argument("--contract-out", default=str(TRACE_CONTRACT_PATH))
    parser.add_argument("--audit-out", default=str(TRACE_AUDIT_REPORT_PATH))
    parser.add_argument("--sample-per-dataset", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260408)
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"index json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_entries(entries: List[Dict[str, Any]], sample_n: int, seed: int) -> List[Dict[str, Any]]:
    if len(entries) <= sample_n:
        return entries
    rng = random.Random(seed)
    idx = list(range(len(entries)))
    rng.shuffle(idx)
    picked = sorted(idx[:sample_n])
    return [entries[i] for i in picked]


def _as_bool(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.bool_:
        return arr
    return arr.astype(bool)


def _audit_npz(path: Path) -> Dict[str, Any]:
    out = {
        "path": str(path),
        "exists": path.exists(),
        "missing_fields": [],
        "bad_shape": False,
        "synthetic_marker": False,
        "finite_ratio": 0.0,
        "dataset": "",
        "clip_id": "",
    }
    if not path.exists():
        out["missing_fields"] = list(REQUIRED_FIELDS)
        return out

    try:
        payload = np.load(path, allow_pickle=True)
    except Exception:
        out["bad_shape"] = True
        return out

    for key in REQUIRED_FIELDS:
        if key not in payload.files:
            out["missing_fields"].append(key)

    if out["missing_fields"]:
        return out

    dataset = str(payload["dataset"].item())
    clip_id = str(payload["clip_id"].item())
    track_source = str(payload["track_source"].item())

    out["dataset"] = dataset
    out["clip_id"] = clip_id
    out["synthetic_marker"] = "deterministic" in track_source.lower() or "synthetic" in track_source.lower()

    t2d = np.asarray(payload["tracks_2d"], dtype=np.float32)
    t3d = np.asarray(payload["tracks_3d"], dtype=np.float32)
    valid = _as_bool(np.asarray(payload["valid"]))
    vis = _as_bool(np.asarray(payload["visibility"]))
    pids = np.asarray(payload["point_ids"])

    if t2d.ndim != 3 or t2d.shape[-1] != 2:
        out["bad_shape"] = True
        return out
    if t3d.ndim != 3 or t3d.shape[-1] != 3:
        out["bad_shape"] = True
        return out
    if valid.shape != t2d.shape[:2] or vis.shape != t2d.shape[:2]:
        out["bad_shape"] = True
        return out
    if pids.ndim != 1 or pids.shape[0] != t2d.shape[1]:
        out["bad_shape"] = True
        return out

    finite_2d = np.isfinite(t2d).all(axis=-1)
    finite_3d = np.isfinite(t3d).all(axis=-1)
    finite = finite_2d & finite_3d

    denom = float(max(valid.size, 1))
    out["finite_ratio"] = float(finite.sum() / denom)
    return out


def _audit_dataset(index_payload: Dict[str, Any], sample_n: int, seed: int) -> Dict[str, Any]:
    entries = [e for e in index_payload.get("entries", []) if isinstance(e, dict)]
    sampled = _sample_entries(entries, sample_n=sample_n, seed=seed)

    sample_reports = []
    missing_files = 0
    missing_field_files = 0
    bad_shape_files = 0
    synthetic_marker_files = 0
    finite_ratios: List[float] = []

    for rec in sampled:
        path = Path(str(rec.get("cache_path", "")))
        rep = _audit_npz(path)
        sample_reports.append(rep)

        if not rep["exists"]:
            missing_files += 1
        if rep["missing_fields"]:
            missing_field_files += 1
        if rep["bad_shape"]:
            bad_shape_files += 1
        if rep["synthetic_marker"]:
            synthetic_marker_files += 1
        finite_ratios.append(float(rep["finite_ratio"]))

    min_finite = float(min(finite_ratios)) if finite_ratios else 0.0
    mean_finite = float(sum(finite_ratios) / len(finite_ratios)) if finite_ratios else 0.0

    passed = (
        len(sampled) > 0
        and missing_files == 0
        and missing_field_files == 0
        and bad_shape_files == 0
        and synthetic_marker_files == 0
        and min_finite >= 0.99
    )

    return {
        "dataset": str(index_payload.get("dataset", "unknown")),
        "index_path": str(index_payload.get("index_path", "")),
        "source_root": str(index_payload.get("source_root", "")),
        "cache_root": str(index_payload.get("cache_root", "")),
        "track_source": str(index_payload.get("track_source", "")),
        "clip_len": int(index_payload.get("clip_len", 0)),
        "obs_len": int(index_payload.get("obs_len", 0)),
        "fut_len": int(index_payload.get("fut_len", 0)),
        "stats": dict(index_payload.get("stats", {})),
        "sample_size": int(len(sampled)),
        "missing_files": int(missing_files),
        "missing_field_files": int(missing_field_files),
        "bad_shape_files": int(bad_shape_files),
        "synthetic_marker_files": int(synthetic_marker_files),
        "finite_ratio_min": min_finite,
        "finite_ratio_mean": mean_finite,
        "status": "pass" if passed else "fail",
        "samples": sample_reports,
    }


def main() -> None:
    args = parse_args()
    point_index = _load_json(Path(args.point_index))
    kubric_index = _load_json(Path(args.kubric_index))

    point_audit = _audit_dataset(point_index, sample_n=int(args.sample_per_dataset), seed=int(args.seed))
    kubric_audit = _audit_dataset(kubric_index, sample_n=int(args.sample_per_dataset), seed=int(args.seed) + 1)

    p0_ready = point_audit["status"] == "pass" and kubric_audit["status"] == "pass"

    contract_payload = build_contract_payload(
        generated_at_utc=now_iso(),
        schema_version="stage1_v2_trace_cache_contract_v1",
        feature_layout=FEATURE_INDEX,
        dataset_entries=[
            {
                "dataset_name": "pointodyssey",
                "cache_root": point_audit["cache_root"],
                "index_path": point_audit["index_path"],
                "source_root": point_audit["source_root"],
                "track_source": point_audit["track_source"],
                "enabled": True,
                "clip_len": point_audit["clip_len"],
                "obs_len": point_audit["obs_len"],
                "fut_len": point_audit["fut_len"],
                "split_stats": point_audit["stats"].get("clip_counts", {}),
                "total_clips": int(point_audit["stats"].get("total_clips", 0)),
                "audit_status": point_audit["status"],
            },
            {
                "dataset_name": "kubric",
                "cache_root": kubric_audit["cache_root"],
                "index_path": kubric_audit["index_path"],
                "source_root": kubric_audit["source_root"],
                "track_source": kubric_audit["track_source"],
                "enabled": True,
                "clip_len": kubric_audit["clip_len"],
                "obs_len": kubric_audit["obs_len"],
                "fut_len": kubric_audit["fut_len"],
                "split_stats": kubric_audit["stats"].get("clip_counts", {}),
                "total_clips": int(kubric_audit["stats"].get("total_clips", 0)),
                "audit_status": kubric_audit["status"],
            },
        ],
        summary={
            "p0_trace_cache_ready": bool(p0_ready),
            "pointodyssey_status": point_audit["status"],
            "kubric_status": kubric_audit["status"],
            "required_fields": REQUIRED_FIELDS,
        },
    )

    contract_path = save_contract(contract_payload, path=Path(args.contract_out))

    audit_payload = {
        "generated_at_utc": now_iso(),
        "contract_path": str(contract_path),
        "p0_trace_cache_ready": bool(p0_ready),
        "datasets": {
            "pointodyssey": point_audit,
            "kubric": kubric_audit,
        },
    }

    audit_out = Path(args.audit_out)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[trace-cache-audit] contract={contract_path}")
    print(f"[trace-cache-audit] report={audit_out}")
    print(f"[trace-cache-audit] p0_trace_cache_ready={p0_ready}")


if __name__ == "__main__":
    main()
