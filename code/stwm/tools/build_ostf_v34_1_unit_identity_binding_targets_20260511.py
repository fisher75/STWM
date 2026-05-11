#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


IDENTITY_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/semantic_identity_targets/pointodyssey"
GLOBAL_ROOT = ROOT / "outputs/cache/stwm_ostf_v33_8_complete_h32_m128/global_identity_labels/pointodyssey"
MEAS_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_semantic_measurement_bank/pointodyssey"
OUT_ROOT = ROOT / "outputs/cache/stwm_ostf_v34_1_unit_identity_binding_targets/pointodyssey"
REPORT = ROOT / "reports/stwm_ostf_v34_1_unit_identity_binding_target_build_20260511.json"
DOC = ROOT / "docs/STWM_OSTF_V34_1_UNIT_IDENTITY_BINDING_TARGET_BUILD_20260511.md"


def instance_to_unit(point_to_instance: np.ndarray, units: int = 16) -> tuple[np.ndarray, dict[str, int]]:
    target = np.full(point_to_instance.shape, -1, dtype=np.int64)
    valid = point_to_instance >= 0
    values, counts = np.unique(point_to_instance[valid], return_counts=True) if valid.any() else (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
    order = values[np.argsort(-counts)] if len(values) else values
    mapping: dict[str, int] = {}
    for i, inst in enumerate(order[:units]):
        mapping[str(int(inst))] = int(i)
        target[point_to_instance == inst] = int(i)
    # Rare overflow instances remain supervised by a deterministic bucket so every valid point has a target.
    for inst in order[units:]:
        bucket = int(abs(int(inst)) % units)
        mapping[str(int(inst))] = bucket
        target[point_to_instance == inst] = bucket
    return target, mapping


def same_instance_pairs(point_to_instance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = point_to_instance >= 0
    same = (point_to_instance[:, None] == point_to_instance[None, :]) & valid[:, None] & valid[None, :]
    pair_mask = valid[:, None] & valid[None, :]
    np.fill_diagonal(pair_mask, False)
    np.fill_diagonal(same, False)
    return same.astype(bool), pair_mask.astype(bool)


def semantic_purity_target(meas_path: Path, point_to_unit: np.ndarray, units: int) -> np.ndarray:
    z = np.load(meas_path, allow_pickle=True)
    obs = np.asarray(z["obs_semantic_measurements"], dtype=np.float32)
    mask = np.asarray(z["obs_semantic_measurement_mask"]).astype(bool)
    pooled = (obs * mask[..., None]).sum(axis=1) / np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    pooled = pooled / np.maximum(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-6)
    out = np.zeros((units,), dtype=np.float32)
    for u in range(units):
        idx = np.where(point_to_unit == u)[0]
        if idx.size <= 1:
            out[u] = 1.0 if idx.size == 1 else 0.0
            continue
        sim = pooled[idx] @ pooled[idx].T
        tri = sim[np.triu_indices(idx.size, 1)]
        out[u] = float(np.nanmean(tri)) if tri.size else 0.0
    return out


def build_one(split: str, path: Path, units: int) -> dict[str, Any]:
    uid = path.stem
    ident = np.load(path, allow_pickle=True)
    glob_path = GLOBAL_ROOT / split / f"{uid}.npz"
    meas_path = MEAS_ROOT / split / f"{uid}.npz"
    if not glob_path.exists() or not meas_path.exists():
        return {"uid": uid, "written": False, "blocker": "missing_global_or_measurement_sidecar"}
    glob = np.load(glob_path, allow_pickle=True)
    point_to_instance = np.asarray(ident["point_to_instance_id"], dtype=np.int64)
    point_to_unit, mapping = instance_to_unit(point_to_instance, units=units)
    same, pair_mask = same_instance_pairs(point_to_instance)
    unit_identity_purity = np.zeros((units,), dtype=np.float32)
    for u in range(units):
        idx = np.where(point_to_unit == u)[0]
        if idx.size == 0:
            unit_identity_purity[u] = 0.0
        else:
            vals, counts = np.unique(point_to_instance[idx], return_counts=True)
            valid_counts = counts[vals >= 0]
            unit_identity_purity[u] = float(valid_counts.max() / max(valid_counts.sum(), 1)) if valid_counts.size else 0.0
    unit_semantic_purity = semantic_purity_target(meas_path, point_to_unit, units)
    out = {
        "sample_uid": np.asarray(uid),
        "point_id": np.asarray(ident["point_id"], dtype=np.int64),
        "point_to_instance_id": point_to_instance,
        "fut_global_instance_id": np.asarray(glob["fut_global_instance_id"], dtype=np.int64),
        "fut_global_instance_available_mask": np.asarray(glob["fut_global_instance_available_mask"]).astype(bool),
        "fut_same_instance_as_obs": np.asarray(ident["fut_same_instance_as_obs"]).astype(bool),
        "point_to_unit_target": point_to_unit.astype(np.int64),
        "same_instance_unit_pair_mask": same.astype(bool),
        "same_instance_pair_available_mask": pair_mask.astype(bool),
        "unit_identity_purity_target": unit_identity_purity,
        "unit_semantic_purity_target": unit_semantic_purity,
        "unit_temporal_consistency_target": np.asarray(ident["fut_same_instance_as_obs"]).mean(axis=1).astype(np.float32),
        "instance_to_unit_map_json": np.asarray(json.dumps(mapping, sort_keys=True)),
        "leakage_safe": np.asarray(True),
        "future_labels_supervision_only": np.asarray(True),
    }
    dst = OUT_ROOT / split / f"{uid}.npz"
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    return {
        "uid": uid,
        "written": True,
        "pair_count": int(pair_mask.sum()),
        "same_pair_count": int(same.sum()),
        "instance_count": int(len([k for k in mapping.keys()])),
        "valid_point_count": int((point_to_unit >= 0).sum()),
    }


def main() -> int:
    units = 16
    stats: dict[str, Any] = {}
    blockers: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        rows = []
        for path in sorted((IDENTITY_ROOT / split).glob("*.npz")):
            r = build_one(split, path, units)
            rows.append(r)
            if not r.get("written"):
                blockers.append(r)
        written = [r for r in rows if r.get("written")]
        stats[split] = {
            "samples": len(rows),
            "written": len(written),
            "coverage": float(len(written) / max(len(rows), 1)),
            "pair_count": int(sum(r.get("pair_count", 0) for r in written)),
            "same_pair_count": int(sum(r.get("same_pair_count", 0) for r in written)),
            "instance_count_mean": float(np.mean([r.get("instance_count", 0) for r in written])) if written else 0.0,
            "valid_point_count_mean": float(np.mean([r.get("valid_point_count", 0) for r in written])) if written else 0.0,
        }
    ok = all(v["coverage"] >= 0.95 for v in stats.values())
    payload = {
        "generated_at_utc": utc_now(),
        "output_root": str(OUT_ROOT.relative_to(ROOT)),
        "coverage_by_split": {k: v["coverage"] for k, v in stats.items()},
        "pair_count_by_split": {k: v["pair_count"] for k, v in stats.items()},
        "instance_count_by_split": {k: v["instance_count_mean"] for k, v in stats.items()},
        "unit_target_feasibility": bool(ok),
        "unit_identity_binding_targets_built": bool(ok),
        "stats_by_split": stats,
        "leakage_safe": True,
        "exact_blockers": blockers[:20],
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V34.1 Unit Identity Binding Target Build",
        payload,
        ["output_root", "coverage_by_split", "pair_count_by_split", "instance_count_by_split", "unit_target_feasibility", "unit_identity_binding_targets_built", "leakage_safe", "exact_blockers"],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
