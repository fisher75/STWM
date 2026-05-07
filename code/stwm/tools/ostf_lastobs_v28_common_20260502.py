#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_traceanything_common_v26_20260502 import (
    ROOT,
    V24_BRIDGE_PATH,
    V25_CACHE_ROOT,
    batch_from_samples_v26,
    build_v26_rows,
    cache_verification_payload,
)
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import aggregate_item_rows_v26
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc
from stwm.tools.ostf_v27_prior_utils_20260502 import (
    choose_gamma_on_val,
    predict_damped_velocity,
    predict_last,
)


V27_DECISION_PATH = ROOT / "reports/stwm_ostf_v27_prior_decision_20260502.json"
V27_HARD_SUBSETS_PATH = ROOT / "reports/stwm_ostf_hard_subsets_v27_20260502.json"
V27_BASELINE_PATH = ROOT / "reports/stwm_ostf_prior_baseline_hierarchy_v27_20260502.json"
V28_VERIFICATION_PATH = ROOT / "reports/stwm_ostf_v28_cache_hardbench_verification_20260502.json"
V28_VERIFICATION_DOC = ROOT / "docs/STWM_OSTF_V28_CACHE_HARDBENCH_VERIFICATION_20260502.md"
PX_SCALE = 1000.0
V28_SUBSET_KEYS = (
    "last_observed_hard_top20",
    "last_observed_hard_top30",
    "damped_cv_hard_top20",
    "nonlinear_displacement_hard_top20",
    "occlusion_reappearance",
    "large_displacement_top20",
    "semantic_identity_confuser",
    "target_side_extraction_uncertainty_top20",
)


def load_v27_prior_decision() -> dict[str, Any]:
    if not V27_DECISION_PATH.exists():
        return {}
    return json.loads(V27_DECISION_PATH.read_text(encoding="utf-8"))


def selected_damped_gamma(combo: str, rows: dict[str, list[Any]], proto_centers: np.ndarray) -> float:
    if V27_BASELINE_PATH.exists():
        payload = json.loads(V27_BASELINE_PATH.read_text(encoding="utf-8"))
        val = payload.get("combos", {}).get(combo, {}).get("val_selected_gamma")
        if val is not None:
            return float(val)
    gamma, _ = choose_gamma_on_val(rows.get("val", []), proto_centers)
    return float(gamma)


def _endpoint_l1(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    if not np.any(valid[:, -1]):
        if not np.any(valid):
            return 0.0
        return float((np.abs(pred - gt).sum(axis=-1)[valid] * PX_SCALE).mean())
    return float((np.abs(pred[:, -1] - gt[:, -1]).sum(axis=-1)[valid[:, -1]] * PX_SCALE).mean())


def _observed_curvature(sample: Any) -> float:
    pts = sample.obs_points
    if pts.shape[1] < 4:
        return 0.0
    accel = pts[:, 2:] - 2.0 * pts[:, 1:-1] + pts[:, :-2]
    valid = sample.obs_vis[:, 2:] & sample.obs_vis[:, 1:-1] & sample.obs_vis[:, :-2]
    if not np.any(valid):
        return 0.0
    return float(np.linalg.norm(accel, axis=-1)[valid].mean() * PX_SCALE)


def _future_curvature(sample: Any) -> float:
    pts = sample.fut_points
    if pts.shape[1] < 4:
        return 0.0
    accel = pts[:, 2:] - 2.0 * pts[:, 1:-1] + pts[:, :-2]
    valid = sample.fut_vis[:, 2:] & sample.fut_vis[:, 1:-1] & sample.fut_vis[:, :-2]
    if not np.any(valid):
        return 0.0
    return float(np.linalg.norm(accel, axis=-1)[valid].mean() * PX_SCALE)


def _low_confidence_score(sample: Any) -> float:
    valid_ratio = float(sample.fut_vis.mean()) if sample.fut_vis.size else 0.0
    conf = float(sample.fut_conf[sample.fut_vis].mean()) if np.any(sample.fut_vis) else 0.0
    return float((1.0 - valid_ratio) + (1.0 - conf))


def _pct(values: list[float], pct: float) -> float:
    if not values:
        return float("inf")
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def annotate_v28_subsets(samples: list[Any], *, damped_gamma: float) -> dict[str, Any]:
    if not samples:
        return {"item_count": 0, "thresholds": {}, "counts": {}}
    last_pred = predict_last(samples)
    damped_pred = predict_damped_velocity(samples, damped_gamma)
    last_endpoint = [_endpoint_l1(last_pred[i], s.fut_points, s.fut_vis) for i, s in enumerate(samples)]
    damped_endpoint = [_endpoint_l1(damped_pred[i], s.fut_points, s.fut_vis) for i, s in enumerate(samples)]
    obs_curve = [_observed_curvature(s) for s in samples]
    fut_curve = [_future_curvature(s) for s in samples]
    nonlinear = [max(a, b) for a, b in zip(obs_curve, fut_curve)]
    uncertainty = [_low_confidence_score(s) for s in samples]
    semantic_counts: dict[tuple[str, int], int] = {}
    for s in samples:
        semantic_counts[(s.item_key, int(s.semantic_id))] = semantic_counts.get((s.item_key, int(s.semantic_id)), 0) + 1
    thr = {
        "last_observed_top20_endpoint_px": _pct(last_endpoint, 80.0),
        "last_observed_top30_endpoint_px": _pct(last_endpoint, 70.0),
        "damped_cv_top20_endpoint_px": _pct(damped_endpoint, 80.0),
        "large_displacement_top20_endpoint_px": _pct(last_endpoint, 80.0),
        "nonlinear_top20_score": _pct(nonlinear, 80.0),
        "extraction_uncertainty_top20_score": _pct(uncertainty, 80.0),
    }
    counts = {key: 0 for key in V28_SUBSET_KEYS}
    by_dataset = {key: {} for key in V28_SUBSET_KEYS}
    for i, s in enumerate(samples):
        flags = dict(s.subset_flags)
        flags.update(
            {
                "last_observed_hard_top20": bool(last_endpoint[i] >= thr["last_observed_top20_endpoint_px"]),
                "last_observed_hard_top30": bool(last_endpoint[i] >= thr["last_observed_top30_endpoint_px"]),
                "damped_cv_hard_top20": bool(damped_endpoint[i] >= thr["damped_cv_top20_endpoint_px"]),
                "large_displacement_top20": bool(last_endpoint[i] >= thr["large_displacement_top20_endpoint_px"]),
                "nonlinear_displacement_hard_top20": bool(nonlinear[i] >= thr["nonlinear_top20_score"]),
                "occlusion_reappearance": bool(s.occlusion_ratio >= 0.25 or s.reappearance_flag > 0.5),
                "semantic_identity_confuser": bool(semantic_counts.get((s.item_key, int(s.semantic_id)), 0) > 1),
                "target_side_extraction_uncertainty_top20": bool(uncertainty[i] >= thr["extraction_uncertainty_top20_score"]),
            }
        )
        hard = (
            1.0
            + 1.25 * float(flags["last_observed_hard_top20"])
            + 0.75 * float(flags["damped_cv_hard_top20"])
            + 0.50 * float(flags["occlusion_reappearance"])
            + 0.35 * float(flags["nonlinear_displacement_hard_top20"])
            + 0.25 * float(flags["semantic_identity_confuser"])
            + 0.25 * float(flags["target_side_extraction_uncertainty_top20"])
        )
        s.subset_flags = flags
        s.hardness_score = float(min(max(hard, 1.0), 4.5))
        for key in V28_SUBSET_KEYS:
            if flags.get(key, False):
                counts[key] += 1
                ds_counts = by_dataset[key]
                ds_counts[s.dataset] = ds_counts.get(s.dataset, 0) + 1
    return {
        "item_count": len(samples),
        "damped_gamma": float(damped_gamma),
        "thresholds": thr,
        "counts": {k: {"count": int(v), "by_dataset": dict(sorted(by_dataset[k].items()))} for k, v in counts.items()},
    }


def build_v28_rows(combo: str, seed: int = 42) -> tuple[dict[str, list[Any]], np.ndarray, float, dict[str, Any]]:
    rows, proto_centers = build_v26_rows(combo, seed=seed)
    gamma = selected_damped_gamma(combo, rows, proto_centers)
    subset_summary = {split: annotate_v28_subsets(split_rows, damped_gamma=gamma) for split, split_rows in rows.items()}
    return rows, proto_centers, gamma, subset_summary


def add_v28_flags_to_item_rows(rows: list[dict[str, Any]], samples: list[Any]) -> list[dict[str, Any]]:
    for row, sample in zip(rows, samples):
        for key in V28_SUBSET_KEYS:
            row[key] = bool(sample.subset_flags.get(key, False))
    return rows


def v28_subset_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {key: aggregate_item_rows_v26(rows, subset_key=key) for key in V28_SUBSET_KEYS}
    out["old_top20_cv_hard"] = aggregate_item_rows_v26(rows, subset_key="top20_cv_hard")
    return out


def write_v28_cache_hardbench_verification() -> dict[str, Any]:
    base = cache_verification_payload()
    v27_decision = load_v27_prior_decision()
    hard_payload = json.loads(V27_HARD_SUBSETS_PATH.read_text(encoding="utf-8")) if V27_HARD_SUBSETS_PATH.exists() else {}
    prior_payload = json.loads(V27_BASELINE_PATH.read_text(encoding="utf-8")) if V27_BASELINE_PATH.exists() else {}
    combos = ["M128_H32", "M512_H32", "M128_H64", "M512_H64"]
    combo_paths = {}
    for combo in combos:
        combo_root = V25_CACHE_ROOT / combo
        combo_paths[combo] = {
            "path": str(combo_root.relative_to(ROOT)),
            "exists": combo_root.exists(),
            "split_file_counts": {
                split: len(list((combo_root / split).glob("*.npz"))) if (combo_root / split).exists() else 0
                for split in ["train", "val", "test"]
            },
        }
    strong = str(v27_decision.get("strongest_nonlearned_prior") or prior_payload.get("strongest_nonlearned_prior") or "")
    payload = {
        "audit_name": "stwm_ostf_v28_cache_hardbench_verification",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "combo_paths": combo_paths,
        "revised_hard_subsets_exist": bool(hard_payload.get("revised_hardbench_ready")),
        "revised_hard_subset_names": list(V28_SUBSET_KEYS),
        "strongest_nonlearned_prior": strong,
        "recommended_model_prior": str(v27_decision.get("recommended_model_prior") or prior_payload.get("recommended_model_prior") or ""),
        "v27_target_semantics_valid": bool(v27_decision.get("target_semantics_valid")),
        "v27_target_extraction_bug_detected": bool(v27_decision.get("target_extraction_bug_detected")),
        "semantic_identity_bridge_ready": bool(base.get("semantic_identity_bridge_ready")),
        "teacher_source_traceanything_only": bool(base.get("teacher_source_traceanything_only")),
        "model_input_observed_only": bool(base.get("model_input_observed_only")),
        "no_future_leakage": bool(base.get("model_input_observed_only") and base.get("teacher_source_traceanything_only")),
        "cache_verified": bool(
            base.get("cache_verified")
            and hard_payload.get("revised_hardbench_ready")
            and strong in {"last_observed_copy", "learned_global_gamma", "damped_velocity"}
            and v27_decision.get("target_semantics_valid")
            and not v27_decision.get("target_extraction_bug_detected")
        ),
    }
    dump_json(V28_VERIFICATION_PATH, payload)
    write_doc(
        V28_VERIFICATION_DOC,
        "STWM OSTF V28 Cache Hardbench Verification",
        payload,
        [
            "cache_verified",
            "strongest_nonlearned_prior",
            "recommended_model_prior",
            "revised_hard_subsets_exist",
            "v27_target_semantics_valid",
            "v27_target_extraction_bug_detected",
            "teacher_source_traceanything_only",
            "semantic_identity_bridge_ready",
            "model_input_observed_only",
            "no_future_leakage",
        ],
    )
    return payload


__all__ = [
    "ROOT",
    "V28_SUBSET_KEYS",
    "batch_from_samples_v26",
    "build_v28_rows",
    "add_v28_flags_to_item_rows",
    "v28_subset_aggregate",
    "write_v28_cache_hardbench_verification",
    "predict_damped_velocity",
    "predict_last",
]
