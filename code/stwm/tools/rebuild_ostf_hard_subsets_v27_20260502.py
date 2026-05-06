#!/usr/bin/env python3
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v27_prior_utils_20260502 import (
    dataset_counts,
    load_combo,
    predict_damped_velocity,
    predict_last,
    scalar,
)


REPORT_PATH = ROOT / "reports/stwm_ostf_hard_subsets_v27_20260502.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_HARD_SUBSETS_V27_20260502.md"
COMBOS = ("M128_H32", "M512_H32", "M128_H64", "M512_H64")


def _endpoint_error(samples: list[Any], pred: np.ndarray) -> np.ndarray:
    vals = []
    for i, s in enumerate(samples):
        valid = s.fut_vis[:, -1]
        err = np.abs(pred[i, :, -1] - s.fut_points[:, -1]).sum(axis=-1) * 1000.0
        vals.append(float(err[valid].mean()) if np.any(valid) else float(np.abs(pred[i] - s.fut_points).sum(axis=-1)[s.fut_vis].mean() * 1000.0))
    return np.asarray(vals, dtype=np.float64)


def _future_displacement(samples: list[Any]) -> np.ndarray:
    vals = []
    for s in samples:
        valid = s.fut_vis[:, -1]
        disp = np.abs(s.fut_points[:, -1] - s.obs_points[:, -1]).sum(axis=-1) * 1000.0
        vals.append(float(disp[valid].mean()) if np.any(valid) else 0.0)
    return np.asarray(vals, dtype=np.float64)


def _curvature(samples: list[Any]) -> np.ndarray:
    vals = []
    for s in samples:
        seq = np.concatenate([s.obs_points[:, -3:], s.fut_points], axis=1)
        if seq.shape[1] < 3:
            vals.append(0.0)
            continue
        acc = seq[:, 2:] - 2.0 * seq[:, 1:-1] + seq[:, :-2]
        valid = np.concatenate([s.obs_vis[:, -3:], s.fut_vis], axis=1)[:, 2:]
        curv = np.linalg.norm(acc, axis=-1) * 1000.0
        vals.append(float(curv[valid].mean()) if np.any(valid) else 0.0)
    return np.asarray(vals, dtype=np.float64)


def _extraction_uncertainty(samples: list[Any]) -> np.ndarray:
    vals = []
    for s in samples:
        try:
            z = np.load(ROOT / s.source_cache_path, allow_pickle=True)
            obj = int(s.object_index)
            nn = np.asarray(z["nn_distance"], dtype=np.float32)[obj]
            conf = np.asarray(z["confidence"], dtype=np.float32)[obj]
            same = float(scalar(z["same_trajectory_fraction"]))
            fut_slice = slice(int(s.obs_points.shape[1]), int(s.obs_points.shape[1]) + int(s.h))
            score = float(np.nanmean(nn[:, fut_slice]) - 0.05 * np.nanmean(conf[:, fut_slice]) + 10.0 * same)
        except Exception:
            score = float(1.0 - s.fut_vis.mean())
        vals.append(score)
    return np.asarray(vals, dtype=np.float64)


def _semantic_confuser(samples: list[Any]) -> list[bool]:
    by_item: dict[str, Counter] = defaultdict(Counter)
    for s in samples:
        by_item[s.item_key][int(s.semantic_id)] += 1
    flags = []
    for s in samples:
        duplicated_sem = by_item[s.item_key][int(s.semantic_id)] > 1
        flags.append(bool(duplicated_sem or s.subset_flags.get("interaction_hard", False)))
    return flags


def _top_mask(vals: np.ndarray, q: float) -> tuple[list[bool], float]:
    if vals.size == 0:
        return [], 0.0
    thr = float(np.percentile(vals, q))
    return [bool(v >= thr) for v in vals], thr


def _summarize_combo(combo: str) -> dict[str, Any]:
    rows, proto = load_combo(combo)
    samples = rows["test"]
    if not samples:
        return {"test_object_count": 0}
    last_err = _endpoint_error(samples, predict_last(samples))
    damp025_err = _endpoint_error(samples, predict_damped_velocity(samples, 0.25))
    nonlinear_vals = _curvature(samples)
    large_disp_vals = _future_displacement(samples)
    uncertainty_vals = _extraction_uncertainty(samples)
    last_top20, last_thr20 = _top_mask(last_err, 80)
    last_top30, last_thr30 = _top_mask(last_err, 70)
    damp_top20, damp_thr20 = _top_mask(damp025_err, 80)
    nonlinear_top20, nonlinear_thr20 = _top_mask(nonlinear_vals, 80)
    large_disp_top20, large_disp_thr20 = _top_mask(large_disp_vals, 80)
    uncertainty_top20, uncertainty_thr20 = _top_mask(uncertainty_vals, 80)
    occlusion = [bool(s.occlusion_ratio >= 0.4 or s.reappearance_flag > 0.0) for s in samples]
    semantic_confuser = _semantic_confuser(samples)
    flags = {
        "last_observed_hard_top20": last_top20,
        "last_observed_hard_top30": last_top30,
        "damped_cv_hard_top20": damp_top20,
        "nonlinear_displacement_hard_top20": nonlinear_top20,
        "occlusion_reappearance": occlusion,
        "large_displacement_top20": large_disp_top20,
        "semantic_identity_confuser": semantic_confuser,
        "target_side_extraction_uncertainty_top20": uncertainty_top20,
    }
    counts = dataset_counts(samples, flags)
    ids = {
        name: sorted({s.item_key for s, flag in zip(samples, mask) if flag})[:100]
        for name, mask in flags.items()
    }
    overlap_with_old_cv_hard = {}
    old_cv = [bool(s.subset_flags.get("top20_cv_hard", False)) for s in samples]
    for name, mask in flags.items():
        denom = max(sum(mask), 1)
        overlap_with_old_cv_hard[name] = float(sum(a and b for a, b in zip(mask, old_cv)) / denom)
    return {
        "test_object_count": len(samples),
        "thresholds": {
            "last_observed_hard_top20_endpoint_px": last_thr20,
            "last_observed_hard_top30_endpoint_px": last_thr30,
            "damped_cv_hard_top20_endpoint_px": damp_thr20,
            "nonlinear_displacement_top20_curvature_px": nonlinear_thr20,
            "large_displacement_top20_endpoint_disp_px": large_disp_thr20,
            "target_side_extraction_uncertainty_top20_score": uncertainty_thr20,
        },
        "counts": counts,
        "clip_ids_first_100": ids,
        "old_cv_hard_overlap_rate": overlap_with_old_cv_hard,
        "dataset_totals": dict(Counter(s.dataset for s in samples)),
    }


def main() -> int:
    combos = {combo: _summarize_combo(combo) for combo in COMBOS}
    payload = {
        "audit_name": "stwm_ostf_hard_subsets_v27",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hard_subset_policy": "Replaces CV-hard-only subsets with last-observed, damped-velocity, nonlinear, occlusion, large-displacement, semantic-confuser, and extraction-uncertainty subsets.",
        "combos": combos,
        "revised_hardbench_ready": all(v.get("test_object_count", 0) > 0 for v in combos.values()),
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF Hard Subsets V27",
        payload,
        ["hard_subset_policy", "revised_hardbench_ready"],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
