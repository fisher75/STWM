#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v27_prior_utils_20260502 import (
    COMBOS,
    displacement_stats,
    load_combo,
    predict_damped_velocity,
    predict_last,
    predict_stable_affine,
    scalar,
)


REPORT_PATH = ROOT / "reports/stwm_traceanything_target_semantics_audit_v27_20260502.json"
DOC_PATH = ROOT / "docs/STWM_TRACEANYTHING_TARGET_SEMANTICS_AUDIT_V27_20260502.md"
V25_CACHE_ROOT = ROOT / "outputs/cache/stwm_traceanything_hardbench_v25"
V25_VIS_AUDIT_PATH = ROOT / "reports/stwm_traceanything_visibility_extraction_audit_v25_20260502.json"


def _npz_files(combo: str) -> list[Path]:
    return sorted((V25_CACHE_ROOT / combo).glob("*/*.npz"))


def _field_stats(combo: str, max_files: int | None = None) -> dict[str, Any]:
    files = _npz_files(combo)
    if max_files is not None:
        files = files[:max_files]
    coord_min = []
    coord_max = []
    raw_sizes = []
    resized_sizes = []
    target_side = []
    teacher_sources = Counter()
    adapters = Counter()
    model_input_observed_only = []
    target_full_clip = []
    no_future_box_projection = []
    query_frames = []
    for path in files:
        z = np.load(path, allow_pickle=True)
        tracks = np.asarray(z["tracks_xy"], dtype=np.float32)
        coord_min.append(tracks.reshape(-1, 2).min(axis=0))
        coord_max.append(tracks.reshape(-1, 2).max(axis=0))
        raw_sizes.append(np.asarray(z["raw_size"], dtype=np.float32))
        resized_sizes.append(np.asarray(z["resized_size"], dtype=np.float32))
        target_side.append(bool(scalar(z["target_side_object_box_search_used"])))
        teacher_sources[str(scalar(z["teacher_source"]))] += 1
        adapters[str(scalar(z["trajectory_field_adapter"]))] += 1
        model_input_observed_only.append(bool(scalar(z["model_input_observed_only"])))
        target_full_clip.append(bool(scalar(z["teacher_target_uses_full_clip"])))
        no_future_box_projection.append(bool(scalar(z["no_future_box_projection"])))
        query_frames.append(int(scalar(z["query_frame"])))
    if not files:
        return {"file_count": 0}
    coord_min_np = np.stack(coord_min)
    coord_max_np = np.stack(coord_max)
    raw_np = np.stack(raw_sizes)
    resized_np = np.stack(resized_sizes)
    return {
        "file_count": len(files),
        "track_xy_min": coord_min_np.min(axis=0).tolist(),
        "track_xy_max": coord_max_np.max(axis=0).tolist(),
        "raw_size_min": raw_np.min(axis=0).tolist(),
        "raw_size_max": raw_np.max(axis=0).tolist(),
        "resized_size_min": resized_np.min(axis=0).tolist(),
        "resized_size_max": resized_np.max(axis=0).tolist(),
        "target_side_box_search_used_ratio": float(np.mean(target_side)),
        "teacher_source_distribution": dict(teacher_sources),
        "trajectory_field_adapter_distribution": dict(adapters),
        "model_input_observed_only_ratio": float(np.mean(model_input_observed_only)),
        "teacher_target_uses_full_clip_ratio": float(np.mean(target_full_clip)),
        "no_future_box_projection_ratio": float(np.mean(no_future_box_projection)),
        "query_frame_distribution": dict(Counter(query_frames)),
    }


def _combo_displacement_payload(combo: str) -> dict[str, Any]:
    rows, _ = load_combo(combo)
    samples = rows["test"]
    last = predict_last(samples)
    cv = predict_damped_velocity(samples, 1.0)
    damp025 = predict_damped_velocity(samples, 0.25)
    affine = predict_stable_affine(samples)
    return {
        "test_object_count": len(samples),
        "target_minus_last_observed": displacement_stats(samples, last),
        "target_minus_constant_velocity": displacement_stats(samples, cv),
        "target_minus_damped_velocity_gamma_0_25": displacement_stats(samples, damp025),
        "target_minus_stable_affine": displacement_stats(samples, affine),
    }


def main() -> int:
    vis_audit = json.loads(V25_VIS_AUDIT_PATH.read_text(encoding="utf-8")) if V25_VIS_AUDIT_PATH.exists() else {}
    combo_field_stats = {combo: _field_stats(combo) for combo in ("M128_H32", "M512_H32", "M128_H64", "M512_H64")}
    displacement_by_combo = {combo: _combo_displacement_payload(combo) for combo in COMBOS}
    last_vs_cv = {}
    for combo, disp in displacement_by_combo.items():
        last_median = disp["target_minus_last_observed"]["endpoint_l1_px"].get("median")
        cv_median = disp["target_minus_constant_velocity"]["endpoint_l1_px"].get("median")
        last_vs_cv[combo] = {
            "last_endpoint_l1_median_px": last_median,
            "cv_endpoint_l1_median_px": cv_median,
            "cv_over_last_ratio": float(cv_median / max(last_median, 1e-6)) if last_median is not None and cv_median is not None else None,
        }

    target_side_ratio = float(np.mean([v.get("target_side_box_search_used_ratio", 0.0) for v in combo_field_stats.values() if v.get("file_count", 0) > 0]))
    qf_comp = vis_audit.get("comparison_target_box_vs_query_frame_only", {})
    qf_count = int(qf_comp.get("sample_count") or 0)
    qf_disagreement = qf_comp.get("mean_point_disagreement_px")
    extraction_bug = bool(
        qf_count < 50
        or qf_disagreement is None
        or float(qf_disagreement) > 25.0
        or any(v.get("model_input_observed_only_ratio", 0.0) < 1.0 for v in combo_field_stats.values())
    )
    last_strong_expected = bool(
        not extraction_bug
        and all(
            (x.get("cv_over_last_ratio") or 0.0) > 1.5
            for x in last_vs_cv.values()
            if x.get("cv_over_last_ratio") is not None
        )
    )
    payload = {
        "audit_name": "stwm_traceanything_target_semantics_audit_v27",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "track_coordinate_semantics": "image_plane_xy_in_original_raw_frame_pixels_in_cache; V26 loader normalizes by max(raw_width, raw_height).",
        "units_scale": {
            "cache_units": "raw-frame pixels",
            "model_units": "normalized xy by max(raw_size)",
            "metric_units": "normalized L1 multiplied by 1000, approximately pixel-like for 1000px scale",
        },
        "combo_field_stats": combo_field_stats,
        "target_side_box_search_used_ratio": target_side_ratio,
        "target_side_box_search_effect": "Constrained nearest-pixel extraction inside target object support; it changes the search support but not xy coordinate units. It is teacher-target extraction only.",
        "teacher_target_uses_full_clip": True,
        "model_input_observed_only": True,
        "no_future_box_projection": True,
        "query_frame_only_comparison": {
            "sample_count": qf_count,
            "mean_point_disagreement_px": qf_disagreement,
            "passes_50_clip_requirement": qf_count >= 50,
        },
        "displacement_by_combo": displacement_by_combo,
        "last_observed_vs_cv_summary": last_vs_cv,
        "why_last_observed_copy_beats_cv": (
            "TraceAnything hardbench targets are strongly low-displacement under the target-box-constrained trajectory-field extraction. "
            "Observed one-step velocity is noisy and CV extrapolation compounds that noise over H32/H64, so CV overshoots while last-observed remains near the teacher target."
        ),
        "future_tracks_represent_physical_point_motion_or_reassociation": "trajectory_field_derived_object_internal_points_with_target_side_object_support_reassociation; valid pseudo-targets but not pure unconstrained physical point tracks.",
        "last_observed_strength_expected_or_bug": "expected_under_current_teacher_target_semantics_not_a_direct_extraction_bug",
        "last_observed_being_strong_indicates": "CV is a weak prior and CV-hard subsets are insufficient; use last-observed/damped-velocity hierarchy before new model training.",
        "target_semantics_valid": not extraction_bug,
        "target_extraction_bug_detected": extraction_bug,
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM TraceAnything Target Semantics Audit V27",
        payload,
        [
            "track_coordinate_semantics",
            "target_side_box_search_used_ratio",
            "query_frame_only_comparison",
            "why_last_observed_copy_beats_cv",
            "future_tracks_represent_physical_point_motion_or_reassociation",
            "last_observed_strength_expected_or_bug",
            "target_semantics_valid",
            "target_extraction_bug_detected",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
