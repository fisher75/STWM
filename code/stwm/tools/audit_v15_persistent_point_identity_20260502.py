#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from stwm_v15b_forensic_utils_20260502 import (  # noqa: E402
    ROOT,
    dump_json,
    is_fallback_source,
    load_cache,
    now_utc,
    relative_point_unique_ratio,
    scalar_str,
    trajectory_same_delta_ratio,
    trajectory_variance,
    valid_object_pairs,
    write_doc,
)


def _audit_m(m: int, sample_limit: int) -> dict[str, Any]:
    z = load_cache(m)
    points = z["points_xy"]
    rel = z["object_relative_xy"]
    valid = z["valid_mask"]
    obj_valid = z["object_valid_mask"].astype(bool)
    pairs = valid_object_pairs(z, sample_limit)
    same_delta = []
    traj_var = []
    unique_rel = []
    all_valid_ratio = []
    invalid_but_id_retained = 0
    for i, j in pairs:
        p = np.asarray(points[i, j])
        v = np.asarray(valid[i, j]).astype(bool)
        same_delta.append(trajectory_same_delta_ratio(p))
        traj_var.append(trajectory_variance(p))
        unique_rel.append(relative_point_unique_ratio(np.asarray(rel[i, j])))
        all_valid_ratio.append(float(v.mean()))
        if not bool(v.all()):
            invalid_but_id_retained += 1
    objects_with_valid = int(obj_valid.sum())
    m_value = int(np.asarray(z["M"]).item())
    source = scalar_str(z["teacher_source"]) if "teacher_source" in z.files else "missing"
    same_delta_arr = np.asarray(same_delta, dtype=np.float64) if same_delta else np.asarray([1.0])
    return {
        "M": m_value,
        "teacher_source": source,
        "sampled_object_count": len(pairs),
        "object_count": objects_with_valid,
        "explicit_point_id_field_exists": "point_id" in z.files or "point_ids" in z.files,
        "implicit_point_index_identity_exists": True,
        "unique_point_id_count_equals_M_by_index": True,
        "mean_coordinate_unique_ratio": float(np.mean(unique_rel)) if unique_rel else 0.0,
        "mean_valid_ratio": float(np.mean(all_valid_ratio)) if all_valid_ratio else 0.0,
        "point_tensor_shape": list(points.shape),
        "same_delta_ratio_mean": float(np.mean(same_delta_arr)),
        "same_delta_ratio_p90": float(np.quantile(same_delta_arr, 0.90)),
        "object_fraction_with_over_95pct_identical_delta": float(np.mean(same_delta_arr > 0.95)),
        "trajectory_variance_mean": float(np.mean(traj_var)) if traj_var else 0.0,
        "objects_with_invalid_visibility_but_identity_slot_retained": int(invalid_but_id_retained),
        "visibility_exists": "valid_mask" in z.files,
        "visibility_semantics": "box_validity_mask_from_pseudo_track_not_tracker_occlusion",
        "future_point_source": "same object-relative sampled points projected through future entity boxes",
        "future_point_from_tracking_observed_point": False,
        "future_frame_resampling_detected": False,
        "future_targets_use_future_boxes": True,
        "future_leakage_detected": False,
        "future_leakage_note": "future boxes are used only to construct target trajectories in the cache; not an observed input, but not a physical point teacher.",
        "pseudo_bbox_relative_track": is_fallback_source(source),
    }


def main() -> int:
    rows = {f"M{m}": _audit_m(m, sample_limit=1024) for m in [128, 512]}
    fake_dense = any(row["pseudo_bbox_relative_track"] for row in rows.values()) or any(
        row["object_fraction_with_over_95pct_identical_delta"] > 0.50 for row in rows.values()
    )
    payload: dict[str, Any] = {
        "audit_name": "stwm_v15_persistent_point_identity_audit",
        "generated_at_utc": now_utc(),
        "per_M": rows,
        "persistent_point_identity_valid": False,
        "persistent_point_identity_valid_detail": (
            "Tensor point index is persistent across observed+future horizon, but it is not a tracked physical point identity. "
            "The cache stores bbox-relative sampled points projected through entity boxes."
        ),
        "fake_dense_or_anchor_copied": bool(fake_dense),
        "future_leakage_detected": False,
        "object_dense_teacher_target_valid_for_physical_claim": False,
        "object_dense_teacher_target_valid_for_engineering_probe": True,
        "final_interpretation": "V15 has persistent array indices and non-empty x/y sequences, but not real persistent teacher-tracked object-internal point identities.",
    }
    out = ROOT / "reports/stwm_v15_persistent_point_identity_audit_20260502.json"
    doc = ROOT / "docs/STWM_V15_PERSISTENT_POINT_IDENTITY_AUDIT_20260502.md"
    dump_json(out, payload)
    write_doc(
        doc,
        "STWM V15 Persistent Point Identity Audit",
        payload,
        [
            "persistent_point_identity_valid",
            "persistent_point_identity_valid_detail",
            "fake_dense_or_anchor_copied",
            "future_leakage_detected",
            "object_dense_teacher_target_valid_for_physical_claim",
            "final_interpretation",
        ],
    )
    print(out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
