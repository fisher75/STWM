#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM Current Trace-Unit Semantics Audit V15", ""]
    for key in [
        "current_K_meaning",
        "current_can_claim_dense_trace_field",
        "object_internal_point_traces_exist",
        "recommended_wording",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    lines.append("## Entity Count")
    for key, value in payload.get("valid_entity_count_stats", {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## K Scaling Valid Supervision")
    for key, value in payload.get("K_scaling_valid_supervision", {}).items():
        lines.append(f"- {key}: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0}
    arr = sorted(int(v) for v in values)
    return {
        "count": len(arr),
        "mean": float(statistics.fmean(arr)),
        "median": float(statistics.median(arr)),
        "p90": float(np.percentile(np.asarray(arr), 90)),
        "max": int(max(arr)),
    }


def _density_audit() -> dict[str, Any]:
    path = Path("reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json")
    return _load_json(path)


def main() -> int:
    future = np.load("outputs/cache/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428/prototype_targets.npz", allow_pickle=True)
    observed = np.load("outputs/cache/stwm_mixed_observed_semantic_prototype_targets_v2_20260428/observed_proto_targets_c32.npz", allow_pickle=True)
    fkeys = [str(x) for x in future["item_keys"].tolist()]
    obs_index = {str(k): i for i, k in enumerate(observed["item_keys"].tolist())}
    valid_entity_counts: list[int] = []
    for i, key in enumerate(fkeys):
        fmask = np.asarray(future["target_mask"][i], dtype=bool)
        slot_valid = fmask.any(axis=0)
        oi = obs_index.get(key)
        if oi is not None:
            slot_valid = slot_valid | np.asarray(observed["observed_semantic_proto_mask"][oi], dtype=bool)[: slot_valid.shape[0]]
        valid_entity_counts.append(int(slot_valid.sum()))

    density = _density_audit()
    k_supervision = {}
    for name, row in density.get("by_K", {}).items():
        k_supervision[name] = {
            "slot_count": row.get("slot_count"),
            "valid_joint_supervised_slots": row.get("valid_joint_supervised_slots"),
            "new_valid_slots_added_vs_K8": row.get("new_valid_slots_added_vs_K8"),
            "joint_supervision_coverage": row.get("joint_supervision_coverage"),
        }

    payload = {
        "audit_name": "stwm_current_trace_unit_semantics_audit_v15",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_K_meaning": "max_entities_per_sample / semantic entity slots, not object-internal dense trajectories",
        "max_entities_per_sample_config": 8,
        "valid_entity_count_stats": _stats(valid_entity_counts),
        "K_scaling_valid_supervision": k_supervision,
        "K8_K16_K32_add_dense_object_internal_points": False,
        "entity_state_observed_fields": ["bbox center x/y", "valid/visibility-like flag", "velocity x/y", "bbox width/height"],
        "entity_state_only_bbox_center_velocity_visibility": True,
        "object_internal_point_traces_exist": False,
        "current_can_claim_dense_trace_field": False,
        "recommended_wording": "semantic entity trace-unit field; object-dense trace field requires V15 object-internal point supervision",
        "supporting_reports": [
            "reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json",
            "reports/stwm_fstf_scaling_claim_verification_v13_20260502.json",
        ],
    }
    _dump(Path("reports/stwm_current_trace_unit_semantics_audit_v15_20260502.json"), payload)
    _write_doc(Path("docs/STWM_CURRENT_TRACE_UNIT_SEMANTICS_AUDIT_V15_20260502.md"), payload)
    print("reports/stwm_current_trace_unit_semantics_audit_v15_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
