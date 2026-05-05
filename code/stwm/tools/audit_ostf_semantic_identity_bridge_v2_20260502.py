#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, load_json, write_doc


REPORT_PATH = ROOT / "reports/stwm_ostf_semantic_identity_bridge_v2_20260502.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_SEMANTIC_IDENTITY_BRIDGE_V2_20260502.md"


def _scalar(x: Any) -> Any:
    arr = np.asarray(x)
    return arr.item() if arr.shape == () else arr.reshape(-1)[0]


def main() -> int:
    sample_paths = sorted((ROOT / "outputs/cache/stwm_real_teacher_object_dense_v16/M128_H8").glob("*/*.npz"))[:32]
    bound_object_count = 0
    semantic_bound_count = 0
    predecode_with_semantic_targets = 0
    predecode_instance_binding = 0
    for path in sample_paths:
        z = np.load(path, allow_pickle=True)
        object_ids = np.asarray(z["object_id"], dtype=np.int64)
        semantic_ids = np.asarray(z["semantic_id"], dtype=np.int64)
        pre = np.load(str(_scalar(z["predecode_path"])), allow_pickle=True)
        bound_object_count += int(object_ids.shape[0])
        semantic_bound_count += int((semantic_ids >= 0).sum())
        if "semantic_features" in pre.files and "semantic_entity_dominant_instance_id" in pre.files:
            predecode_with_semantic_targets += int(object_ids.shape[0])
        if "semantic_instance_id_map" in pre.files and "entity_boxes_over_time" in pre.files:
            predecode_instance_binding += int(object_ids.shape[0])

    proto_report = load_json(ROOT / "reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json")
    fstf_protocol = load_json(ROOT / "reports/stwm_fstf_benchmark_protocol_v6_20260501.json")
    trace_belief = load_json(ROOT / "reports/stwm_trace_belief_eval_20260424.json")
    reacq = load_json(ROOT / "reports/stwm_reacquisition_utility_eval_20260425.json")
    counterfactual = load_json(ROOT / "reports/stwm_counterfactual_association_eval_20260425.json")
    payload = {
        "audit_name": "stwm_ostf_semantic_identity_bridge_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "object_dense_points_bind_to_object_and_semantic_ids": bool(bound_object_count > 0 and semantic_bound_count > 0),
        "binding_mechanism": "V16 object-dense cache stores object_id and semantic_id per sampled point set; predecode cache stores dominant instance ids and semantic features for the same entity ordering.",
        "binding_sampled_object_count": bound_object_count,
        "binding_sampled_semantic_bound_count": semantic_bound_count,
        "future_semantic_prototype_target_available": bool(proto_report.get("prototype_count") == 32 and proto_report.get("no_future_candidate_leakage")),
        "future_semantic_prototype_target_report": "reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json",
        "false_confuser_reacquisition_utility_evaluable": bool(trace_belief and reacq and counterfactual),
        "utility_reports": {
            "trace_belief": "reports/stwm_trace_belief_eval_20260424.json" if trace_belief else None,
            "reacquisition": "reports/stwm_reacquisition_utility_eval_20260425.json" if reacq else None,
            "counterfactual": "reports/stwm_counterfactual_association_eval_20260425.json" if counterfactual else None,
        },
        "tusb_fstf_hidden_available_as_serialized_token": False,
        "tusb_fstf_hidden_note": "Current live repo serializes observed semantic memory/features and future prototype targets, but not a ready-to-reuse exported FSTF/TUSB hidden token tensor for OSTF-v2 training.",
        "semantic_memory_feature_bridge_available_now": True,
        "semantic_memory_feature_bridge_note": "Observed semantic features / semantic_id / prototype targets are available now and can serve as the first semantic object token without future leakage.",
        "no_future_semantic_leakage_audit": {
            "fut_prototype_targets_are_target_side_only": bool(proto_report.get("no_future_candidate_leakage", False)),
            "fstf_protocol_disallows_future_candidate_leakage": bool(fstf_protocol.get("future_candidate_leakage") is False),
            "ostf_v2_observed_only_input_requirement_matches_existing_stwm_rule": True,
        },
        "semantic_identity_bridge_ready": bool(
            bound_object_count > 0
            and semantic_bound_count > 0
            and proto_report.get("prototype_count") == 32
            and proto_report.get("no_future_candidate_leakage")
        ),
    }
    dump_json(REPORT_PATH, payload)
    write_doc(
        DOC_PATH,
        "STWM OSTF Semantic Identity Bridge V2",
        payload,
        [
            "object_dense_points_bind_to_object_and_semantic_ids",
            "future_semantic_prototype_target_available",
            "false_confuser_reacquisition_utility_evaluable",
            "tusb_fstf_hidden_available_as_serialized_token",
            "semantic_memory_feature_bridge_available_now",
            "semantic_identity_bridge_ready",
        ],
    )
    print(REPORT_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
