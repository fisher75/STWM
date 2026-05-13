#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


AUDIT = ROOT / "reports/stwm_ostf_v34_11_v34_10_semantic_measurement_failure_audit_20260513.json"
QUALITY = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_quality_probe_20260513.json"
LOCAL_DECISION = ROOT / "reports/stwm_ostf_v34_11_local_semantic_usage_oracle_probe_decision_20260513.json"
REPAIR = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_bank_repair_20260513.json"
VIS = ROOT / "reports/stwm_ostf_v34_11_semantic_measurement_causality_visualization_manifest_20260513.json"
V3410 = ROOT / "reports/stwm_ostf_v34_10_decision_20260512.json"
REPORT = ROOT / "reports/stwm_ostf_v34_11_decision_20260513.json"
DOC = ROOT / "docs/STWM_OSTF_V34_11_DECISION_20260513.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def split_val(src: dict[str, Any], key: str, fallback: dict[str, Any]) -> Any:
    return src.get(key, fallback.get(key))


def main() -> int:
    audit = load(AUDIT)
    quality = load(QUALITY)
    local = load(LOCAL_DECISION)
    repair = load(REPAIR)
    vis = load(VIS)
    v3410 = load(V3410)
    local_ran = bool(local.get("local_semantic_usage_probe_ran", False))
    local_passed = local.get("local_semantic_usage_probe_passed", "not_run")
    if not quality.get("semantic_measurement_quality_passed", False):
        recommended = "fix_semantic_measurement_bank"
    elif local_ran and local_passed is not True:
        recommended = "fix_local_semantic_usage_loss"
    elif local_ran and local.get("semantic_measurements_load_bearing_on_residual") and not local.get("assignment_load_bearing_on_residual"):
        recommended = "fix_assignment_bound_residual_model"
    elif local_ran and local_passed is True:
        recommended = "fix_residual_gate"
    else:
        recommended = "fix_local_semantic_usage_loss"
    semantic_hard = split_val(local, "semantic_hard_signal", v3410)
    changed = split_val(local, "changed_semantic_signal", v3410)
    stable = split_val(local, "stable_preservation", v3410)
    payload: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "中文结论": "V34.11 完成 semantic measurement 因果失败审计和 measurement quality probe；measurement 本身有信息量，但 local usage oracle 仍未形成可通过 gate 的 semantic field claim。",
        "semantic_measurement_failure_audit_done": bool(AUDIT.exists()),
        "semantic_measurement_quality_probe_done": bool(quality.get("semantic_measurement_quality_probe_done", False)),
        "semantic_measurement_quality_passed": bool(quality.get("semantic_measurement_quality_passed", False)),
        "measurement_beats_random": bool(quality.get("measurement_beats_random", False)),
        "measurement_beats_pointwise_on_hard": bool(quality.get("measurement_beats_pointwise_on_hard", False)),
        "measurement_beats_pointwise_on_changed": bool(quality.get("measurement_beats_pointwise_on_changed", False)),
        "local_semantic_usage_probe_ran": local_ran,
        "local_semantic_usage_probe_passed": local_passed,
        "semantic_measurement_bank_repair_ran": bool(repair.get("semantic_measurement_bank_repair_ran", False)),
        "best_measurement_bank": repair.get("best_measurement_bank", "v34_9_trace_preserving_clip_vit_b32_local"),
        "v30_backbone_frozen": bool(split_val(local, "v30_backbone_frozen", v3410)),
        "future_leakage_detected": bool(split_val(local, "future_leakage_detected", v3410)),
        "trajectory_degraded": bool(split_val(local, "trajectory_degraded", v3410)),
        "semantic_hard_signal": semantic_hard,
        "changed_semantic_signal": changed,
        "stable_preservation": stable,
        "pointwise_baseline_dominates": bool(split_val(local, "pointwise_baseline_dominates", v3410)),
        "causal_assignment_subset_gain": split_val(local, "causal_assignment_subset_gain", v3410),
        "strict_residual_subset_gain": split_val(local, "strict_residual_subset_gain", v3410),
        "unit_memory_load_bearing_on_residual": bool(split_val(local, "unit_memory_load_bearing_on_residual", v3410)),
        "semantic_measurements_load_bearing_on_residual": bool(split_val(local, "semantic_measurements_load_bearing_on_residual", v3410)),
        "assignment_load_bearing_on_residual": bool(split_val(local, "assignment_load_bearing_on_residual", v3410)),
        "effective_units": split_val(local, "effective_units", v3410),
        "unit_dominant_instance_purity": split_val(local, "unit_dominant_instance_purity", v3410),
        "unit_semantic_purity": split_val(local, "unit_semantic_purity", v3410),
        "semantic_measurements_have_variance": bool(audit.get("semantic_measurements_have_variance", False)),
        "measurement_confidence_degenerate": bool(audit.get("measurement_confidence_degenerate", False)),
        "teacher_agreement_used_in_training": bool(audit.get("teacher_agreement_used_in_training", False)),
        "semantic_pooling_too_global": bool(audit.get("semantic_pooling_too_global", False)),
        "usage_loss_too_global": bool(audit.get("usage_loss_too_global", False)),
        "semantic_measurement_not_load_bearing_confirmed": bool(audit.get("semantic_measurement_not_load_bearing_confirmed", False)),
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": recommended,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "V34.11 final decision 中文报告",
        payload,
        [
            "中文结论",
            "semantic_measurement_failure_audit_done",
            "semantic_measurement_quality_probe_done",
            "semantic_measurement_quality_passed",
            "measurement_beats_random",
            "measurement_beats_pointwise_on_hard",
            "measurement_beats_pointwise_on_changed",
            "local_semantic_usage_probe_ran",
            "local_semantic_usage_probe_passed",
            "semantic_measurement_bank_repair_ran",
            "best_measurement_bank",
            "v30_backbone_frozen",
            "future_leakage_detected",
            "trajectory_degraded",
            "semantic_hard_signal",
            "changed_semantic_signal",
            "stable_preservation",
            "causal_assignment_subset_gain",
            "strict_residual_subset_gain",
            "unit_memory_load_bearing_on_residual",
            "semantic_measurements_load_bearing_on_residual",
            "assignment_load_bearing_on_residual",
            "effective_units",
            "unit_dominant_instance_purity",
            "unit_semantic_purity",
            "visualization_ready",
            "integrated_identity_field_claim_allowed",
            "integrated_semantic_field_claim_allowed",
            "recommended_next_step",
        ],
    )
    print(f"已写出 V34.11 final decision: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
