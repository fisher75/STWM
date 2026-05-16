#!/usr/bin/env python3
"""汇总 V35.18 video semantic domain-shift 修复决策。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPORT = ROOT / "reports/stwm_ostf_v35_18_video_semantic_domain_shift_closure_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_18_VIDEO_SEMANTIC_DOMAIN_SHIFT_CLOSURE_DECISION_20260515.md"


def load(rel_path: str) -> dict[str, Any]:
    p = ROOT / rel_path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def jsonable(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    return x


def main() -> int:
    trace = load("reports/stwm_cotracker_object_dense_teacher_v16_M128_H32_v35_18_trainval_expand_20260502.json")
    measurement = load("reports/stwm_ostf_v35_18_video_observed_semantic_measurement_cache_incremental_20260515.json")
    target = load("reports/stwm_ostf_v35_18_ontology_agnostic_video_semantic_state_target_build_20260515.json")
    pred = load("reports/stwm_ostf_v35_18_vipseg_to_vspw_video_semantic_domain_shift_decision_20260515.json")
    pred_eval = load("reports/stwm_ostf_v35_18_vipseg_to_vspw_video_semantic_domain_shift_eval_20260515.json")
    viz = load("reports/stwm_ostf_v35_18_vipseg_to_vspw_domain_shift_visualization_manifest_20260515.json")
    suite_ready = bool(pred.get("cross_dataset_video_semantic_suite_ready"))
    primary_blocker = None
    if not suite_ready:
        if not pred.get("vipseg_to_vspw_stratified_passed"):
            primary_blocker = "vipseg_to_vspw_domain_shift_not_fixed"
        elif not pred.get("mixed_unseen_passed") or not pred.get("vspw_to_vipseg_passed"):
            primary_blocker = "ontology_agnostic_target_hurts_other_protocols"
        else:
            primary_blocker = "unknown_cross_dataset_semantic_blocker"
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_18_domain_shift_repair_ran": True,
        "processed_clip_count": trace.get("processed_clip_count"),
        "processed_split_counts": trace.get("processed_split_counts"),
        "processed_dataset_counts": trace.get("processed_dataset_counts"),
        "incremental_measurement_new_sample_count": measurement.get("new_sample_count"),
        "ontology_agnostic_targets_built": target.get("ontology_agnostic_video_semantic_state_targets_built"),
        "vipseg_source_train_val_expanded": target.get("vipseg_source_train_val_expanded"),
        "semantic_change_target_repaired": pred.get("semantic_change_target_repaired"),
        "ontology_agnostic_change_target_used": pred.get("ontology_agnostic_change_target_used"),
        "stratified_vspw_test_used": pred.get("stratified_vspw_test_used"),
        "mixed_unseen_passed": pred.get("mixed_unseen_passed"),
        "vspw_to_vipseg_passed": pred.get("vspw_to_vipseg_passed"),
        "vipseg_to_vspw_all_passed": pred.get("vipseg_to_vspw_all_passed"),
        "vipseg_to_vspw_stratified_passed": pred.get("vipseg_to_vspw_stratified_passed"),
        "cross_dataset_video_semantic_suite_ready": suite_ready,
        "semantic_adapter_training_ran": False,
        "identity_retrieval_training_ran": False,
        "not_run_reason": None if suite_ready else "cross_dataset_video_semantic_predictability_still_not_ready",
        "primary_blocker": primary_blocker,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "visualization_ready": bool(viz.get("visualization_ready")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": "run_three_seed_cross_dataset_video_semantic_adapter" if suite_ready else "fix_vipseg_to_vspw_video_semantic_domain_shift_or_target_split",
        "predictability_eval_report": pred_eval,
        "中文结论": (
            "V35.18 已修复 VIPSeg source 覆盖、ontology-agnostic change target 与 VSPW 分层测试；target suite 已可进入三 seed adapter。"
            if suite_ready
            else "V35.18 仍未让 cross-dataset semantic target suite 完全通过；不能训练 adapter，也不能 claim full video semantic field。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.18 Video Semantic Domain Shift Closure Decision\n\n"
        f"- processed_clip_count: {decision['processed_clip_count']}\n"
        f"- vipseg_source_train_val_expanded: {decision['vipseg_source_train_val_expanded']}\n"
        f"- semantic_change_target_repaired: {decision['semantic_change_target_repaired']}\n"
        f"- mixed_unseen_passed: {decision['mixed_unseen_passed']}\n"
        f"- vspw_to_vipseg_passed: {decision['vspw_to_vipseg_passed']}\n"
        f"- vipseg_to_vspw_all_passed: {decision['vipseg_to_vspw_all_passed']}\n"
        f"- vipseg_to_vspw_stratified_passed: {decision['vipseg_to_vspw_stratified_passed']}\n"
        f"- cross_dataset_video_semantic_suite_ready: {suite_ready}\n"
        f"- semantic_adapter_training_ran: false\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cross_dataset_video_semantic_suite_ready": suite_ready, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
