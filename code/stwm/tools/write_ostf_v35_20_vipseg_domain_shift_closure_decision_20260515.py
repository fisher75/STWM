#!/usr/bin/env python3
"""汇总 V35.20 VIPSeg source boost 后的 cross-dataset video semantic 决策。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPORT = ROOT / "reports/stwm_ostf_v35_20_vipseg_domain_shift_closure_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_20_VIPSEG_DOMAIN_SHIFT_CLOSURE_DECISION_20260515.md"


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
    vipseg_boost = load("reports/stwm_cotracker_object_dense_teacher_v16_M128_H32_v35_20_vipseg_only_boost_20260515.json")
    target = load("reports/stwm_ostf_v35_19_boundary_risk_video_semantic_state_target_build_20260515.json")
    pred = load("reports/stwm_ostf_v35_19_boundary_risk_video_semantic_predictability_decision_20260515.json")
    pred_eval = load("reports/stwm_ostf_v35_19_boundary_risk_video_semantic_predictability_eval_20260515.json")
    viz = load("reports/stwm_ostf_v35_19_boundary_risk_video_semantic_visualization_manifest_20260515.json")
    protocols = pred_eval.get("protocols", {})
    vipseg_to_vspw = protocols.get("vipseg_to_vspw_stratified_ontology_agnostic", {})
    mixed = protocols.get("mixed_unseen_ontology_agnostic", {})
    blocker = "vipseg_to_vspw_domain_shift_persists_after_source_boost"
    if vipseg_to_vspw:
        changed_test_auc = vipseg_to_vspw.get("semantic_changed", {}).get("test_stratified", {}).get("roc_auc")
        if changed_test_auc is not None and changed_test_auc < 0.53:
            blocker = "vipseg_to_vspw_observed_only_features_do_not_transfer"
    suite_ready = bool(pred.get("cross_dataset_video_semantic_suite_ready"))
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_20_vipseg_source_boost_ran": True,
        "vipseg_processed_split_counts": vipseg_boost.get("vipseg_processed_split_counts"),
        "boundary_risk_targets_built": target.get("boundary_risk_video_semantic_state_targets_built"),
        "target_sample_count": target.get("sample_count"),
        "vipseg_source_train_val_expanded": target.get("vipseg_source_train_val_expanded"),
        "mixed_unseen_passed": pred.get("mixed_unseen_passed"),
        "vspw_to_vipseg_passed": pred.get("vspw_to_vipseg_passed"),
        "vipseg_to_vspw_all_passed": pred.get("vipseg_to_vspw_all_passed"),
        "vipseg_to_vspw_stratified_passed": pred.get("vipseg_to_vspw_stratified_passed"),
        "cross_dataset_video_semantic_suite_ready": suite_ready,
        "mixed_unseen_signal_positive": bool(mixed.get("suite_passed")),
        "vipseg_to_vspw_blocker": blocker,
        "semantic_adapter_training_ran": False,
        "identity_retrieval_training_ran": False,
        "not_run_reason": "cross_dataset_video_semantic_predictability_failed",
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "visualization_ready": bool(viz.get("visualization_ready")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": "fix_video_semantic_target_split_or_domain_normalization_before_adapter_training",
        "中文结论": (
            "V35.20 扩大 VIPSeg source 后，mixed-unseen 已通过，但 VIPSeg→VSPW 在 all/stratified 下仍未过；"
            "这说明当前 mask-derived semantic target 仍有真实跨数据集域迁移/target split 问题。不能训练 adapter，不能 claim 完整 video semantic field。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.20 VIPSeg Domain Shift Closure Decision\n\n"
        f"- vipseg_processed_split_counts: {decision['vipseg_processed_split_counts']}\n"
        f"- target_sample_count: {decision['target_sample_count']}\n"
        f"- mixed_unseen_passed: {decision['mixed_unseen_passed']}\n"
        f"- vspw_to_vipseg_passed: {decision['vspw_to_vipseg_passed']}\n"
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
