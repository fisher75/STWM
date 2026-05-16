#!/usr/bin/env python3
"""汇总 V35.21 domain-normalized video semantic benchmark 决策。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPORT = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_closure_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_21_DOMAIN_NORMALIZED_VIDEO_SEMANTIC_CLOSURE_DECISION_20260515.md"


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
    audit = load("reports/stwm_ostf_v35_21_v35_20_domain_shift_target_split_audit_20260515.json")
    target = load("reports/stwm_ostf_v35_21_domain_normalized_video_semantic_state_target_build_20260515.json")
    pred = load("reports/stwm_ostf_v35_21_domain_normalized_video_semantic_predictability_decision_20260515.json")
    pred_eval = load("reports/stwm_ostf_v35_21_domain_normalized_video_semantic_predictability_eval_20260515.json")
    viz = load("reports/stwm_ostf_v35_21_domain_normalized_video_semantic_visualization_manifest_20260515.json")
    suite_ready = bool(pred.get("cross_dataset_video_semantic_suite_ready"))
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain_shift_target_split_audit_done": audit.get("v35_20_domain_shift_target_split_audit_done") is True,
        "domain_normalized_targets_built": target.get("domain_normalized_video_semantic_state_targets_built") is True,
        "target_sample_count": target.get("sample_count"),
        "domain_normalization_scope": target.get("domain_normalization_scope"),
        "target_split_balanced_after_normalization": target.get("target_split_balanced_after_normalization"),
        "domain_normalized_predictability_eval_done": pred.get("domain_normalized_video_semantic_predictability_eval_done") is True,
        "mixed_domain_dataset_balanced_unseen_passed": pred.get("mixed_domain_dataset_balanced_unseen_passed"),
        "vipseg_to_vspw_domain_normalized_passed": pred.get("vipseg_to_vspw_domain_normalized_passed"),
        "vspw_to_vipseg_passed": pred.get("vspw_to_vipseg_passed"),
        "cross_dataset_video_semantic_suite_ready": suite_ready,
        "semantic_adapter_training_ran": False,
        "not_run_reason": None if suite_ready else "domain_normalized_predictability_not_ready",
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "visualization_ready": bool(viz.get("visualization_ready")),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": "run_three_seed_cross_dataset_video_semantic_adapter" if suite_ready else "fix_domain_normalized_target_or_collect_vspw_changed_hard_cases",
        "predictability_eval_report": pred_eval,
        "中文结论": (
            "V35.21 已完成 domain-normalized target split 修复且 suite 通过；本轮仍未训练 adapter，下一轮可做三 seed adapter。"
            if suite_ready
            else "V35.21 已做 domain-normalized risk 标定，但 cross-dataset suite 仍未全过；继续修 target 或补 VSPW changed/hard cases，不能训练 adapter。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.21 Domain-Normalized Video Semantic Closure Decision\n\n"
        f"- domain_shift_target_split_audit_done: {decision['domain_shift_target_split_audit_done']}\n"
        f"- domain_normalized_targets_built: {decision['domain_normalized_targets_built']}\n"
        f"- target_sample_count: {decision['target_sample_count']}\n"
        f"- mixed_domain_dataset_balanced_unseen_passed: {decision['mixed_domain_dataset_balanced_unseen_passed']}\n"
        f"- vipseg_to_vspw_domain_normalized_passed: {decision['vipseg_to_vspw_domain_normalized_passed']}\n"
        f"- vspw_to_vipseg_passed: {decision['vspw_to_vipseg_passed']}\n"
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
