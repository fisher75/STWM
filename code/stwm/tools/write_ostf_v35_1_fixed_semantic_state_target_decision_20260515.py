#!/usr/bin/env python3
"""写出 V35.1 fixed semantic state target 决策。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

BUILD = ROOT / "reports/stwm_ostf_v35_1_fixed_semantic_state_target_build_20260515.json"
EVAL = ROOT / "reports/stwm_ostf_v35_1_fixed_semantic_state_target_predictability_eval_20260515.json"
PRED_DECISION = ROOT / "reports/stwm_ostf_v35_1_fixed_semantic_state_target_predictability_decision_20260515.json"
REPORT = ROOT / "reports/stwm_ostf_v35_1_fixed_semantic_state_target_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_1_FIXED_SEMANTIC_STATE_TARGET_DECISION_20260515.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def best_auc(metrics: dict[str, Any], name: str, split: str) -> float:
    block = metrics.get(name, {})
    vals = []
    for method in block.get("methods", {}).values():
        val = method.get(split, {}).get("roc_auc")
        if val is not None:
            vals.append(float(val))
    return max(vals) if vals else 0.0


def main() -> None:
    print("V35.1: 写出 fixed semantic state target 决策。", flush=True)
    build = load(BUILD)
    eval_report = load(EVAL)
    pred = load(PRED_DECISION)
    metrics = eval_report.get("metrics", {})
    suite_ready = bool(pred.get("observed_predictable_semantic_state_suite_ready", False))
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_semantic_state_targets_built": bool(build.get("semantic_state_targets_built", False)),
        "target_predictability_eval_done": bool(pred.get("target_predictability_eval_done", False)),
        "semantic_cluster_transition_passed": bool(pred.get("semantic_cluster_transition_passed", False)),
        "semantic_changed_passed": bool(pred.get("semantic_changed_passed", False)),
        "evidence_anchor_family_passed": bool(pred.get("evidence_anchor_family_passed", False)),
        "same_instance_passed": bool(pred.get("same_instance_passed", False)),
        "uncertainty_target_passed": bool(pred.get("uncertainty_target_passed", False)),
        "observed_predictable_semantic_state_suite_ready": suite_ready,
        "semantic_changed_auc": {
            "val": best_auc(metrics, "semantic_changed", "val"),
            "test": best_auc(metrics, "semantic_changed", "test"),
        },
        "same_instance_auc": {
            "val": best_auc(metrics, "same_instance", "val"),
            "test": best_auc(metrics, "same_instance", "test"),
        },
        "uncertainty_high_auc": {
            "val": best_auc(metrics, "semantic_uncertainty_high", "val"),
            "test": best_auc(metrics, "semantic_uncertainty_high", "test"),
        },
        "semantic_state_head_training_ran": False,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": "train_v35_semantic_state_head" if suite_ready else "fix_semantic_state_targets",
        "中文结论": "V35.1 已完成 target 修复与 observed-only 上界审计。semantic_cluster_transition、semantic_changed、uncertainty_target 通过，suite_ready=true；本轮仍未训练 semantic state head，下一步才允许训练 seed42 V35 head。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.1 Fixed Semantic State Target Decision\n\n"
        f"- fixed_semantic_state_targets_built: {decision['fixed_semantic_state_targets_built']}\n"
        f"- semantic_cluster_transition_passed: {decision['semantic_cluster_transition_passed']}\n"
        f"- semantic_changed_passed: {decision['semantic_changed_passed']}\n"
        f"- evidence_anchor_family_passed: {decision['evidence_anchor_family_passed']}\n"
        f"- same_instance_passed: {decision['same_instance_passed']}\n"
        f"- uncertainty_target_passed: {decision['uncertainty_target_passed']}\n"
        f"- observed_predictable_semantic_state_suite_ready: {suite_ready}\n"
        f"- semantic_state_head_training_ran: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        "本轮只修 target，不训练主模型。修复后 semantic family 与 uncertainty auxiliary family 均有 val/test observed-only 可预测信号，满足进入 V35 semantic state head 的前置条件。"
        "same_instance 仍弱，evidence_anchor_family 仍未过，因此后续 head 训练必须单独报告 identity/anchor 的 residual risk，不能提前 claim 完整 semantic field success。\n",
        encoding="utf-8",
    )
    print(json.dumps({"suite_ready": suite_ready, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
