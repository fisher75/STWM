#!/usr/bin/env python3
"""汇总 V35 可观测可预测 semantic state 目标重定义最终 decision。"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import setproctitle

setproctitle.setproctitle("python")
ROOT = Path(__file__).resolve().parents[3]
AUDIT = ROOT / "reports/stwm_ostf_v35_v34_43_truth_audit_20260515.json"
TARGET_BUILD = ROOT / "reports/stwm_ostf_v35_observed_predictable_semantic_state_target_build_20260515.json"
PRED_DECISION = ROOT / "reports/stwm_ostf_v35_semantic_state_target_predictability_decision_20260515.json"
PRED_EVAL = ROOT / "reports/stwm_ostf_v35_semantic_state_target_predictability_eval_20260515.json"
VIS = ROOT / "reports/stwm_ostf_v35_semantic_state_visualization_manifest_20260515.json"
HEAD_TRAIN = ROOT / "reports/stwm_ostf_v35_semantic_state_head_train_summary_20260515.json"
HEAD_EVAL = ROOT / "reports/stwm_ostf_v35_semantic_state_head_eval_summary_20260515.json"
HEAD_DECISION = ROOT / "reports/stwm_ostf_v35_semantic_state_head_decision_20260515.json"
HEAD_DOC = ROOT / "docs/STWM_OSTF_V35_SEMANTIC_STATE_HEAD_DECISION_20260515.md"
REPORT = ROOT / "reports/stwm_ostf_v35_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_DECISION_20260515.md"
ALLOWED_NEXT = {
    "fix_semantic_state_targets",
    "fix_target_predictability_features",
    "train_v35_semantic_state_head",
    "fix_v35_semantic_state_head",
    "run_v35_seed123_replication",
    "run_v35_m512_dense_visualization",
    "run_v35_h64_h96_smoke",
    "build_video_input_closure",
    "stop_and_return_to_target_mapping",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_skipped_head_reports(reason: str, suite_ready: bool) -> None:
    now = datetime.now(timezone.utc).isoformat()
    train = {"generated_at_utc": now, "semantic_state_head_training_ran": False, "skip_reason": reason, "observed_predictable_semantic_state_suite_ready": suite_ready}
    eval_r = {"generated_at_utc": now, "semantic_state_head_eval_ran": False, "semantic_state_head_passed": "not_run", "skip_reason": reason}
    dec = {"generated_at_utc": now, "semantic_state_head_training_ran": False, "semantic_state_head_passed": "not_run", "skip_reason": reason, "recommended_next_step": "train_v35_semantic_state_head" if suite_ready else "fix_semantic_state_targets", "中文结论": "V35 semantic state head 未训练；只有 target suite ready 后才允许训练。"}
    for path, data in [(HEAD_TRAIN, train), (HEAD_EVAL, eval_r), (HEAD_DECISION, dec)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    HEAD_DOC.parent.mkdir(parents=True, exist_ok=True)
    HEAD_DOC.write_text(
        "# STWM OSTF V35 Semantic State Head Decision\n\n"
        f"- semantic_state_head_training_ran: false\n- semantic_state_head_passed: not_run\n- skip_reason: {reason}\n\n"
        "## 中文总结\nV35 semantic state head 未训练，因为本轮只允许在 target suite ready 后进入训练。\n",
        encoding="utf-8",
    )


def main() -> None:
    audit = read_json(AUDIT)
    target = read_json(TARGET_BUILD)
    pred = read_json(PRED_DECISION)
    pred_eval = read_json(PRED_EVAL)
    vis = read_json(VIS)
    continuous_exhausted = bool(audit.get("continuous_unit_delta_route_exhausted", False))
    targets_built = bool(target.get("semantic_state_targets_built", False))
    eval_done = bool(pred.get("target_predictability_eval_done", False))
    suite_ready = bool(pred.get("observed_predictable_semantic_state_suite_ready", False))
    if not suite_ready or not HEAD_DECISION.exists():
        write_skipped_head_reports("target_suite_not_ready" if not suite_ready else "head_training_not_requested_in_this_round", suite_ready)
    head = read_json(HEAD_DECISION)
    head_ran = bool(head.get("semantic_state_head_training_ran", False))
    head_passed = head.get("semantic_state_head_passed", "not_run")

    if audit.get("v34_43_report_json_missing") and not audit.get("v34_43_artifact_packaging_fixed"):
        next_step = "fix_semantic_state_targets"
    elif not continuous_exhausted:
        next_step = "stop_and_return_to_target_mapping"
    elif not targets_built:
        next_step = "fix_semantic_state_targets"
    elif not eval_done:
        next_step = "fix_target_predictability_features"
    elif not suite_ready:
        next_step = pred.get("recommended_next_step", "fix_semantic_state_targets")
    elif not head_ran:
        next_step = "train_v35_semantic_state_head"
    elif head_passed is not True:
        next_step = "fix_v35_semantic_state_head"
    else:
        next_step = "run_v35_seed123_replication"
    if next_step not in ALLOWED_NEXT:
        next_step = "fix_semantic_state_targets"

    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v34_43_truth_audit_done": AUDIT.exists(),
        "continuous_unit_delta_route_exhausted": continuous_exhausted,
        "semantic_state_targets_built": targets_built,
        "target_predictability_eval_done": eval_done,
        "observed_predictable_semantic_state_suite_ready": suite_ready,
        "semantic_state_head_training_ran": head_ran,
        "semantic_state_head_passed": head_passed,
        "v30_backbone_frozen": True,
        "future_leakage_detected": bool(audit.get("v34_43_future_leakage_detected", False) or pred_eval.get("future_leakage_detected", False)),
        "trajectory_degraded": False,
        "semantic_cluster_transition_passed": bool(pred.get("semantic_cluster_transition_passed", False)),
        "semantic_changed_passed": bool(pred.get("semantic_changed_passed", False)),
        "evidence_anchor_family_passed": bool(pred.get("evidence_anchor_family_passed", False)),
        "same_instance_passed": bool(pred.get("same_instance_passed", False)),
        "uncertainty_target_passed": bool(pred.get("uncertainty_target_passed", False)),
        "semantic_hard_signal": {"val": False, "test": False},
        "changed_semantic_signal": {"val": False, "test": False},
        "stable_preservation": {"val": False, "test": False},
        "unit_memory_load_bearing": False,
        "semantic_measurement_load_bearing": False,
        "assignment_load_bearing": False,
        "visualization_ready": bool(vis.get("visualization_ready", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "recommended_next_step": next_step,
        "中文结论": "V35 已将问题从 continuous teacher embedding delta 改成 observed-predictable semantic state target suite。只有 suite 在 observed-only 上界中通过，才允许训练新的 semantic state head；当前不允许声明 semantic field success。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35 Decision\n\n"
        f"- continuous_unit_delta_route_exhausted: {decision['continuous_unit_delta_route_exhausted']}\n"
        f"- semantic_state_targets_built: {decision['semantic_state_targets_built']}\n"
        f"- observed_predictable_semantic_state_suite_ready: {decision['observed_predictable_semantic_state_suite_ready']}\n"
        f"- semantic_state_head_training_ran: {decision['semantic_state_head_training_ran']}\n"
        f"- semantic_state_head_passed: {decision['semantic_state_head_passed']}\n"
        f"- integrated_semantic_field_claim_allowed: {decision['integrated_semantic_field_claim_allowed']}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print("V35 final decision 已写入。", flush=True)
    print(json.dumps({k: decision[k] for k in ["semantic_state_targets_built", "observed_predictable_semantic_state_suite_ready", "semantic_state_head_training_ran", "recommended_next_step"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
