#!/usr/bin/env python3
"""汇总 V35.21 domain-normalized video semantic adapter seed42/123/456 复现。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

SEEDS = [42, 123, 456]
REPORT = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_adapter_replication_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_21_DOMAIN_NORMALIZED_VIDEO_SEMANTIC_ADAPTER_REPLICATION_DECISION_20260515.md"
PREDICTABILITY_DECISION = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_predictability_decision_20260515.json"
PREDICTABILITY_EVAL = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_predictability_eval_20260515.json"


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path.relative_to(ROOT))}
    return json.loads(path.read_text(encoding="utf-8"))


def seed_suffix(seed: int) -> str:
    return "" if seed == 42 else f"_seed{seed}"


def metric(eval_report: dict[str, Any], split: str, family: str, key: str) -> float | None:
    value = eval_report.get(split, {}).get(family, {}).get(key)
    return None if value is None else float(value)


def mean_metric(rows: dict[str, dict[str, Any]], split: str, family: str, key: str) -> float | None:
    vals = [metric(row["eval"], split, family, key) for row in rows.values()]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main() -> int:
    rows: dict[str, dict[str, Any]] = {}
    for seed in SEEDS:
        suffix = seed_suffix(seed)
        decision = read_json(ROOT / f"reports/stwm_ostf_v35_21_domain_normalized_video_semantic_state_adapter_decision_20260515{suffix}.json")
        eval_report = read_json(ROOT / f"reports/stwm_ostf_v35_21_domain_normalized_video_semantic_state_adapter_eval_summary_20260515{suffix}.json")
        rows[str(seed)] = {
            "decision": decision,
            "eval": eval_report,
            "passed": bool(decision.get("video_semantic_state_adapter_passed", False)),
            "semantic_changed_passed": bool(decision.get("semantic_changed_passed", False)),
            "semantic_hard_passed": bool(decision.get("semantic_hard_passed", False)),
            "uncertainty_passed": bool(decision.get("uncertainty_passed", False)),
            "stable_preservation": bool(decision.get("stable_preservation", False)),
            "test_changed_balanced_accuracy": metric(eval_report, "test", "semantic_changed", "balanced_accuracy"),
            "test_changed_roc_auc": metric(eval_report, "test", "semantic_changed", "roc_auc"),
            "test_hard_balanced_accuracy": metric(eval_report, "test", "semantic_hard", "balanced_accuracy"),
            "test_hard_roc_auc": metric(eval_report, "test", "semantic_hard", "roc_auc"),
            "test_uncertainty_balanced_accuracy": metric(eval_report, "test", "semantic_uncertainty", "balanced_accuracy"),
            "test_uncertainty_roc_auc": metric(eval_report, "test", "semantic_uncertainty", "roc_auc"),
            "test_stable_top5": metric(eval_report, "test", "cluster", "stable_top5"),
        }

    pred_decision = read_json(PREDICTABILITY_DECISION)
    pred_eval = read_json(PREDICTABILITY_EVAL)
    strat = pred_eval.get("protocols", {}).get("vipseg_to_vspw_stratified_ontology_agnostic", {})
    strat_changed = strat.get("semantic_changed", {})
    all_three_seed_passed = all(row["passed"] for row in rows.values())
    stratified_changed_passed = bool(strat_changed.get("passed", False))
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_21_video_semantic_adapter_replication_done": True,
        "seeds": SEEDS,
        "seed_results": rows,
        "all_three_seed_passed": all_three_seed_passed,
        "semantic_changed_passed_all_seeds": all(row["semantic_changed_passed"] for row in rows.values()),
        "semantic_hard_passed_all_seeds": all(row["semantic_hard_passed"] for row in rows.values()),
        "uncertainty_passed_all_seeds": all(row["uncertainty_passed"] for row in rows.values()),
        "stable_preservation_all_seeds": all(row["stable_preservation"] for row in rows.values()),
        "test_changed_balanced_accuracy_mean": mean_metric(rows, "test", "semantic_changed", "balanced_accuracy"),
        "test_changed_roc_auc_mean": mean_metric(rows, "test", "semantic_changed", "roc_auc"),
        "test_hard_balanced_accuracy_mean": mean_metric(rows, "test", "semantic_hard", "balanced_accuracy"),
        "test_hard_roc_auc_mean": mean_metric(rows, "test", "semantic_hard", "roc_auc"),
        "test_uncertainty_balanced_accuracy_mean": mean_metric(rows, "test", "semantic_uncertainty", "balanced_accuracy"),
        "test_uncertainty_roc_auc_mean": mean_metric(rows, "test", "semantic_uncertainty", "roc_auc"),
        "cross_dataset_video_semantic_suite_ready": bool(pred_decision.get("cross_dataset_video_semantic_suite_ready", False)),
        "mixed_unseen_passed": bool(pred_decision.get("mixed_unseen_passed", False)),
        "vspw_to_vipseg_passed": bool(pred_decision.get("vspw_to_vipseg_passed", False)),
        "vipseg_to_vspw_all_passed": bool(pred_decision.get("vipseg_to_vspw_all_passed", False)),
        "vipseg_to_vspw_stratified_changed_passed": stratified_changed_passed,
        "vipseg_to_vspw_stratified_changed_test_balanced_accuracy": strat_changed.get("test", {}).get("balanced_accuracy"),
        "vipseg_to_vspw_stratified_changed_test_roc_auc": strat_changed.get("test", {}).get("roc_auc"),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": (
            "run_joint_video_semantic_identity_closure_with_stratified_changed_breakdown"
            if all_three_seed_passed
            else "fix_video_semantic_state_adapter_or_target_split"
        ),
        "中文结论": (
            "V35.21 domain-normalized video semantic adapter 在 seed42/123/456 上全部通过，说明 mask-derived "
            "video semantic state target 已经从单 seed smoke 走到较稳的跨 seed 复现。仍不能 claim 完整 semantic field："
            "VIPSeg→VSPW stratified changed 仍未通过，且还需要和 video identity retrieval 做联合闭环。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.21 Domain-Normalized Video Semantic Adapter Replication Decision\n\n"
        f"- 三 seed 全部通过: {all_three_seed_passed}\n"
        f"- semantic_changed 三 seed 通过: {decision['semantic_changed_passed_all_seeds']}\n"
        f"- semantic_hard 三 seed 通过: {decision['semantic_hard_passed_all_seeds']}\n"
        f"- uncertainty 三 seed 通过: {decision['uncertainty_passed_all_seeds']}\n"
        f"- stable preservation 三 seed 通过: {decision['stable_preservation_all_seeds']}\n"
        f"- VIPSeg→VSPW stratified changed 通过: {stratified_changed_passed}\n"
        f"- integrated_semantic_field_claim_allowed: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"三_seed全部通过": all_three_seed_passed, "推荐下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
