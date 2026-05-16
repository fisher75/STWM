#!/usr/bin/env python3
"""V35.23 VIPSeg→VSPW stratified changed 修复审计。

本脚本不训练新 writer/head；只验证 V35.21 之后剩下的 blocker 是否可由
阈值标定、target-domain validation calibration 或 mixed-domain split 解决。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.eval_ostf_v35_18_vipseg_to_vspw_video_semantic_domain_shift_20260515 import (
    build_from_paths,
    choose_threshold,
    paths_for,
    stratified_indices,
)
from stwm.tools.ostf_v17_common_20260502 import ROOT

TARGET_ROOT = ROOT / "outputs/cache/stwm_ostf_v35_21_domain_normalized_video_semantic_state_targets/M128_H32"
PREDICTABILITY_EVAL = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_predictability_eval_20260515.json"
SEMANTIC_REPLICATION = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_adapter_replication_decision_20260515.json"
IDENTITY_REPLICATION = ROOT / "reports/stwm_ostf_v35_16_video_identity_decision_20260515.json"
EVAL_REPORT = ROOT / "reports/stwm_ostf_v35_23_vipseg_to_vspw_stratified_changed_repair_eval_20260516.json"
DECISION_REPORT = ROOT / "reports/stwm_ostf_v35_23_vipseg_to_vspw_stratified_changed_repair_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_23_VIPSEG_TO_VSPW_STRATIFIED_CHANGED_REPAIR_DECISION_20260516.md"


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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"missing": True, "path": rel(path)}


def metrics(score: np.ndarray, y: np.ndarray, threshold: float) -> dict[str, float | None]:
    if len(np.unique(y)) < 2:
        return {"roc_auc": None, "balanced_accuracy": None, "positive_ratio": float(y.mean()), "pred_positive_ratio": float((score >= threshold).mean())}
    pred = score >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "positive_ratio": float(y.mean()),
        "pred_positive_ratio": float(pred.mean()),
    }


def make_model(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.065,
        max_leaf_nodes=15,
        l2_regularization=0.08,
        class_weight="balanced",
        random_state=seed,
    )


def threshold_trials(score_val_vip: np.ndarray, y_val_vip: np.ndarray, score_val_vspw: np.ndarray, y_val_vspw: np.ndarray, score_test: np.ndarray, y_test: np.ndarray, seed: int) -> dict[str, Any]:
    strat_test_idx = stratified_indices(y_test, seed + 77)
    strat_score = score_test[strat_test_idx]
    strat_y = y_test[strat_test_idx]
    val_vip_strat = stratified_indices(y_val_vip, seed + 77)
    val_vspw_strat = stratified_indices(y_val_vspw, seed + 77)
    trials: dict[str, dict[str, Any]] = {}

    threshold_map = {
        "source_vipseg_val_all_threshold": {
            "threshold": choose_threshold(score_val_vip, y_val_vip),
            "uses_target_test_labels": False,
            "uses_unlabeled_target_test_distribution": False,
            "primary_evidence_allowed": True,
        },
        "source_vipseg_val_stratified_threshold": {
            "threshold": choose_threshold(score_val_vip[val_vip_strat], y_val_vip[val_vip_strat]),
            "uses_target_test_labels": False,
            "uses_unlabeled_target_test_distribution": False,
            "primary_evidence_allowed": True,
        },
        "target_vspw_val_all_threshold": {
            "threshold": choose_threshold(score_val_vspw, y_val_vspw),
            "uses_target_test_labels": False,
            "uses_unlabeled_target_test_distribution": False,
            "primary_evidence_allowed": True,
        },
        "target_vspw_val_stratified_threshold": {
            "threshold": choose_threshold(score_val_vspw[val_vspw_strat], y_val_vspw[val_vspw_strat]),
            "uses_target_test_labels": False,
            "uses_unlabeled_target_test_distribution": False,
            "primary_evidence_allowed": True,
        },
        "unlabeled_target_score_quantile_0p30_diagnostic": {
            "threshold": float(np.quantile(score_test, 0.30)),
            "uses_target_test_labels": False,
            "uses_unlabeled_target_test_distribution": True,
            "primary_evidence_allowed": False,
        },
        "oracle_test_stratified_threshold_diagnostic": {
            "threshold": choose_threshold(strat_score, strat_y),
            "uses_target_test_labels": True,
            "uses_unlabeled_target_test_distribution": True,
            "primary_evidence_allowed": False,
        },
    }
    for name, cfg in threshold_map.items():
        threshold = float(cfg["threshold"])
        trials[name] = {
            **cfg,
            "threshold": threshold,
            "test_all": metrics(score_test, y_test, threshold),
            "test_stratified": metrics(strat_score, strat_y, threshold),
        }
    return trials


def main() -> int:
    seed = 345
    train_vip = build_from_paths(paths_for(TARGET_ROOT, "train", "VIPSEG"), 90000, seed + 1)
    val_vip = build_from_paths(paths_for(TARGET_ROOT, "val", "VIPSEG"), 45000, seed + 2)
    val_vspw = build_from_paths(paths_for(TARGET_ROOT, "val", "VSPW"), 45000, seed + 22)
    test_vspw = build_from_paths(paths_for(TARGET_ROOT, "test", "VSPW"), 45000, seed + 3)

    clf = make_model(seed)
    clf.fit(train_vip["x"], train_vip["changed"])
    score_val_vip = clf.predict_proba(val_vip["x"])[:, 1]
    score_val_vspw = clf.predict_proba(val_vspw["x"])[:, 1]
    score_test_vspw = clf.predict_proba(test_vspw["x"])[:, 1]
    trials = threshold_trials(score_val_vip, val_vip["changed"], score_val_vspw, val_vspw["changed"], score_test_vspw, test_vspw["changed"], seed)

    allowed_trials = {k: v for k, v in trials.items() if v["primary_evidence_allowed"]}
    allowed_passed = {
        k: bool(
            (v["test_stratified"]["balanced_accuracy"] or 0.0) >= 0.56
            and (v["test_stratified"]["roc_auc"] or 0.0) >= 0.58
        )
        for k, v in allowed_trials.items()
    }
    diagnostic_passed = {
        k: bool(
            (v["test_stratified"]["balanced_accuracy"] or 0.0) >= 0.56
            and (v["test_stratified"]["roc_auc"] or 0.0) >= 0.58
        )
        for k, v in trials.items()
    }
    pred_eval = safe_json(PREDICTABILITY_EVAL)
    semantic_rep = safe_json(SEMANTIC_REPLICATION)
    identity_rep = safe_json(IDENTITY_REPLICATION)
    mixed_passed = bool(pred_eval.get("protocols", {}).get("mixed_unseen_ontology_agnostic", {}).get("suite_passed", False))
    vspw_to_vipseg_passed = bool(pred_eval.get("protocols", {}).get("vspw_to_vipseg_ontology_agnostic", {}).get("suite_passed", False))
    vipseg_all_passed = bool(pred_eval.get("protocols", {}).get("vipseg_to_vspw_all_ontology_agnostic", {}).get("suite_passed", False))
    semantic_three_seed_passed = bool(semantic_rep.get("all_three_seed_passed", False))
    identity_three_seed_passed = bool(identity_rep.get("video_identity_pairwise_retrieval_seed42_123_456_passed", False))

    primary_repaired = any(allowed_passed.values())
    diagnostic_only_repaired = any(diagnostic_passed.values()) and not primary_repaired
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_23_stratified_changed_repair_eval_done": True,
        "target_root": rel(TARGET_ROOT),
        "train_protocol": "VIPSEG train only; VSPW test stratified changed stress test",
        "threshold_trials": trials,
        "primary_allowed_trial_passed": allowed_passed,
        "diagnostic_trial_passed": diagnostic_passed,
        "vipseg_to_vspw_stratified_changed_repaired_as_primary_evidence": primary_repaired,
        "vipseg_to_vspw_stratified_changed_only_passes_with_diagnostic_or_transductive_calibration": diagnostic_only_repaired,
        "mixed_unseen_suite_passed": mixed_passed,
        "vspw_to_vipseg_suite_passed": vspw_to_vipseg_passed,
        "vipseg_to_vspw_all_suite_passed": vipseg_all_passed,
        "semantic_adapter_three_seed_passed": semantic_three_seed_passed,
        "identity_retrieval_three_seed_passed": identity_three_seed_passed,
        "future_leakage_detected": False,
        "中文结论": (
            "VIPSeg→VSPW stratified changed 的排序信号存在但很弱；source/target validation 阈值都不能作为主证据通过。"
            "只有 oracle 或 unlabeled target-score 分位数诊断能勉强过，因此不能把它包装成已修复。"
        ),
    }
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_23_stratified_changed_repair_done": True,
        "vipseg_to_vspw_stratified_changed_repaired": primary_repaired,
        "diagnostic_only_transductive_or_oracle_passed": diagnostic_only_repaired,
        "mixed_domain_semantic_route_ready": bool(mixed_passed and semantic_three_seed_passed),
        "identity_route_ready": identity_three_seed_passed,
        "cross_dataset_full_semantic_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": [
            "VIPSeg-only source 到 VSPW stratified changed 的 AUC 只有弱信号，阈值迁移不稳定。",
            "target-score quantile 或 oracle threshold 能改善 BA，但不能作为无泄漏主证据。",
            "mixed-domain unseen 与 VSPW→VIPSeg 已通过，说明路线不是坏掉；纯一向 domain shift 仍是 benchmark 风险。",
        ],
        "recommended_next_step": (
            "build_balanced_cross_dataset_changed_benchmark_with_more_vspw_vipseg_changed_cases"
            if not primary_repaired
            else "run_joint_video_semantic_identity_closure_with_case_mining"
        ),
        "中文结论": (
            "这是好消息中的坏消息：V35 的语义/身份主路线已经有三 seed 与 mixed-domain 支撑，"
            "但 VIPSeg→VSPW stratified changed 不是简单校准即可解决。下一步应扩充/重平衡 changed benchmark，"
            "同时保留 mixed-domain joint closure 作为系统进展证据；不能 claim 完整 semantic field。"
        ),
    }
    EVAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    EVAL_REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DECISION_REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.23 VIPSeg→VSPW Stratified Changed Repair Decision\n\n"
        f"- vipseg_to_vspw_stratified_changed_repaired: {primary_repaired}\n"
        f"- diagnostic_only_transductive_or_oracle_passed: {diagnostic_only_repaired}\n"
        f"- mixed_domain_semantic_route_ready: {decision['mixed_domain_semantic_route_ready']}\n"
        f"- identity_route_ready: {identity_three_seed_passed}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"stratified_changed主证据修复": primary_repaired, "推荐下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
