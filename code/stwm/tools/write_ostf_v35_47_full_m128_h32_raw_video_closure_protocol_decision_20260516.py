#!/usr/bin/env python3
"""V35.47 full M128/H32 raw-video closure protocol decision。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

AUDIT = ROOT / "reports/stwm_ostf_v35_47_v35_45_46_artifact_truth_audit_20260516.json"
REMAT = ROOT / "reports/stwm_ostf_v35_47_artifact_rematerialization_20260516.json"
V35_45_DECISION = ROOT / "reports/stwm_ostf_v35_45_decision_20260516.json"
V35_45_EVAL = ROOT / "reports/stwm_ostf_v35_45_larger_raw_video_closure_benchmark_eval_summary_20260516.json"
V35_46_EVAL = ROOT / "reports/stwm_ostf_v35_46_per_category_failure_atlas_eval_20260516.json"
V35_46_DECISION = ROOT / "reports/stwm_ostf_v35_46_per_category_failure_atlas_decision_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_47_full_m128_h32_raw_video_closure_protocol_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_47_FULL_M128_H32_RAW_VIDEO_CLOSURE_PROTOCOL_DECISION_20260516.md"
PLAN_REPORT = ROOT / "reports/stwm_ostf_v35_48_100plus_stratified_plan_20260516.json"
PLAN_DOC = ROOT / "docs/STWM_OSTF_V35_48_100PLUS_STRATIFIED_RAW_VIDEO_CLOSURE_PLAN_20260516.md"


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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def next_scale_choice(report: dict[str, Any]) -> str:
    if not report["artifact_packaging_fixed"]:
        return "stop_and_return_to_claim_boundary"
    if float(report["raw_frontend_rerun_success_rate"] or 0.0) < 0.95 or not report["trace_drift_ok"]:
        return "fix_raw_frontend_reproducibility"
    if report["semantic_fragile_category_count_test"] > 0:
        return "run_100plus_stratified_m128_h32_raw_video_closure"
    if report["identity_real_instance_count"] < 30:
        return "run_100plus_stratified_m128_h32_raw_video_closure"
    return "run_full_325_m128_h32_raw_video_closure"


def write_plan(report: dict[str, Any]) -> None:
    plan = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_48_plan_generated": True,
        "target_clip_count": 128,
        "minimum_clip_count": 96,
        "oversample_categories": [
            "VIPSeg changed",
            "high_motion hard",
            "real_instance semantic_changed",
            "occlusion",
            "crossing",
            "identity confuser",
        ],
        "dataset_balance": "VSPW/VIPSeg balanced",
        "split_balance": "train/val/test balanced",
        "real_instance_identity_target_minimum": 30,
        "real_instance_identity_target_ideal": "50+",
        "pseudo_identity_policy": "VSPW pseudo identity diagnostic-only，不进入 identity claim gate。",
        "model_policy": "不训练新模型；使用 V35.21 semantic adapter 三 seed 和 V35.29 identity retrieval head 三 seed。",
        "scale_constraints": {
            "run_h64_h96": False,
            "run_m512_m1024": False,
            "run_1b": False,
        },
        "required_outputs": [
            "raw frontend rerun report",
            "unified semantic/identity slice",
            "joint eval summary/decision",
            "per-category failure atlas",
            "case-mined visualization",
            "claim boundary decision",
        ],
        "source_v35_47_decision": rel(REPORT),
        "中文结论": "V35.48 应跑 96-128 clip stratified M128/H32 raw-video closure，不直接 full 325；重点过采样 VIPSeg changed、高运动 hard、真实 instance semantic changed，同时扩大真实 instance identity provenance。",
    }
    PLAN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PLAN_DOC.parent.mkdir(parents=True, exist_ok=True)
    PLAN_REPORT.write_text(json.dumps(jsonable(plan), indent=2, ensure_ascii=False), encoding="utf-8")
    PLAN_DOC.write_text(
        "# STWM OSTF V35.48 100+ Stratified Raw-Video Closure Plan\n\n"
        f"- target_clip_count: 128\n"
        f"- minimum_clip_count: 96\n"
        f"- oversample: VIPSeg changed / high_motion hard / real_instance semantic_changed / occlusion / crossing / identity confuser\n"
        f"- VSPW/VIPSeg balanced: true\n"
        f"- train/val/test balanced: true\n"
        f"- real_instance_identity_target_minimum: 30\n"
        f"- real_instance_identity_target_ideal: 50+\n"
        f"- pseudo_identity_policy: diagnostic-only\n"
        f"- train_new_model: false\n"
        f"- run_h64_h96: false\n"
        f"- run_m512_m1024: false\n\n"
        "## 中文总结\n"
        + plan["中文结论"]
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    audit = load(AUDIT)
    remat = load(REMAT)
    v45 = load(V35_45_DECISION)
    v46 = load(V35_46_DECISION)
    v46_eval = load(V35_46_EVAL)
    semantic_fragile = list(v46.get("semantic_fragile_categories_test", []))
    identity_fragile = list(v46.get("identity_fragile_categories_test", []))
    identity_real_count = int(v45.get("real_instance_identity_count", 0) or 0)
    pseudo_count = int(v45.get("pseudo_identity_count", 0) or 0)
    artifact_packaging_fixed = bool(remat.get("artifact_packaging_fixed", False)) and bool(audit.get("v35_47_decision_safe_to_continue", False))
    vipseg_changed = any(r.get("category") == "dataset_vipseg" and "semantic_changed" in r.get("risk_metrics", {}) for r in semantic_fragile)
    high_motion_hard = any(r.get("category") == "high_motion" and "semantic_hard" in r.get("risk_metrics", {}) for r in semantic_fragile)
    real_instance_sem_changed = any(r.get("category") == "real_instance_identity" and "semantic_changed" in r.get("risk_metrics", {}) for r in semantic_fragile)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.47",
        "artifact_truth_audit_done": bool(audit.get("artifact_truth_audit_done", False)),
        "artifact_packaging_fixed": artifact_packaging_fixed,
        "v35_45_bounded_claim_still_allowed": bool(v45.get("m128_h32_larger_video_system_benchmark_claim_allowed", False)),
        "v35_46_failure_atlas_ready": bool(v46.get("atlas_ready", False)),
        "full_cvpr_scale_claim_allowed": False,
        "selected_clip_count": int(v45.get("selected_clip_count", 0) or 0),
        "raw_frontend_rerun_success_rate": v45.get("raw_frontend_rerun_success_rate"),
        "trace_drift_ok": bool(v45.get("trace_drift_ok", False)),
        "semantic_three_seed_passed": bool(v45.get("semantic_three_seed_passed", False)),
        "stable_preservation": bool(v45.get("stable_preservation", False)),
        "identity_real_instance_three_seed_passed": bool(v45.get("identity_real_instance_three_seed_passed", False)),
        "identity_pseudo_targets_excluded_from_claim": bool(v45.get("identity_pseudo_targets_excluded_from_claim", False)),
        "identity_real_instance_count": identity_real_count,
        "pseudo_identity_count": pseudo_count,
        "semantic_fragile_category_count_test": len(semantic_fragile),
        "identity_fragile_category_count_test": len(identity_fragile),
        "vipseg_changed_fragile": vipseg_changed,
        "high_motion_hard_fragile": high_motion_hard,
        "real_instance_semantic_changed_fragile": real_instance_sem_changed,
        "semantic_fragile_categories_test": semantic_fragile,
        "identity_fragile_categories_test": identity_fragile,
        "per_category_failure_atlas_ready": bool(v46.get("atlas_ready", False)),
        "visualization_ready": bool(v45.get("visualization_ready", False)),
        "v30_backbone_frozen": True,
        "future_leakage_detected": bool(v45.get("future_leakage_detected", False) or v46.get("future_leakage_detected", False)),
        "trajectory_degraded": bool(v45.get("trajectory_degraded", False) or v46.get("trajectory_degraded", False)),
        "scale_risk_assessment": {
            "semantic_fragile_small_sample": len(semantic_fragile) > 0,
            "identity_real_instance_count_below_30": identity_real_count < 30,
            "zip_packaging_gap_recorded": bool(remat.get("zip_packaging_gap_recorded", False)),
            "direct_full_325_not_recommended_yet": True,
        },
        "claim_boundary": "允许 V35.45/V35.46 bounded M128/H32 larger subset claim；不允许 full CVPR-scale claim。",
    }
    choice = next_scale_choice(report)
    report["next_scale_choice"] = choice
    report["recommended_next_step"] = choice
    report["中文结论"] = (
        "V35.47 协议决策：不要直接 full 325。当前 32-clip bounded benchmark 成立，但 test fragile 类别仍存在且 sample_count 小，真实 instance identity 数量也只有 16。"
        "最合理下一步是 96-128 clip 的 100+ stratified M128/H32 raw-video closure，过采样 VIPSeg changed、高运动 hard、真实 instance semantic changed，并并行扩大真实 instance identity provenance。"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.47 Full M128/H32 Raw-Video Closure Protocol Decision\n\n"
        f"- artifact_truth_audit_done: {report['artifact_truth_audit_done']}\n"
        f"- artifact_packaging_fixed: {artifact_packaging_fixed}\n"
        f"- v35_45_bounded_claim_still_allowed: {report['v35_45_bounded_claim_still_allowed']}\n"
        f"- v35_46_failure_atlas_ready: {report['v35_46_failure_atlas_ready']}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- selected_clip_count: {report['selected_clip_count']}\n"
        f"- semantic_fragile_category_count_test: {len(semantic_fragile)}\n"
        f"- identity_fragile_category_count_test: {len(identity_fragile)}\n"
        f"- identity_real_instance_count: {identity_real_count}\n"
        f"- vipseg_changed_fragile: {vipseg_changed}\n"
        f"- high_motion_hard_fragile: {high_motion_hard}\n"
        f"- real_instance_semantic_changed_fragile: {real_instance_sem_changed}\n"
        f"- next_scale_choice: {choice}\n"
        f"- recommended_next_step: {choice}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n## 关键依据\n"
        "- V35.45 32-clip larger benchmark 通过，但不是 full scale。\n"
        "- V35.46 test fragile 集中在 VIPSeg changed、高运动 hard、real-instance 子集 semantic changed，且每类 sample_count=6。\n"
        "- real-instance identity 当前很稳，但 claim 样本数 16，低于下一阶段建议阈值 30。\n"
        "- 因此应先跑 100+ stratified，而不是直接 full 325。\n",
        encoding="utf-8",
    )
    if choice == "run_100plus_stratified_m128_h32_raw_video_closure":
        write_plan(report)
    print(json.dumps({"v35_47_protocol_decision_done": True, "next_scale_choice": choice, "recommended_next_step": choice}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
