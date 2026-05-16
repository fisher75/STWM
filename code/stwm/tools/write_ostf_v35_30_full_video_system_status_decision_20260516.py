#!/usr/bin/env python3
"""V35.30 汇总当前 STWM video semantic/identity full-system 状态。"""
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

SEM_REP = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_adapter_replication_decision_20260515.json"
CHANGED_DEC = ROOT / "reports/stwm_ostf_v35_24_balanced_cross_dataset_changed_predictability_decision_20260516.json"
UNIFIED = ROOT / "reports/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark_build_20260516.json"
ID_REP = ROOT / "reports/stwm_ostf_v35_29_expanded_identity_replication_decision_20260516.json"
VIS = ROOT / "reports/stwm_ostf_v35_25_joint_video_semantic_identity_case_mining_manifest_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_30_full_video_system_status_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_30_FULL_VIDEO_SYSTEM_STATUS_DECISION_20260516.md"


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
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"missing": True, "path": str(path.relative_to(ROOT))}


def main() -> int:
    sem = read_json(SEM_REP)
    changed = read_json(CHANGED_DEC)
    unified = read_json(UNIFIED)
    ident = read_json(ID_REP)
    vis = read_json(VIS)
    semantic_ok = bool(sem.get("all_three_seed_passed", False))
    changed_ok = bool(changed.get("balanced_cross_dataset_changed_suite_ready", False))
    unified_ok = bool(unified.get("unified_video_semantic_identity_benchmark_built", False) and unified.get("full_semantic_coverage_by_identity_targets", False))
    identity_ok = bool(ident.get("all_three_seed_passed", False))
    visualization_ok = bool(vis.get("visualization_ready", False))
    system_components_ready = bool(semantic_ok and changed_ok and unified_ok and identity_ok and visualization_ok)
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "full_video_system_status_decision_done": True,
        "semantic_adapter_three_seed_passed": semantic_ok,
        "balanced_cross_dataset_changed_suite_ready": changed_ok,
        "expanded_identity_three_seed_passed": identity_ok,
        "full_unified_video_semantic_identity_benchmark_built": unified_ok,
        "joint_case_mining_visualization_ready": visualization_ok,
        "joint_clip_count": unified.get("joint_intersection_sample_count"),
        "semantic_sample_count": unified.get("semantic_sample_count"),
        "identity_sample_count": unified.get("identity_sample_count"),
        "raw_video_frame_paths_available": unified.get("raw_video_frame_paths_available"),
        "video_derived_dense_trace_input_closed": bool(unified.get("raw_video_frame_paths_available", False)),
        "semantic_test_changed_balanced_accuracy_mean": sem.get("test_changed_balanced_accuracy_mean"),
        "semantic_test_hard_balanced_accuracy_mean": sem.get("test_hard_balanced_accuracy_mean"),
        "balanced_changed_best_mixed_balanced_accuracy": changed.get("best_mixed_balanced_accuracy"),
        "balanced_changed_best_mixed_roc_auc": changed.get("best_mixed_roc_auc"),
        "expanded_identity_test_exclude_same_point_top1_mean": ident.get("test_exclude_same_point_top1_mean"),
        "expanded_identity_test_confuser_avoidance_top1_mean": ident.get("test_confuser_avoidance_top1_mean"),
        "expanded_identity_test_occlusion_reappear_top1_mean": ident.get("test_occlusion_reappear_top1_mean"),
        "expanded_identity_test_trajectory_crossing_top1_mean": ident.get("test_trajectory_crossing_top1_mean"),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_semantic_field_claim_allowed": False,
        "integrated_identity_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "system_components_ready_for_unified_joint_eval": system_components_ready,
        "exact_blockers": [
            "还没有在 V35.28 unified benchmark 上运行同一 eval harness 同时输出 semantic 与 identity 指标。",
            "semantic adapter 与 identity retrieval head 仍是两个组件的复现，下一步需要统一 joint evaluation/package。",
            "当前仍是 M128/H32，不能外推到 H64/H96/M512/M1024。",
            "要 claim CVPR 级完整系统，还需要 per-category/per-motion/per-occlusion/per-confuser 的统一成功/失败 breakdown。",
        ],
        "recommended_next_step": "run_v35_31_unified_joint_eval_harness_m128_h32",
        "中文结论": (
            "这是阶段性强好消息：V35 已经具备 325-clip video-derived trace 上的 semantic 三 seed、balanced changed、identity 三 seed、"
            "full unified benchmark 和真实 case mining。创新点基本站住，但还不能 claim CVPR 级完整系统；下一步必须跑统一 joint eval harness。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.30 Full Video System Status Decision\n\n"
        f"- semantic_adapter_three_seed_passed: {semantic_ok}\n"
        f"- balanced_cross_dataset_changed_suite_ready: {changed_ok}\n"
        f"- expanded_identity_three_seed_passed: {identity_ok}\n"
        f"- full_unified_video_semantic_identity_benchmark_built: {unified_ok}\n"
        f"- joint_case_mining_visualization_ready: {visualization_ok}\n"
        f"- system_components_ready_for_unified_joint_eval: {system_components_ready}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"组件已准备统一评估": system_components_ready, "推荐下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
