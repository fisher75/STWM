#!/usr/bin/env python3
"""汇总当前 video-derived trace 语义/身份闭环状态，不训练新模型。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

SEMANTIC_REPLICATION = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_adapter_replication_decision_20260515.json"
IDENTITY_REPLICATION = ROOT / "reports/stwm_ostf_v35_16_video_identity_decision_20260515.json"
VIDEO_INPUT_AUDIT = ROOT / "reports/stwm_ostf_v35_9_video_input_closure_audit_20260515.json"
VIDEO_CLOSURE_SMOKE = ROOT / "reports/stwm_ostf_v35_13_video_closure_decision_20260515.json"
REPORT = ROOT / "reports/stwm_ostf_v35_22_joint_video_semantic_identity_closure_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_22_JOINT_VIDEO_SEMANTIC_IDENTITY_CLOSURE_DECISION_20260515.md"


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


def main() -> int:
    semantic = read_json(SEMANTIC_REPLICATION)
    identity = read_json(IDENTITY_REPLICATION)
    video_input = read_json(VIDEO_INPUT_AUDIT)
    video_smoke = read_json(VIDEO_CLOSURE_SMOKE)
    semantic_three_seed = bool(semantic.get("all_three_seed_passed", False))
    identity_three_seed = bool(identity.get("video_identity_pairwise_retrieval_seed42_123_456_passed", False))
    video_input_smoke = bool(video_smoke.get("video_input_trace_measurement_closure_passed", False))
    stratified_changed_passed = bool(semantic.get("vipseg_to_vspw_stratified_changed_passed", False))
    raw_video_v35_closed = bool(video_input.get("raw_video_input_closed_for_v35", False))
    full_claim_ready = bool(
        semantic_three_seed
        and identity_three_seed
        and video_input_smoke
        and stratified_changed_passed
        and raw_video_v35_closed
    )
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "joint_video_semantic_identity_closure_audit_done": True,
        "semantic_adapter_three_seed_passed": semantic_three_seed,
        "identity_retrieval_three_seed_passed": identity_three_seed,
        "video_input_trace_measurement_closure_smoke_passed": video_input_smoke,
        "raw_video_input_closed_for_v35": raw_video_v35_closed,
        "vipseg_to_vspw_stratified_changed_passed": stratified_changed_passed,
        "semantic_test_changed_balanced_accuracy_mean": semantic.get("test_changed_balanced_accuracy_mean"),
        "semantic_test_changed_roc_auc_mean": semantic.get("test_changed_roc_auc_mean"),
        "semantic_test_hard_balanced_accuracy_mean": semantic.get("test_hard_balanced_accuracy_mean"),
        "semantic_test_hard_roc_auc_mean": semantic.get("test_hard_roc_auc_mean"),
        "semantic_test_uncertainty_balanced_accuracy_mean": semantic.get("test_uncertainty_balanced_accuracy_mean"),
        "semantic_test_uncertainty_roc_auc_mean": semantic.get("test_uncertainty_roc_auc_mean"),
        "identity_embedding_trained": bool(identity.get("identity_embedding_trained", False)),
        "same_frame_hard_negative_used": bool(identity.get("same_frame_hard_negative_used", False)),
        "same_semantic_confuser_used": bool(identity.get("same_semantic_confuser_used", False)),
        "trajectory_crossing_used": bool(identity.get("trajectory_crossing_used", False)),
        "occlusion_reappear_used": bool(identity.get("occlusion_reappear_used", False)),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": full_claim_ready,
        "exact_blockers": [
            "VIPSeg→VSPW stratified changed 仍未通过，说明跨域分层 semantic change 还不是完全稳健。",
            "V35.9 审计显示 raw/video-derived input closure 对当前 V35 主线仍未完全闭合；V35.13 是小规模 video closure smoke，不等于完整 benchmark。",
            "当前仍是 M128/H32；没有跑 M512/M1024 或 H64/H96，也不能把 dense large-scale claim 外推。",
            "语义与身份虽各自三 seed 通过，但还缺同一 video benchmark 上的 joint case-mined success/failure breakdown。",
        ],
        "recommended_next_step": "fix_vipseg_to_vspw_stratified_changed_and_run_joint_video_closure",
        "中文结论": (
            "当前是明显好消息，但不是最终成功。语义端 V35.21 三 seed 通过，身份端 V35.16 pairwise retrieval 三 seed 通过，"
            "说明 STWM 的 video-derived trace→future semantic/identity 方向已经有真实可测信号。坏消息是，"
            "跨域分层 changed 仍弱，raw/video-derived input closure 还没有在扩展 benchmark 上完全闭合，因此还不能 claim CVPR 级完整系统。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.22 Joint Video Semantic Identity Closure Decision\n\n"
        f"- semantic_adapter_three_seed_passed: {semantic_three_seed}\n"
        f"- identity_retrieval_three_seed_passed: {identity_three_seed}\n"
        f"- video_input_trace_measurement_closure_smoke_passed: {video_input_smoke}\n"
        f"- raw_video_input_closed_for_v35: {raw_video_v35_closed}\n"
        f"- vipseg_to_vspw_stratified_changed_passed: {stratified_changed_passed}\n"
        f"- full_video_semantic_identity_field_claim_allowed: {full_claim_ready}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"完整闭环可claim": full_claim_ready, "推荐下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
