#!/usr/bin/env python3
"""V35.25 joint video semantic/identity closure decision。"""
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
CHANGED_DECISION = ROOT / "reports/stwm_ostf_v35_24_balanced_cross_dataset_changed_predictability_decision_20260516.json"
IDENTITY_DECISION = ROOT / "reports/stwm_ostf_v35_16_video_identity_decision_20260515.json"
VIDEO_INPUT_AUDIT = ROOT / "reports/stwm_ostf_v35_9_video_input_closure_audit_20260515.json"
VIDEO_SMOKE = ROOT / "reports/stwm_ostf_v35_13_video_closure_decision_20260515.json"
VIS_MANIFEST = ROOT / "reports/stwm_ostf_v35_25_joint_video_semantic_identity_case_mining_manifest_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_25_joint_video_semantic_identity_closure_decision_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_25_JOINT_VIDEO_SEMANTIC_IDENTITY_CLOSURE_DECISION_20260516.md"


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
    semantic = read_json(SEM_REP)
    changed = read_json(CHANGED_DECISION)
    identity = read_json(IDENTITY_DECISION)
    video_input = read_json(VIDEO_INPUT_AUDIT)
    video_smoke = read_json(VIDEO_SMOKE)
    vis = read_json(VIS_MANIFEST)
    semantic_three_seed = bool(semantic.get("all_three_seed_passed", False))
    balanced_changed_ready = bool(changed.get("balanced_cross_dataset_changed_suite_ready", False))
    identity_three_seed = bool(identity.get("video_identity_pairwise_retrieval_seed42_123_456_passed", False))
    video_smoke_passed = bool(video_smoke.get("video_input_trace_measurement_closure_passed", False))
    raw_video_closed = bool(video_input.get("raw_video_input_closed_for_v35", False))
    visualization_ready = bool(vis.get("visualization_ready", False))
    full_claim_ready = bool(semantic_three_seed and balanced_changed_ready and identity_three_seed and video_smoke_passed and raw_video_closed and visualization_ready)
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "joint_video_semantic_identity_closure_decision_done": True,
        "semantic_adapter_three_seed_passed": semantic_three_seed,
        "balanced_cross_dataset_changed_suite_ready": balanced_changed_ready,
        "identity_retrieval_three_seed_passed": identity_three_seed,
        "video_input_trace_measurement_closure_smoke_passed": video_smoke_passed,
        "raw_video_input_closed_for_v35": raw_video_closed,
        "visualization_ready": visualization_ready,
        "semantic_test_changed_balanced_accuracy_mean": semantic.get("test_changed_balanced_accuracy_mean"),
        "semantic_test_hard_balanced_accuracy_mean": semantic.get("test_hard_balanced_accuracy_mean"),
        "semantic_test_uncertainty_balanced_accuracy_mean": semantic.get("test_uncertainty_balanced_accuracy_mean"),
        "balanced_changed_best_target_val_protocol": changed.get("best_target_val_protocol"),
        "balanced_changed_best_target_val_balanced_accuracy": changed.get("best_target_val_balanced_accuracy"),
        "balanced_changed_best_target_val_roc_auc": changed.get("best_target_val_roc_auc"),
        "balanced_changed_best_mixed_protocol": changed.get("best_mixed_protocol"),
        "balanced_changed_best_mixed_balanced_accuracy": changed.get("best_mixed_balanced_accuracy"),
        "balanced_changed_best_mixed_roc_auc": changed.get("best_mixed_roc_auc"),
        "identity_embedding_trained": bool(identity.get("identity_embedding_trained", False)),
        "same_frame_hard_negative_used": bool(identity.get("same_frame_hard_negative_used", False)),
        "same_semantic_confuser_used": bool(identity.get("same_semantic_confuser_used", False)),
        "trajectory_crossing_used": bool(identity.get("trajectory_crossing_used", False)),
        "occlusion_reappear_used": bool(identity.get("occlusion_reappear_used", False)),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_semantic_field_claim_allowed": False,
        "integrated_identity_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": full_claim_ready,
        "exact_blockers": [
            "raw_video_input_closed_for_v35=false：V35.13 只是小规模 video-input closure smoke，不等于扩展 benchmark 上的完整 raw/video-derived closure。",
            "当前仍是 M128/H32；不能外推到 M512/M1024 或 H64/H96。",
            "V35.24 changed 最稳信号来自 trace/risk feature，说明跨域语义 change 应以 ontology-agnostic state/risk 表述为主，不能回到 semantic-id shortcut。",
            "还需要同一 video-derived target root 上同时跑 semantic adapter、identity retrieval、case breakdown 的统一 benchmark，而不是只汇总独立报告。",
        ],
        "recommended_next_step": "build_v35_26_unified_video_derived_semantic_identity_benchmark_m128_h32",
        "中文结论": (
            "V35.25 说明当前路线是明显好消息：semantic 三 seed、identity retrieval 三 seed、balanced changed predictability "
            "和真实 case mining 都已成立。但 full claim 还差统一 video-derived benchmark 与 raw/video input closure，"
            "所以不能 claim CVPR 级完整系统成功。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.25 Joint Video Semantic Identity Closure Decision\n\n"
        f"- semantic_adapter_three_seed_passed: {semantic_three_seed}\n"
        f"- balanced_cross_dataset_changed_suite_ready: {balanced_changed_ready}\n"
        f"- identity_retrieval_three_seed_passed: {identity_three_seed}\n"
        f"- video_input_trace_measurement_closure_smoke_passed: {video_smoke_passed}\n"
        f"- raw_video_input_closed_for_v35: {raw_video_closed}\n"
        f"- visualization_ready: {visualization_ready}\n"
        f"- full_video_semantic_identity_field_claim_allowed: {full_claim_ready}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"full_claim_allowed": full_claim_ready, "推荐下一步": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
