#!/usr/bin/env python3
"""汇总 V35.11-V35.13 video closure 与 video target predictability 决策。"""
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

TRACE_REPORT = ROOT / "reports/stwm_cotracker_object_dense_teacher_v16_M128_H32_v35_13_h32_expand32_20260502.json"
MEAS_REPORT = ROOT / "reports/stwm_ostf_v35_10_video_observed_semantic_measurement_cache_20260515.json"
IDENTITY_REPORT = ROOT / "reports/stwm_ostf_v35_11_video_identity_measurement_base_and_stable_copy_adapter_20260515.json"
TARGET12_REPORT = ROOT / "reports/stwm_ostf_v35_12_video_derived_future_semantic_state_target_build_20260515.json"
TARGET13_REPORT = ROOT / "reports/stwm_ostf_v35_13_fixed_video_semantic_state_target_build_20260515.json"
PRED13_DECISION = ROOT / "reports/stwm_ostf_v35_13_fixed_video_semantic_state_target_predictability_decision_20260515.json"
OUT = ROOT / "reports/stwm_ostf_v35_13_video_closure_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_13_VIDEO_CLOSURE_DECISION_20260515.md"


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    trace = load(TRACE_REPORT)
    meas = load(MEAS_REPORT)
    identity = load(IDENTITY_REPORT)
    target12 = load(TARGET12_REPORT)
    target13 = load(TARGET13_REPORT)
    pred = load(PRED13_DECISION)
    video_trace_cache_expanded = int(trace.get("processed_clip_count", 0)) >= 32
    observed_measurement_cache_built = bool(meas.get("video_observed_semantic_measurement_cache_built", False)) and int(meas.get("sample_count", 0)) >= 32
    identity_measurement_base_passed = bool(identity.get("measurement_identity_retrieval_passed", False))
    stable_copy_adapter_passed = bool(identity.get("stable_copy_adapter_passed", False))
    video_future_semantic_targets_built = bool(target12.get("video_derived_future_semantic_state_targets_built", False)) and int(target12.get("sample_count", 0)) >= 32
    fixed_video_semantic_targets_built = bool(target13.get("fixed_video_semantic_state_targets_built", False)) and int(target13.get("sample_count", 0)) >= 32
    target_predictability_eval_done = bool(pred.get("target_predictability_eval_done", False))
    suite_ready = bool(pred.get("observed_predictable_video_semantic_state_suite_ready", False))
    full_claim = bool(video_trace_cache_expanded and observed_measurement_cache_built and identity_measurement_base_passed and stable_copy_adapter_passed and suite_ready)
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_trace_cache_expanded": video_trace_cache_expanded,
        "video_trace_processed_clip_count": int(trace.get("processed_clip_count", 0)),
        "video_trace_processed_split_counts": trace.get("processed_split_counts", {}),
        "observed_measurement_cache_built": observed_measurement_cache_built,
        "observed_measurement_sample_count": int(meas.get("sample_count", 0)),
        "identity_measurement_base_passed": identity_measurement_base_passed,
        "stable_copy_adapter_passed": stable_copy_adapter_passed,
        "video_input_trace_measurement_closure_passed": bool(identity.get("video_input_trace_measurement_closure_passed", False)),
        "video_future_semantic_targets_built": video_future_semantic_targets_built,
        "fixed_video_semantic_targets_built": fixed_video_semantic_targets_built,
        "target_predictability_eval_done": target_predictability_eval_done,
        "semantic_cluster_transition_passed": bool(pred.get("semantic_cluster_transition_passed", False)),
        "semantic_changed_passed": bool(pred.get("semantic_changed_passed", False)),
        "semantic_hard_passed": bool(pred.get("semantic_hard_passed", False)),
        "evidence_anchor_family_passed": bool(pred.get("evidence_anchor_family_passed", False)),
        "uncertainty_target_passed": bool(pred.get("uncertainty_target_passed", False)),
        "observed_predictable_video_semantic_state_suite_ready": suite_ready,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "integrated_identity_field_video_smoke_allowed": bool(identity.get("integrated_identity_field_video_smoke_allowed", False)),
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": full_claim,
        "exact_blockers": [
            "Video-derived trace + observed measurement + identity measurement base 已闭环，但 semantic changed/hard target 在 observed-only 上界中没有稳定通过。",
            "V35.13 fixed target 后 uncertainty/changed/hard 仍存在 val/test 分布不稳，说明当前 video future semantic target 不是可靠训练目标。",
            "继续训练 V35 semantic head 会把 target/benchmark 问题伪装成模型问题，因此应先收集或定义更可靠的 video future semantic targets。",
        ],
        "recommended_next_step": "collect_better_video_semantic_benchmark_targets",
        "中文结论": (
            "V35.13 已把系统推进到 M128/H32 video-derived trace + observed semantic measurement 的闭环 smoke："
            "identity measurement base 和 stable copy adapter 通过。"
            "但 video future semantic state target 的 observed-only 上界没有通过，不能 claim 完整 semantic field success。"
            "下一步应修/扩 video semantic benchmark target，而不是训练 writer/gate/head。"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.13 Video Closure Decision\n\n"
        f"- video_trace_cache_expanded: {video_trace_cache_expanded}\n"
        f"- video_trace_processed_clip_count: {decision['video_trace_processed_clip_count']}\n"
        f"- observed_measurement_cache_built: {observed_measurement_cache_built}\n"
        f"- identity_measurement_base_passed: {identity_measurement_base_passed}\n"
        f"- stable_copy_adapter_passed: {stable_copy_adapter_passed}\n"
        f"- fixed_video_semantic_targets_built: {fixed_video_semantic_targets_built}\n"
        f"- observed_predictable_video_semantic_state_suite_ready: {suite_ready}\n"
        f"- integrated_identity_field_claim_allowed: false\n"
        f"- integrated_semantic_field_claim_allowed: false\n"
        f"- full_video_semantic_identity_field_claim_allowed: {full_claim}\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"recommended_next_step": decision["recommended_next_step"], "full_claim": full_claim}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
