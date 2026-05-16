#!/usr/bin/env python3
"""汇总 V35.15 扩展 M128/H32 video semantic benchmark 的最终决策。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPORT = ROOT / "reports/stwm_ostf_v35_15_expanded_video_closure_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_15_EXPANDED_VIDEO_CLOSURE_DECISION_20260515.md"


def load(path: str) -> dict[str, Any]:
    p = ROOT / path
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


def seed_decision(seed: int) -> dict[str, Any]:
    suffix = "" if seed == 42 else f"_seed{seed}"
    return load(f"reports/stwm_ostf_v35_15_expanded_video_semantic_state_adapter_decision_20260515{suffix}.json")


def seed_eval(seed: int) -> dict[str, Any]:
    suffix = "" if seed == 42 else f"_seed{seed}"
    return load(f"reports/stwm_ostf_v35_15_expanded_video_semantic_state_adapter_eval_summary_20260515{suffix}.json")


def main() -> int:
    cot = load("reports/stwm_cotracker_object_dense_teacher_v16_M128_H32_v35_15_expand96_20260502.json")
    meas = load("reports/stwm_ostf_v35_10_video_observed_semantic_measurement_cache_20260515.json")
    target = load("reports/stwm_ostf_v35_15_expanded_mask_derived_video_semantic_state_target_build_20260515.json")
    pred = load("reports/stwm_ostf_v35_15_expanded_mask_video_semantic_state_predictability_decision_20260515.json")
    viz = load("reports/stwm_ostf_v35_15_expanded_video_semantic_state_visualization_manifest_20260515.json")
    identity = load("reports/stwm_ostf_v35_11_video_identity_measurement_base_and_stable_copy_adapter_20260515.json")
    decisions = {str(seed): seed_decision(seed) for seed in [42, 123, 456]}
    evals = {str(seed): seed_eval(seed) for seed in [42, 123, 456]}
    all_seed_passed = all(bool(d.get("video_semantic_state_adapter_passed")) for d in decisions.values())
    all_stable = all(bool(d.get("stable_preservation")) for d in decisions.values())
    no_leak = all(not bool(d.get("future_leakage_detected")) for d in decisions.values()) and not bool(pred.get("future_leakage_detected"))
    traj_ok = all(not bool(d.get("trajectory_degraded")) for d in decisions.values())
    sample_count = int(target.get("sample_count") or cot.get("processed_clip_count") or 0)
    expanded_passed = bool(
        sample_count >= 96
        and pred.get("observed_predictable_video_semantic_state_suite_ready") is True
        and all_seed_passed
        and all_stable
        and no_leak
        and traj_ok
        and viz.get("visualization_ready") is True
    )
    next_step = (
        "fix_video_identity_field_as_pairwise_retrieval_on_mask_tracks"
        if expanded_passed
        else "expand_or_repair_mask_video_semantic_benchmark_m128_h32"
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expanded_v35_14_mask_video_benchmark_done": True,
        "target_total_requested": cot.get("requested_split_counts"),
        "processed_clip_count": cot.get("processed_clip_count"),
        "measurement_sample_count": meas.get("sample_count"),
        "mask_target_sample_count": sample_count,
        "video_semantic_target_source": target.get("video_semantic_target_source"),
        "semantic_changed_is_real_video_state": target.get("semantic_changed_is_real_video_state"),
        "identity_confuser_target_built": target.get("identity_confuser_target_built"),
        "current_video_cache_insufficient_for_semantic_change_benchmark": target.get("current_video_cache_insufficient_for_semantic_change_benchmark"),
        "observed_predictable_video_semantic_state_suite_ready": pred.get("observed_predictable_video_semantic_state_suite_ready"),
        "semantic_cluster_transition_passed": pred.get("semantic_cluster_transition_passed"),
        "semantic_changed_passed": pred.get("semantic_changed_passed"),
        "semantic_hard_passed": pred.get("semantic_hard_passed"),
        "evidence_anchor_family_passed": pred.get("evidence_anchor_family_passed"),
        "uncertainty_target_passed": pred.get("uncertainty_target_passed"),
        "video_semantic_state_adapter_seed42_123_456_passed": all_seed_passed,
        "seed_decisions": decisions,
        "seed_eval_summaries": evals,
        "identity_measurement_base_passed": identity.get("measurement_identity_retrieval_passed"),
        "stable_copy_adapter_passed": identity.get("stable_copy_adapter_passed"),
        "v30_backbone_frozen": True,
        "future_leakage_detected": not no_leak,
        "trajectory_degraded": not traj_ok,
        "visualization_ready": viz.get("visualization_ready") is True,
        "video_m128_h32_expanded_benchmark_passed": expanded_passed,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": next_step,
        "中文结论": (
            "V35.15 扩展 M128/H32 video semantic benchmark 已通过三 seed smoke；下一步应修 video identity field 的 pairwise/retrieval 闭环。"
            if expanded_passed
            else "V35.15 扩展 benchmark 尚未完全通过，不能进入 video identity field claim。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.15 Expanded Video Closure Decision\n\n"
        f"- expanded_v35_14_mask_video_benchmark_done: true\n"
        f"- mask_target_sample_count: {sample_count}\n"
        f"- observed_predictable_video_semantic_state_suite_ready: {report['observed_predictable_video_semantic_state_suite_ready']}\n"
        f"- video_semantic_state_adapter_seed42_123_456_passed: {all_seed_passed}\n"
        f"- visualization_ready: {report['visualization_ready']}\n"
        f"- video_m128_h32_expanded_benchmark_passed: {expanded_passed}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {next_step}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"video_m128_h32_expanded_benchmark_passed": expanded_passed, "recommended_next_step": next_step}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
