#!/usr/bin/env python3
"""汇总 V35.16 video identity pairwise retrieval 决策。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPORT = ROOT / "reports/stwm_ostf_v35_16_video_identity_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_16_VIDEO_IDENTITY_DECISION_20260515.md"


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
    return load(f"reports/stwm_ostf_v35_16_video_identity_pairwise_retrieval_decision_20260515{suffix}.json")


def seed_eval(seed: int) -> dict[str, Any]:
    suffix = "" if seed == 42 else f"_seed{seed}"
    return load(f"reports/stwm_ostf_v35_16_video_identity_pairwise_retrieval_eval_summary_20260515{suffix}.json")


def main() -> int:
    target = load("reports/stwm_ostf_v35_16_video_identity_pairwise_target_build_20260515.json")
    sem = load("reports/stwm_ostf_v35_15_expanded_video_closure_decision_20260515.json")
    viz = load("reports/stwm_ostf_v35_16_video_identity_retrieval_visualization_manifest_20260515.json")
    decisions = {str(s): seed_decision(s) for s in [42, 123, 456]}
    evals = {str(s): seed_eval(s) for s in [42, 123, 456]}
    all_passed = all(bool(d.get("video_identity_pairwise_retrieval_passed")) for d in decisions.values())
    no_leak = all(not bool(d.get("future_leakage_detected")) for d in decisions.values())
    traj_ok = all(not bool(d.get("trajectory_degraded")) for d in decisions.values())
    semantic_smoke = bool(sem.get("video_m128_h32_expanded_benchmark_passed"))
    identity_video_smoke = bool(target.get("video_identity_pairwise_targets_built") and all_passed and no_leak and traj_ok and viz.get("visualization_ready"))
    next_step = "expand_video_benchmark_cross_dataset_and_video_input_closure" if identity_video_smoke and semantic_smoke else "fix_video_identity_pairwise_retrieval_head"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "video_identity_pairwise_targets_built": target.get("video_identity_pairwise_targets_built"),
        "video_identity_pairwise_retrieval_seed42_123_456_passed": all_passed,
        "identity_embedding_trained": True,
        "same_frame_hard_negative_used": True,
        "same_semantic_confuser_used": True,
        "trajectory_crossing_used": True,
        "occlusion_reappear_used": True,
        "measurement_preserving_residual_head": True,
        "seed_decisions": decisions,
        "seed_eval_summaries": evals,
        "v35_15_video_semantic_expanded_benchmark_passed": semantic_smoke,
        "video_identity_field_m128_h32_smoke_allowed": identity_video_smoke,
        "video_semantic_field_m128_h32_smoke_allowed": semantic_smoke,
        "v30_backbone_frozen": True,
        "future_leakage_detected": not no_leak,
        "trajectory_degraded": not traj_ok,
        "visualization_ready": viz.get("visualization_ready") is True,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": next_step,
        "中文结论": (
            "V35.16 已在 96-clip M128/H32 video benchmark 上通过三 seed pairwise identity retrieval smoke；但完整 CVPR 级 claim 仍需要更大 cross-dataset/video-input closure。"
            if identity_video_smoke and semantic_smoke
            else "V35.16 video identity retrieval 尚未形成稳定 smoke，不能进入完整系统 claim。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.16 Video Identity Decision\n\n"
        f"- video_identity_pairwise_targets_built: {report['video_identity_pairwise_targets_built']}\n"
        f"- video_identity_pairwise_retrieval_seed42_123_456_passed: {all_passed}\n"
        f"- v35_15_video_semantic_expanded_benchmark_passed: {semantic_smoke}\n"
        f"- video_identity_field_m128_h32_smoke_allowed: {identity_video_smoke}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {next_step}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"video_identity_pairwise_retrieval_seed42_123_456_passed": all_passed, "recommended_next_step": next_step}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
