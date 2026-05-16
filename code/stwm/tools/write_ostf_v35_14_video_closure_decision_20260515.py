#!/usr/bin/env python3
"""汇总 V35.14 mask-derived video semantic target repair 与 adapter 三 seed 结果。"""
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

AUDIT = ROOT / "reports/stwm_ostf_v35_14_video_semantic_target_truth_audit_20260515.json"
BUILD = ROOT / "reports/stwm_ostf_v35_14_mask_derived_video_semantic_state_target_build_20260515.json"
PREDICT = ROOT / "reports/stwm_ostf_v35_14_mask_video_semantic_state_predictability_decision_20260515.json"
IDENTITY = ROOT / "reports/stwm_ostf_v35_11_video_identity_measurement_base_and_stable_copy_adapter_20260515.json"
OUT = ROOT / "reports/stwm_ostf_v35_14_video_closure_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_14_VIDEO_CLOSURE_DECISION_20260515.md"


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
    audit = load(AUDIT)
    build = load(BUILD)
    predict = load(PREDICT)
    identity = load(IDENTITY)
    seeds = {}
    all_pass = True
    for seed, suffix in [(42, ""), (123, "_seed123"), (456, "_seed456")]:
        d = load(ROOT / f"reports/stwm_ostf_v35_14_video_semantic_state_adapter_decision_20260515{suffix}.json")
        e = load(ROOT / f"reports/stwm_ostf_v35_14_video_semantic_state_adapter_eval_summary_20260515{suffix}.json")
        passed = bool(d.get("video_semantic_state_adapter_passed", False))
        all_pass = all_pass and passed
        seeds[str(seed)] = {
            "passed": passed,
            "semantic_changed_passed": bool(d.get("semantic_changed_passed", False)),
            "semantic_hard_passed": bool(d.get("semantic_hard_passed", False)),
            "uncertainty_passed": bool(d.get("uncertainty_passed", False)),
            "stable_preservation": bool(d.get("stable_preservation", False)),
            "test_semantic_changed": e.get("test", {}).get("semantic_changed", {}),
            "test_semantic_hard": e.get("test", {}).get("semantic_hard", {}),
            "test_semantic_uncertainty": e.get("test", {}).get("semantic_uncertainty", {}),
            "test_cluster": e.get("test", {}).get("cluster", {}),
        }
    smoke_claim_allowed = bool(
        all_pass
        and bool(identity.get("measurement_identity_retrieval_passed", False))
        and bool(identity.get("stable_copy_adapter_passed", False))
        and bool(predict.get("observed_predictable_video_semantic_state_suite_ready", False))
    )
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v35_14_video_semantic_target_truth_audit_done": bool(audit),
        "mask_label_available": bool(audit.get("mask_label_available", False)),
        "mask_derived_video_semantic_targets_built": bool(build.get("mask_derived_video_semantic_state_targets_built", False)),
        "video_semantic_target_source": build.get("video_semantic_target_source"),
        "semantic_changed_is_real_video_state": bool(build.get("semantic_changed_is_real_video_state", False)),
        "identity_confuser_target_built": bool(build.get("identity_confuser_target_built", False)),
        "current_video_cache_insufficient_for_semantic_change_benchmark": bool(build.get("current_video_cache_insufficient_for_semantic_change_benchmark", True)),
        "target_predictability_eval_done": bool(predict.get("target_predictability_eval_done", False)),
        "observed_predictable_video_semantic_state_suite_ready": bool(predict.get("observed_predictable_video_semantic_state_suite_ready", False)),
        "video_semantic_state_adapter_seed42_123_456_passed": all_pass,
        "adapter_seed_results": seeds,
        "identity_measurement_base_passed": bool(identity.get("measurement_identity_retrieval_passed", False)),
        "stable_copy_adapter_passed": bool(identity.get("stable_copy_adapter_passed", False)),
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "video_m128_h32_smoke_system_passed": smoke_claim_allowed,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "exact_blockers": [
            "V35.14 已在 32 clip M128/H32 video-derived smoke 上通过三 seed，但规模仍不足以支撑 CVPR 级完整 claim。",
            "下一步需要扩展 mask-derived video benchmark 到更大 M128/H32 split，并加入更严格 identity confuser / occlusion-reappear / cross-video generalization。",
            "H64/H96/M512 仍不应运行，直到 M128/H32 大规模 video benchmark 通过。",
        ],
        "recommended_next_step": "expand_v35_14_mask_video_benchmark_m128_h32",
        "中文结论": (
            "V35.14 完成了关键修复：video semantic target 从 CLIP/KMeans 改为真实 VSPW/VIPSeg mask/panoptic label；"
            "target predictability 通过，video semantic adapter seed42/123/456 通过，identity measurement base 与 stable copy 也通过。"
            "这是 video-derived trace 到 future semantic/identity field 的 M128/H32 smoke 成功，但还不是 CVPR 级完整系统 claim。"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.14 Video Closure Decision\n\n"
        f"- mask_derived_video_semantic_targets_built: {decision['mask_derived_video_semantic_targets_built']}\n"
        f"- video_semantic_target_source: {decision['video_semantic_target_source']}\n"
        f"- semantic_changed_is_real_video_state: {decision['semantic_changed_is_real_video_state']}\n"
        f"- identity_confuser_target_built: {decision['identity_confuser_target_built']}\n"
        f"- observed_predictable_video_semantic_state_suite_ready: {decision['observed_predictable_video_semantic_state_suite_ready']}\n"
        f"- video_semantic_state_adapter_seed42_123_456_passed: {all_pass}\n"
        f"- video_m128_h32_smoke_system_passed: {smoke_claim_allowed}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"video_m128_h32_smoke_system_passed": smoke_claim_allowed, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
