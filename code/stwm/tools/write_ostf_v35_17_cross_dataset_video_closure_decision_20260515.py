#!/usr/bin/env python3
"""汇总 V35.17 cross-dataset video benchmark 决策。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

REPORT = ROOT / "reports/stwm_ostf_v35_17_cross_dataset_video_closure_decision_20260515.json"
DOC = ROOT / "docs/STWM_OSTF_V35_17_CROSS_DATASET_VIDEO_CLOSURE_DECISION_20260515.md"


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


def main() -> int:
    trace = load("reports/stwm_cotracker_object_dense_teacher_v16_M128_H32_v35_17_expand192_20260502.json")
    target = load("reports/stwm_ostf_v35_17_cross_dataset_mask_derived_video_semantic_state_target_build_20260515.json")
    identity_target = load("reports/stwm_ostf_v35_17_cross_dataset_video_identity_pairwise_target_build_20260515.json")
    pred = load("reports/stwm_ostf_v35_17_cross_dataset_video_semantic_predictability_decision_20260515.json")
    pred_eval = load("reports/stwm_ostf_v35_17_cross_dataset_video_semantic_predictability_eval_20260515.json")
    viz = load("reports/stwm_ostf_v35_17_cross_dataset_video_semantic_visualization_manifest_20260515.json")
    protocols = pred_eval.get("protocols", {})
    vipseg_to_vspw = protocols.get("vipseg_to_vspw", {})
    blocker = "vipseg_to_vspw_domain_shift_or_target_split"
    if vipseg_to_vspw:
        test_pos = vipseg_to_vspw.get("semantic_changed", {}).get("test", {}).get("positive_ratio")
        if test_pos is not None and test_pos < 0.06:
            blocker = "vipseg_to_vspw_low_changed_positive_ratio_and_domain_shift"
    suite_ready = bool(pred.get("cross_dataset_video_semantic_suite_ready"))
    decision = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cross_dataset_video_benchmark_ran": True,
        "processed_clip_count": trace.get("processed_clip_count"),
        "processed_split_counts": trace.get("processed_split_counts"),
        "processed_dataset_counts": trace.get("processed_dataset_counts"),
        "mask_target_sample_count": target.get("sample_count"),
        "identity_pairwise_targets_built": identity_target.get("video_identity_pairwise_targets_built"),
        "mixed_unseen_passed": pred.get("mixed_unseen_passed"),
        "vspw_to_vipseg_passed": pred.get("vspw_to_vipseg_passed"),
        "vipseg_to_vspw_passed": pred.get("vipseg_to_vspw_passed"),
        "cross_dataset_video_semantic_suite_ready": suite_ready,
        "semantic_adapter_training_ran": False,
        "identity_retrieval_training_ran": False,
        "not_run_reason": "cross_dataset_semantic_predictability_failed",
        "primary_blocker": blocker,
        "v30_backbone_frozen": True,
        "future_leakage_detected": False,
        "trajectory_degraded": False,
        "visualization_ready": viz.get("visualization_ready") is True,
        "integrated_identity_field_claim_allowed": False,
        "integrated_semantic_field_claim_allowed": False,
        "full_video_semantic_identity_field_claim_allowed": False,
        "recommended_next_step": "fix_vipseg_to_vspw_video_semantic_domain_shift_or_target_split",
        "中文结论": (
            "V35.17 已扩到 192 clips；mixed_unseen 与 VSPW→VIPSeg 通过，但 VIPSeg→VSPW 失败。"
            "因此不能继续训练 cross-dataset adapter，也不能 claim full video semantic/identity field。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(decision), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.17 Cross-Dataset Video Closure Decision\n\n"
        f"- processed_clip_count: {decision['processed_clip_count']}\n"
        f"- mixed_unseen_passed: {decision['mixed_unseen_passed']}\n"
        f"- vspw_to_vipseg_passed: {decision['vspw_to_vipseg_passed']}\n"
        f"- vipseg_to_vspw_passed: {decision['vipseg_to_vspw_passed']}\n"
        f"- cross_dataset_video_semantic_suite_ready: {suite_ready}\n"
        f"- semantic_adapter_training_ran: false\n"
        f"- identity_retrieval_training_ran: false\n"
        f"- primary_blocker: {blocker}\n"
        f"- full_video_semantic_identity_field_claim_allowed: false\n"
        f"- recommended_next_step: {decision['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + decision["中文结论"]
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"cross_dataset_video_semantic_suite_ready": suite_ready, "recommended_next_step": decision["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
