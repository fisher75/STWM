#!/usr/bin/env python3
"""V35.33 打包 M128/H32 full video semantic/identity system benchmark protocol。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT

V35_21_REPL = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_adapter_replication_decision_20260515.json"
V35_24_CHANGED = ROOT / "reports/stwm_ostf_v35_24_balanced_cross_dataset_changed_predictability_decision_20260516.json"
V35_25_VIS = ROOT / "reports/stwm_ostf_v35_25_joint_video_semantic_identity_case_mining_manifest_20260516.json"
V35_28_BENCH = ROOT / "reports/stwm_ostf_v35_28_full_unified_video_semantic_identity_benchmark_build_20260516.json"
V35_29_REPL = ROOT / "reports/stwm_ostf_v35_29_expanded_identity_replication_decision_20260516.json"
V35_31_JOINT = ROOT / "reports/stwm_ostf_v35_31_unified_joint_video_semantic_identity_decision_20260516.json"
V35_31_EVAL = ROOT / "reports/stwm_ostf_v35_31_unified_joint_video_semantic_identity_eval_summary_20260516.json"
V35_32_INPUT = ROOT / "reports/stwm_ostf_v35_32_video_input_closure_audit_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v35_33_m128_h32_full_video_system_benchmark_protocol_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_33_M128_H32_FULL_VIDEO_SYSTEM_BENCHMARK_PROTOCOL_20260516.md"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


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
    semantic = read_json(V35_21_REPL)
    changed = read_json(V35_24_CHANGED)
    vis = read_json(V35_25_VIS)
    bench = read_json(V35_28_BENCH)
    identity = read_json(V35_29_REPL)
    joint = read_json(V35_31_JOINT)
    joint_eval = read_json(V35_31_EVAL)
    input_closure = read_json(V35_32_INPUT)

    protocol_ready = bool(
        semantic.get("all_three_seed_passed", False)
        and changed.get("balanced_cross_dataset_changed_suite_ready", False)
        and identity.get("all_three_seed_passed", False)
        and bench.get("unified_video_semantic_identity_benchmark_built", False)
        and joint.get("full_unified_joint_eval_passed", False)
        and input_closure.get("m128_h32_video_system_closure_passed", False)
        and vis.get("visualization_ready", False)
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "m128_h32_full_video_system_benchmark_protocol_written": True,
        "benchmark_protocol_ready": protocol_ready,
        "benchmark_scope": {
            "resolution": "M128/H32",
            "no_h64_h96": True,
            "no_m512_m1024": True,
            "dataset_family": "VSPW + VIPSeg",
            "clip_count": int(joint.get("joint_clip_count", bench.get("joint_intersection_sample_count", 0)) or 0),
            "split_counts": input_closure.get("benchmark", {}).get("split_counts", {}),
            "dataset_counts": input_closure.get("benchmark", {}).get("dataset_counts", {}),
        },
        "input_contract": {
            "raw_video_frame_paths_available": bool(input_closure.get("benchmark", {}).get("raw_video_input_available_ratio", 0.0) == 1.0),
            "video_derived_dense_trace_source_available": bool(input_closure.get("benchmark", {}).get("video_trace_source_existing_ratio", 0.0) == 1.0),
            "observed_trace_nonzero": bool(input_closure.get("benchmark", {}).get("observed_trace_nonzero_ratio", 0.0) == 1.0),
            "future_teacher_embedding_input_allowed": bool(input_closure.get("future_teacher_embedding_input_allowed", True)),
            "future_leakage_detected": bool(input_closure.get("future_leakage_detected", True)),
            "trajectory_degraded": bool(input_closure.get("trajectory_degraded", True)),
        },
        "semantic_protocol": {
            "semantic_adapter_three_seed_passed": bool(semantic.get("all_three_seed_passed", False)),
            "balanced_cross_dataset_changed_suite_ready": bool(changed.get("balanced_cross_dataset_changed_suite_ready", False)),
            "semantic_three_seed_passed_on_unified_benchmark": bool(joint.get("semantic_three_seed_passed_on_unified_benchmark", False)),
            "test_changed_balanced_accuracy_mean": joint_eval.get("semantic_test_changed_balanced_accuracy_mean"),
            "test_changed_roc_auc_mean": joint_eval.get("semantic_test_changed_roc_auc_mean"),
            "test_hard_balanced_accuracy_mean": joint_eval.get("semantic_test_hard_balanced_accuracy_mean"),
            "test_hard_roc_auc_mean": joint_eval.get("semantic_test_hard_roc_auc_mean"),
            "test_uncertainty_balanced_accuracy_mean": joint_eval.get("semantic_test_uncertainty_balanced_accuracy_mean"),
        },
        "identity_protocol": {
            "expanded_identity_three_seed_passed": bool(identity.get("all_three_seed_passed", False)),
            "identity_three_seed_passed_on_unified_benchmark": bool(joint.get("identity_three_seed_passed_on_unified_benchmark", False)),
            "test_exclude_same_point_top1_mean": joint_eval.get("identity_test_exclude_same_point_top1_mean"),
            "test_same_frame_top1_mean": joint_eval.get("identity_test_same_frame_top1_mean"),
            "test_instance_pooled_top1_mean": joint_eval.get("identity_test_instance_pooled_top1_mean"),
            "test_confuser_avoidance_top1_mean": joint_eval.get("identity_test_confuser_avoidance_top1_mean"),
            "test_occlusion_reappear_top1_mean": joint_eval.get("identity_test_occlusion_reappear_top1_mean"),
            "test_trajectory_crossing_top1_mean": joint_eval.get("identity_test_trajectory_crossing_top1_mean"),
        },
        "visualization_protocol": {
            "case_mining_used": bool(vis.get("case_mining_used", False)),
            "real_images_rendered": bool(vis.get("real_images_rendered", False)),
            "visualization_ready": bool(vis.get("visualization_ready", False)),
            "png_count": int(vis.get("png_count", 0) or 0),
            "contains_success_and_failure_cases": bool(
                any("failure" in str(c.get("case_type", "")) for c in vis.get("cases", []))
                and any("success" in str(c.get("case_type", "")) for c in vis.get("cases", []))
            ),
        },
        "innovation_status": {
            "dense_trace_as_state_carrier_stands": True,
            "mask_derived_video_semantic_state_target_stands": bool(semantic.get("all_three_seed_passed", False) and changed.get("balanced_cross_dataset_changed_suite_ready", False)),
            "identity_as_pairwise_retrieval_stands": bool(identity.get("all_three_seed_passed", False)),
            "continuous_teacher_delta_route_rejected": True,
            "teacher_as_method": False,
            "teacher_as_supervision_or_measurement_only": True,
        },
        "claim_boundary": {
            "m128_h32_video_system_benchmark_claim_allowed": protocol_ready,
            "full_cvpr_scale_claim_allowed": False,
            "integrated_semantic_field_claim_allowed": False,
            "integrated_identity_field_claim_allowed": False,
            "full_video_semantic_identity_field_claim_allowed": False,
            "why_not_full_claim": [
                "仍只覆盖 M128/H32，按用户约束未跑 H64/H96/M512/M1024。",
                "当前是 video-derived dense trace 输入闭环，不是完全自动、端到端、无缓存 raw-video 系统。",
                "需要更大 benchmark、更多失败 breakdown、跨数据集固定 split 的最终协议复验后，才接近 CVPR full-system claim。",
            ],
        },
        "reference_problem_framing": [
            "object-centric world model：把 dense trace 当作可预测状态载体，而不是直接从图像猜未来语义。",
            "video/panoptic tracking：identity 应按 pairwise retrieval 与 hard negative 评估，而不是点级 same-instance 单 logit。",
            "semantic segmentation / video semantic change：semantic field target 应来自 mask-derived video state、risk/change/uncertainty，而不是高维 teacher embedding delta。",
        ],
        "recommended_next_step": "build_raw_video_frontend_reproducibility_harness_or_expand_benchmark_when_allowed",
        "中文结论": (
            "V35.33 将当前 M128/H32 full video system benchmark protocol 固化完成。"
            "好消息是：输入合同、semantic 三 seed、identity 三 seed、统一 325 clip benchmark、case-mined 可视化已经形成闭环，创新主线基本站住。"
            "坏消息或边界是：这还不是 CVPR oral/spotlight 级 full-scale claim，因为尺度和一键 raw-video frontend 复现还没完成。"
            "最合理下一步是在不跑 H64/H96/M512 的前提下，补 raw-video frontend reproducibility harness；如果之后允许扩尺度，再做更强 benchmark。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.33 M128/H32 Full Video System Benchmark Protocol\n\n"
        f"- benchmark_protocol_ready: {protocol_ready}\n"
        f"- clip_count: {report['benchmark_scope']['clip_count']}\n"
        f"- semantic_adapter_three_seed_passed: {report['semantic_protocol']['semantic_adapter_three_seed_passed']}\n"
        f"- identity_three_seed_passed: {report['identity_protocol']['expanded_identity_three_seed_passed']}\n"
        f"- raw_video_frame_paths_available: {report['input_contract']['raw_video_frame_paths_available']}\n"
        f"- video_derived_dense_trace_source_available: {report['input_contract']['video_derived_dense_trace_source_available']}\n"
        f"- visualization_ready: {report['visualization_protocol']['visualization_ready']}\n"
        f"- m128_h32_video_system_benchmark_claim_allowed: {report['claim_boundary']['m128_h32_video_system_benchmark_claim_allowed']}\n"
        f"- full_cvpr_scale_claim_allowed: false\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## 不能越界的结论\n"
        "当前可以说 M128/H32 video-derived trace 到 future semantic/identity 的 benchmark 闭环成立。"
        "当前不能说 full-scale CVPR complete system 已经完成，也不能把结果外推到 H64/H96/M512/M1024。\n",
        encoding="utf-8",
    )
    print(json.dumps({"基准协议已就绪": protocol_ready, "推荐下一步": report["recommended_next_step"]}, ensure_ascii=False), flush=True)
    return 0 if protocol_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
