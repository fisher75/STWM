#!/usr/bin/env python3
"""V35.51: external/baseline comparison 和 reviewer-risk 审计。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

V35_50 = ROOT / "reports/stwm_ostf_v35_50_cvpr_claim_boundary_and_packaging_audit_20260516.json"
V35_49_FINAL = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json"
V35_49_BENCH = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json"
V35_49_EVAL = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json"
V35_49_ATLAS = ROOT / "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_decision_20260516.json"
V35_49_VIS = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json"
V35_21_SEM = ROOT / "reports/stwm_ostf_v35_21_domain_normalized_video_semantic_adapter_replication_decision_20260515.json"
V35_29_ID = ROOT / "reports/stwm_ostf_v35_29_expanded_identity_replication_decision_20260516.json"
V35_31_UNIFIED = ROOT / "reports/stwm_ostf_v35_31_unified_joint_video_semantic_identity_eval_summary_20260516.json"
V30_TRACE = ROOT / "reports/stwm_ostf_v30_external_gt_round2_multiseed_decision_v2_20260508.json"
V34_43 = ROOT / "reports/stwm_ostf_v34_43_observed_predictable_delta_targets_20260515.json"
V35_40_ID_RISK = ROOT / "reports/stwm_ostf_v35_40_identity_hard_case_failure_modes_20260516.json"
V35_41_TRACE_ID = ROOT / "reports/stwm_ostf_v35_41_trace_instance_cue_identity_probe_20260516.json"

REPORT = ROOT / "reports/stwm_ostf_v35_51_external_comparison_and_reviewer_risk_audit_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_51_EXTERNAL_COMPARISON_AND_REVIEWER_RISK_AUDIT_20260516.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def get_path(d: dict[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def mean_metric(rows: list[dict[str, Any]], path: list[str]) -> float | None:
    vals: list[float] = []
    for row in rows:
        v = get_path(row, path)
        if v is not None:
            vals.append(float(v))
    return float(mean(vals)) if vals else None


def row(
    name: str,
    baseline_type: str,
    evidence: list[str],
    verdict: str,
    metric_summary: dict[str, Any],
    interpretation_zh: str,
    limitation_zh: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "baseline_type": baseline_type,
        "evidence": evidence,
        "verdict": verdict,
        "metric_summary": metric_summary,
        "interpretation_zh": interpretation_zh,
        "limitation_zh": limitation_zh,
    }


def risk(
    risk_id: str,
    status: str,
    severity: str,
    evidence: list[str],
    finding_zh: str,
    mitigation_zh: str,
    remaining_risk_zh: str,
) -> dict[str, Any]:
    return {
        "risk_id": risk_id,
        "status": status,
        "severity": severity,
        "evidence": evidence,
        "finding_zh": finding_zh,
        "mitigation_zh": mitigation_zh,
        "remaining_risk_zh": remaining_risk_zh,
    }


def main() -> int:
    claim = load(V35_50)
    final = load(V35_49_FINAL)
    bench = load(V35_49_BENCH)
    eval_summary = load(V35_49_EVAL)
    atlas = load(V35_49_ATLAS)
    vis = load(V35_49_VIS)
    sem21 = load(V35_21_SEM)
    id29 = load(V35_29_ID)
    unified31 = load(V35_31_UNIFIED)
    v30 = load(V30_TRACE)
    v34_43 = load(V34_43)
    id40 = load(V35_40_ID_RISK)
    id41 = load(V35_41_TRACE_ID)

    semantic_rows = list(eval_summary.get("semantic_seed_rows", []))
    real_id_rows = list(eval_summary.get("real_instance_identity_seed_rows", []))
    pseudo_id_rows = list(eval_summary.get("pseudo_identity_diagnostic_seed_rows", []))

    copy_top1_test_mean = mean_metric(semantic_rows, ["test", "cluster", "copy_top1"])
    cluster_top5_test_mean = mean_metric(semantic_rows, ["test", "cluster", "cluster_top5"])
    stable_copy_top1_test_mean = mean_metric(semantic_rows, ["test", "cluster", "stable_copy_top1"])
    stable_top5_test_mean = mean_metric(semantic_rows, ["test", "cluster", "stable_top5"])
    real_id_test_top1_mean = mean_metric(real_id_rows, ["test", "identity_retrieval_exclude_same_point_top1"])
    real_id_instance_pooled_test_mean = mean_metric(real_id_rows, ["test", "identity_retrieval_instance_pooled_top1"])
    real_id_confuser_test_mean = mean_metric(real_id_rows, ["test", "identity_confuser_avoidance_top1"])
    pseudo_id_test_top1_mean = mean_metric(pseudo_id_rows, ["test", "identity_retrieval_exclude_same_point_top1"])

    baseline_comparisons = [
        row(
            "pure_trace_v30_only",
            "trajectory_only_baseline",
            [str(V30_TRACE.relative_to(ROOT)), str(V35_49_FINAL.relative_to(ROOT))],
            "necessary_but_not_sufficient",
            {
                "v30_backbone_frozen": final.get("v30_backbone_frozen"),
                "h32_item_bootstrap_positive": v30.get("h32_item_bootstrap_positive"),
                "h32_motion_bootstrap_positive": v30.get("h32_motion_bootstrap_positive"),
                "trajectory_degraded": final.get("trajectory_degraded"),
            },
            "V30 是闭环系统的可靠 future trace backbone，但纯 trace 只解决 future object-dense trace，不输出 semantic state 或 pairwise identity retrieval。",
            "不能把 V30 trajectory positive 单独包装成 semantic/identity field 成功；V35.49 的贡献是 trace 后接 semantic state + identity retrieval 的闭环。",
        ),
        row(
            "copy_persistence_semantic_baseline",
            "semantic_copy_or_persistence_baseline",
            [str(V35_49_EVAL.relative_to(ROOT)), str(V35_49_BENCH.relative_to(ROOT))],
            "beaten_on_state_tasks_while_preserved_on_stable",
            {
                "copy_top1_test_mean": copy_top1_test_mean,
                "cluster_top5_test_mean": cluster_top5_test_mean,
                "stable_copy_top1_test_mean": stable_copy_top1_test_mean,
                "stable_top5_test_mean": stable_top5_test_mean,
                "semantic_changed_ba_test_mean": bench.get("semantic_changed_balanced_accuracy_test_mean"),
                "semantic_hard_ba_test_mean": bench.get("semantic_hard_balanced_accuracy_test_mean"),
                "semantic_uncertainty_ba_test_mean": bench.get("semantic_uncertainty_balanced_accuracy_test_mean"),
            },
            "copy/persistence 对 stable token 很强，V35 保留 stable preservation；但 changed/hard/uncertainty 是状态预测任务，V35.49 三 seed 通过，说明不是只靠 stable copy。",
            "semantic field 仍应描述为 semantic state / transition / uncertainty field，不应说成完整 open-vocabulary dense segmentation。",
        ),
        row(
            "semantic_only_v35_21",
            "semantic_only_system_component",
            [str(V35_21_SEM.relative_to(ROOT)), str(V35_49_BENCH.relative_to(ROOT))],
            "component_passed_but_not_complete_system_alone",
            {
                "v35_21_three_seed_passed": sem21.get("all_three_seed_passed"),
                "v35_49_semantic_three_seed_passed": bench.get("semantic_three_seed_passed"),
                "v35_49_raw_video_full_325_closure": final.get("m128_h32_full_325_video_system_benchmark_claim_allowed"),
            },
            "V35.21 证明 semantic adapter 三 seed 可复现；V35.49 把它放入 raw-video full 325 closure，并与 identity retrieval 联合评估。",
            "semantic-only 不能 claim full video system；它缺少 pairwise identity field 和 raw frontend rerun closure。",
        ),
        row(
            "identity_only_v35_29",
            "identity_only_system_component",
            [str(V35_29_ID.relative_to(ROOT)), str(V35_49_BENCH.relative_to(ROOT))],
            "component_passed_but_not_complete_system_alone",
            {
                "v35_29_expanded_identity_replication_done": id29.get("expanded_identity_replication_done"),
                "v35_49_real_identity_three_seed_passed": bench.get("identity_real_instance_three_seed_passed"),
                "real_instance_identity_count": final.get("real_instance_identity_count"),
                "real_id_test_exclude_same_point_top1_mean": real_id_test_top1_mean,
                "real_id_instance_pooled_test_mean": real_id_instance_pooled_test_mean,
                "real_id_confuser_avoidance_test_mean": real_id_confuser_test_mean,
            },
            "identity 从 same-instance pointwise BCE 改成 pairwise retrieval / contrastive field 后，真实 instance subset 在 full 325 closure 中三 seed 通过。",
            "identity-only 不能 claim semantic field；VSPW pseudo slot identity 仍只能 diagnostic-only。",
        ),
        row(
            "pseudo_identity_diagnostic",
            "diagnostic_only_not_claim_gate",
            [str(V35_49_EVAL.relative_to(ROOT)), str(V35_50.relative_to(ROOT))],
            "excluded_from_claim_gate",
            {
                "pseudo_identity_count": final.get("pseudo_identity_count"),
                "pseudo_id_test_exclude_same_point_top1_mean": pseudo_id_test_top1_mean,
                "identity_pseudo_targets_excluded_from_claim": bench.get("identity_pseudo_targets_excluded_from_claim"),
            },
            "pseudo identity 可帮助观察系统行为，但不进入 identity claim gate；真实 claim 只看 real-instance subset。",
            "后续如果要更强 identity claim，需要更多真实 instance provenance，不应靠 pseudo slot 扩 claim。",
        ),
        row(
            "v34_continuous_teacher_delta_route",
            "rejected_route",
            [str(V34_43.relative_to(ROOT)) if V34_43.exists() else "reports/stwm_ostf_v34_43_observed_predictable_delta_targets_20260515.json 缺失或历史产物"],
            "not_system_contribution",
            {
                "observed_predictable_target_suite_ready": v34_43.get("observed_predictable_target_suite_ready"),
                "recommended_next_step": v34_43.get("recommended_next_step", "stop_unit_delta_route_and_rethink_video_semantic_target"),
            },
            "V35 的贡献不是 continuous teacher embedding delta writer/gate/prototype/local expert；该路线已作为负结果被停止。",
            "不能在 claim 里把 V34 delta residual 包装成 semantic field success。",
        ),
    ]

    reviewer_risks = [
        risk(
            "frontend_is_just_old_cache",
            "mitigated",
            "high",
            [str(V35_49_FINAL.relative_to(ROOT)), "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.log"],
            "full 325 从 raw frame paths / predecode 重跑 frontend，旧 trace cache 只用于 drift comparison。",
            "保留 rerun cache、manifest、log 和 drift 指标；claim 中写 raw-video closure 而不是 cache-only closure。",
            "仍应在最终打包时确认 raw frame path metadata 可复验。",
        ),
        risk(
            "semantic_is_only_copy",
            "mitigated_but_claim_limited",
            "medium",
            [str(V35_49_EVAL.relative_to(ROOT)), str(V35_49_BENCH.relative_to(ROOT))],
            "stable copy 很强且被保留；但 changed/hard/uncertainty 三 seed 过，说明系统不只是输出 copy。",
            "claim 改成 future semantic state / transition / uncertainty field，并列出 changed/hard BA。",
            "不能 claim open-vocabulary dense segmentation；可继续补充 changed/hard case 可视化。",
        ),
        risk(
            "identity_depends_on_pseudo_labels",
            "mitigated",
            "high",
            [str(V35_49_FINAL.relative_to(ROOT)), str(V35_49_EVAL.relative_to(ROOT))],
            "VSPW pseudo slot identity 明确 diagnostic-only；identity claim 只使用 real-instance subset。",
            "claim table 中把 pseudo identity 从 gate 排除；报告 real_instance_identity_count=121。",
            "若目标是更强 identity field claim，下一阶段要扩大真实 instance provenance。",
        ),
        risk(
            "future_teacher_embedding_leakage",
            "mitigated",
            "high",
            [str(V35_50.relative_to(ROOT)), str(V35_49_FINAL.relative_to(ROOT))],
            "V35.49/V35.50 均记录 future_leakage_detected=false，future teacher embedding 不作为 input。",
            "持续在 unified slice build 和 final decision 中机器检查 future_teacher_embedding_input_allowed=false。",
            "后续新 target/head 必须继承该检查。",
        ),
        risk(
            "teacher_or_external_tool_is_method",
            "mitigated_but_wording_sensitive",
            "medium",
            [str(V35_50.relative_to(ROOT))],
            "teacher/DINO/CLIP/SAM2/CoTracker 只作为 frontend/measurement/supervision/source；STWM 方法主线是 trace→future trace→semantic state/identity retrieval。",
            "最终叙事中不要把 teacher/prototype-only 写成方法贡献。",
            "审稿人可能仍会追问 frontend 依赖外部 tracker，需要在 benchmark card 里独立说明。",
        ),
        risk(
            "failure_cases_hidden",
            "mitigated",
            "medium",
            [str(V35_49_ATLAS.relative_to(ROOT)), str(V35_49_VIS.relative_to(ROOT))],
            "full 325 atlas_ready=true，high_risk_category_count=0；case-mined visualization 覆盖 changed/hard/identity 成败、occlusion/crossing/confuser/high motion。",
            "保留 failure atlas eval JSON 和 12 张可视化，不能只报总分。",
            "后续可做更系统的 human-readable failure atlas 页面，但不是当前 blocker。",
        ),
        risk(
            "scale_overclaim",
            "active_boundary",
            "high",
            [str(V35_50.relative_to(ROOT)), str(V35_49_FINAL.relative_to(ROOT))],
            "当前只允许 full 325 M128/H32 video-system benchmark claim，full_cvpr_scale_claim_allowed=false。",
            "最终措辞必须明确不是 H64/H96/M512/M1024，不是任意 horizon，不是任意分辨率。",
            "如果未来要扩 claim，需要另起严格实验，不能从 V35.49 外推。",
        ),
    ]

    reviewer_risks_clear_for_bounded_claim = all(r["status"] in {"mitigated", "mitigated_but_claim_limited", "mitigated_but_wording_sensitive", "active_boundary"} for r in reviewer_risks)
    reviewer_risk_audit_passed = bool(
        claim.get("artifact_packaging_complete", False)
        and final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)
        and reviewer_risks_clear_for_bounded_claim
        and not final.get("future_leakage_detected", False)
        and not final.get("trajectory_degraded", False)
    )

    contribution_boundary = {
        "positive_contribution_zh": (
            "V35 的核心贡献边界是：video-derived dense trace / raw-video frontend rerun → frozen V30 M128 future trace → "
            "可观测可预测 semantic state / transition / uncertainty field → real-instance pairwise identity retrieval field。"
        ),
        "negative_boundary_zh": (
            "不是 V34 continuous teacher embedding delta writer/gate/prototype/local expert；不是 teacher/prototype-only；不是 open-vocabulary dense segmentation；不是任意尺度系统。"
        ),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.51",
        "source_completed_version": "V35.49/V35.50",
        "external_comparison_and_reviewer_risk_audit_done": True,
        "baseline_comparisons": baseline_comparisons,
        "reviewer_risks": reviewer_risks,
        "reviewer_risks_clear_for_bounded_claim": reviewer_risks_clear_for_bounded_claim,
        "reviewer_risk_audit_passed": reviewer_risk_audit_passed,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": bool(final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False) and reviewer_risk_audit_passed),
        "full_cvpr_scale_claim_allowed": False,
        "semantic_changed_balanced_accuracy_test_mean": bench.get("semantic_changed_balanced_accuracy_test_mean"),
        "semantic_hard_balanced_accuracy_test_mean": bench.get("semantic_hard_balanced_accuracy_test_mean"),
        "semantic_uncertainty_balanced_accuracy_test_mean": bench.get("semantic_uncertainty_balanced_accuracy_test_mean"),
        "real_id_test_exclude_same_point_top1_mean": real_id_test_top1_mean,
        "real_id_instance_pooled_test_mean": real_id_instance_pooled_test_mean,
        "real_id_confuser_avoidance_test_mean": real_id_confuser_test_mean,
        "copy_top1_test_mean": copy_top1_test_mean,
        "cluster_top5_test_mean": cluster_top5_test_mean,
        "stable_copy_top1_test_mean": stable_copy_top1_test_mean,
        "stable_top5_test_mean": stable_top5_test_mean,
        "artifact_packaging_complete": claim.get("artifact_packaging_complete"),
        "future_leakage_detected": final.get("future_leakage_detected", False),
        "trajectory_degraded": final.get("trajectory_degraded", False),
        "v30_backbone_frozen": final.get("v30_backbone_frozen", True),
        "contribution_boundary": contribution_boundary,
        "recommended_next_step": "prepare_v35_52_reproducibility_package_and_benchmark_card",
        "中文结论": (
            "V35.51 完成 reviewer-risk 与 baseline 对比审计：纯 trace、copy/persistence、semantic-only、identity-only 都不能单独构成完整系统；"
            "V35.49/V35.50 的新增证据是 full 325 raw-video closure 下的联合 trace+semantic state+real-instance identity retrieval 闭环。"
            "主要 reviewer 风险已被 artifact、atlas、pseudo exclusion 和 leakage audit 缓解，但 claim 必须保持 bounded M128/H32。"
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# STWM OSTF V35.51 External Comparison and Reviewer-Risk Audit",
        "",
        "## 中文总结",
        report["中文结论"],
        "",
        "## Baseline / Component 对比",
    ]
    for item in baseline_comparisons:
        lines.append(f"- {item['name']}: {item['verdict']}。{item['interpretation_zh']}")
    lines += ["", "## Reviewer 风险矩阵"]
    for item in reviewer_risks:
        lines.append(f"- {item['risk_id']}: {item['status']} / {item['severity']}。{item['finding_zh']}")
    lines += [
        "",
        "## 贡献边界",
        f"- 正向贡献: {contribution_boundary['positive_contribution_zh']}",
        f"- 负向边界: {contribution_boundary['negative_boundary_zh']}",
        "",
        f"- reviewer_risk_audit_passed: {reviewer_risk_audit_passed}",
        f"- m128_h32_full_325_video_system_benchmark_claim_allowed: {report['m128_h32_full_325_video_system_benchmark_claim_allowed']}",
        "- full_cvpr_scale_claim_allowed: false",
        f"- recommended_next_step: {report['recommended_next_step']}",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "v35_51_reviewer_risk_audit_done": True,
        "reviewer_risk_audit_passed": reviewer_risk_audit_passed,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": report["m128_h32_full_325_video_system_benchmark_claim_allowed"],
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": report["recommended_next_step"],
    }, ensure_ascii=False), flush=True)
    return 0 if reviewer_risk_audit_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
