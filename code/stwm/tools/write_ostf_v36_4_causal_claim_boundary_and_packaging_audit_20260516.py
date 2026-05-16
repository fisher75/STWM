#!/usr/bin/env python3
"""V36.4: 冻结 V36 causal claim 边界，并检查 packaging 可复验性。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

V35_49_DECISION = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json"
V35_49_BENCH = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json"
V35_49_VIZ = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json"
V35_49_ATLAS = ROOT / "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_eval_20260516.json"
V35_49_MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
V35_49_FRONTEND_LOG = ROOT / "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.log"
V35_49_FIGURE_DIR = ROOT / "outputs/figures/stwm_ostf_v35_49_full_325_raw_video_closure"

V36_AUDIT = ROOT / "reports/stwm_ostf_v36_v35_49_causal_trace_contract_audit_20260516.json"
V36_INPUT = ROOT / "reports/stwm_ostf_v36_past_only_observed_trace_input_build_20260516.json"
V36_ROLLOUT = ROOT / "reports/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.json"
V36_SLICE = ROOT / "reports/stwm_ostf_v36_causal_unified_semantic_identity_slice_build_20260516.json"
V36_ORIG_DECISION = ROOT / "reports/stwm_ostf_v36_decision_20260516.json"
V36_VIZ = ROOT / "reports/stwm_ostf_v36_causal_past_only_world_model_visualization_manifest_20260516.json"

V36_1_TRACE_ATLAS = ROOT / "reports/stwm_ostf_v36_1_trace_rollout_failure_atlas_20260516.json"
V36_1_PRIOR_DECISION = ROOT / "reports/stwm_ostf_v36_1_strongest_prior_downstream_baseline_decision_20260516.json"
V36_1_DECISION = ROOT / "reports/stwm_ostf_v36_1_decision_20260516.json"

V36_2C_SELECTOR = ROOT / "reports/stwm_ostf_v36_2c_conservative_copy_default_selector_rollout_20260516.json"
V36_2C_SLICE = ROOT / "reports/stwm_ostf_v36_2c_conservative_selector_downstream_slice_build_20260516.json"
V36_2C_DOWNSTREAM = ROOT / "reports/stwm_ostf_v36_2c_conservative_selector_downstream_gate_decision_20260516.json"
V36_2C_SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_2c_conservative_selector_downstream_slice/M128_H32"

V36_3_EVAL = ROOT / "reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_eval_summary_20260516.json"
V36_3_DECISION = ROOT / "reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json"

REPORT = ROOT / "reports/stwm_ostf_v36_4_causal_claim_boundary_and_packaging_audit_20260516.json"
CLAIMS = ROOT / "reports/stwm_ostf_v36_4_machine_checkable_claim_table_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_4_CAUSAL_CLAIM_BOUNDARY_AND_PACKAGING_AUDIT_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v36_4_causal_claim_boundary_and_packaging_audit_20260516.log"


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def exists_entry(path: Path, kind: str, required: bool = True) -> dict[str, Any]:
    count = None
    if path.exists() and path.is_dir():
        count = sum(1 for p in path.rglob("*") if p.is_file())
    return {
        "path": rel(path),
        "kind": kind,
        "required": required,
        "exists": path.exists(),
        "file_count": count,
    }


def count_npz(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in ("train", "val", "test"):
        d = root / split
        out[split] = len(list(d.glob("*.npz"))) if d.exists() else 0
    out["all"] = sum(out.values())
    return out


def metric_pair(decision: dict[str, Any], val_key: str, test_key: str) -> dict[str, Any]:
    return {"val": decision.get(val_key), "test": decision.get(test_key)}


def claim_row(
    claim_id: str,
    status: str,
    claim_zh: str,
    evidence_path: Path,
    metric: str,
    value: Any,
    pass_gate: Any,
    forbidden_boundary_zh: str,
    reviewer_risk_zh: str = "",
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "status": status,
        "claim_zh": claim_zh,
        "evidence_path": rel(evidence_path),
        "metric": metric,
        "value": value,
        "pass_gate": pass_gate,
        "forbidden_boundary_zh": forbidden_boundary_zh,
        "reviewer_risk_zh": reviewer_risk_zh,
    }


def main() -> int:
    v35_49 = load(V35_49_DECISION)
    v36_audit = load(V36_AUDIT)
    v36_input = load(V36_INPUT)
    v36_rollout = load(V36_ROLLOUT)
    v36_slice = load(V36_SLICE)
    v36_orig = load(V36_ORIG_DECISION)
    v36_1 = load(V36_1_DECISION)
    v36_2c = load(V36_2C_SELECTOR)
    v36_2c_downstream = load(V36_2C_DOWNSTREAM)
    v36_3 = load(V36_3_DECISION)
    v36_3_eval = load(V36_3_EVAL)
    v36_viz = load(V36_VIZ)
    v35_viz = load(V35_49_VIZ)

    artifact_entries = [
        exists_entry(V35_49_DECISION, "report_json"),
        exists_entry(V35_49_BENCH, "report_json"),
        exists_entry(V35_49_VIZ, "visualization_manifest_json"),
        exists_entry(V35_49_ATLAS, "failure_atlas_json"),
        exists_entry(V35_49_MANIFEST, "manifest_json"),
        exists_entry(V35_49_FRONTEND_LOG, "log"),
        exists_entry(V35_49_FIGURE_DIR, "figure_dir"),
        exists_entry(V36_AUDIT, "report_json"),
        exists_entry(V36_INPUT, "report_json"),
        exists_entry(V36_ROLLOUT, "report_json"),
        exists_entry(V36_SLICE, "report_json"),
        exists_entry(V36_ORIG_DECISION, "report_json"),
        exists_entry(V36_VIZ, "visualization_manifest_json"),
        exists_entry(V36_1_TRACE_ATLAS, "report_json"),
        exists_entry(V36_1_PRIOR_DECISION, "report_json"),
        exists_entry(V36_1_DECISION, "report_json"),
        exists_entry(V36_2C_SELECTOR, "report_json"),
        exists_entry(V36_2C_SLICE, "report_json"),
        exists_entry(V36_2C_DOWNSTREAM, "report_json"),
        exists_entry(V36_2C_SLICE_ROOT, "cache_dir"),
        exists_entry(V36_3_EVAL, "report_json"),
        exists_entry(V36_3_DECISION, "report_json"),
    ]
    missing_required = [e for e in artifact_entries if e["required"] and not e["exists"]]

    v36_slice_counts = count_npz(V36_2C_SLICE_ROOT)
    v36_slice_count_ok = v36_slice_counts.get("all") == 325
    v36_viz_cases = v36_viz.get("cases", [])
    v36_viz_figures_exist = bool(v36_viz_cases) and all((ROOT / c.get("figure_path", "")).exists() for c in v36_viz_cases)
    v35_viz_cases = v35_viz.get("cases", [])
    v35_viz_figures_exist = bool(v35_viz_cases) and all((ROOT / c.get("figure_path", "")).exists() for c in v35_viz_cases)

    occlusion_value = (v36_3.get("identity_test_means") or {}).get("occlusion_reappear_retrieval_top1")
    occlusion_blocker = occlusion_value == 0.0

    causal_claim_allowed = bool(
        v36_3.get("m128_h32_causal_video_world_model_benchmark_claim_allowed")
        and v36_3.get("causal_benchmark_passed")
        and v36_3.get("future_trace_predicted_from_past_only")
        and v36_3.get("trace_no_harm_copy_val")
        and v36_3.get("trace_no_harm_copy_test")
        and v36_3.get("trace_beats_strongest_prior_val")
        and v36_3.get("trace_beats_strongest_prior_test")
        and v36_3.get("semantic_three_seed_passed")
        and v36_3.get("stable_preservation")
        and v36_3.get("identity_real_instance_three_seed_passed")
        and not v36_3.get("future_leakage_detected", True)
        and not v36_3.get("trajectory_degraded", True)
    )
    teacher_upper_bound_claim_allowed = bool(
        v36_audit.get("v35_49_is_teacher_trace_upper_bound")
        and v36_audit.get("claim_boundary_requires_rename")
        and v35_49.get("m128_h32_full_325_video_system_benchmark_claim_allowed")
    )
    packaging_complete = bool(
        not missing_required
        and v36_slice_count_ok
        and v36_viz_figures_exist
        and v35_viz_figures_exist
    )

    claims = [
        claim_row(
            "claim_v35_49_teacher_trace_upper_bound",
            "allowed_with_rename",
            "V35.49 可以作为 full 325 M128/H32 raw-video-derived teacher-trace upper-bound closure。",
            V36_AUDIT,
            "v35_49_is_teacher_trace_upper_bound",
            v36_audit.get("v35_49_is_teacher_trace_upper_bound"),
            "必须同时标注 frontend 读取 obs+future full clip；不能称为 causal past-only world model。",
            "禁止写成 V35.49 已完成因果世界模型闭环。",
            "reviewer 会追问 CoTracker 是否看见未来帧；必须主动披露。",
        ),
        claim_row(
            "claim_v36_3_causal_m128_h32_full325_benchmark",
            "allowed",
            "V36.3 允许 claim M128/H32 full-325 causal video world model benchmark：past-only trace input，经 selector trace、semantic state、real-instance identity retrieval 闭环通过。",
            V36_3_DECISION,
            "causal_benchmark_passed",
            v36_3.get("causal_benchmark_passed"),
            {
                "future_trace_predicted_from_past_only": v36_3.get("future_trace_predicted_from_past_only"),
                "trace_beats_strongest_prior_val": v36_3.get("trace_beats_strongest_prior_val"),
                "trace_beats_strongest_prior_test": v36_3.get("trace_beats_strongest_prior_test"),
                "semantic_three_seed_passed": v36_3.get("semantic_three_seed_passed"),
                "identity_real_instance_three_seed_passed": v36_3.get("identity_real_instance_three_seed_passed"),
                "future_leakage_detected": v36_3.get("future_leakage_detected"),
            },
            "只限 M128/H32、full 325 当前 benchmark；不能外推 H64/H96、M512/M1024、任意数据域、任意 horizon。",
            "需要保留 V35.49 teacher-trace upper-bound gap，并报告 occlusion/reappear identity 硬伤。",
        ),
        claim_row(
            "claim_frozen_v30_selector_trace_no_harm_prior",
            "allowed",
            "V36.2c 保守 copy-default selector 在 val/test 都不伤 copy，并且赢 strongest analytic prior。",
            V36_2C_SELECTOR,
            "copy_default_selector_ADE_minus_strongest_prior",
            {
                "val": ((v36_2c.get("summary_by_split") or {}).get("val") or {}).get("copy_default_selector_minus_strongest_prior_ADE"),
                "test": ((v36_2c.get("summary_by_split") or {}).get("test") or {}).get("copy_default_selector_minus_strongest_prior_ADE"),
            },
            {
                "no_harm_copy_val": v36_2c.get("no_harm_copy_val"),
                "no_harm_copy_test": v36_2c.get("no_harm_copy_test"),
                "beats_strongest_prior_val": v36_2c.get("beats_strongest_prior_val"),
                "beats_strongest_prior_test": v36_2c.get("beats_strongest_prior_test"),
            },
            "这是 frozen V30 + prior selector calibration，不是 V31/V32 trajectory architecture search。",
            "若 reviewer 只看纯 ADE，需要解释 selector 是 motion-forecasting 标准 multi-prior calibration，不是未来泄漏。",
        ),
        claim_row(
            "claim_causal_semantic_state_field",
            "allowed",
            "V36.3 在 causal predicted trace 上通过三 seed future semantic state / transition / uncertainty 评估。",
            V36_3_DECISION,
            "semantic_balanced_accuracy_val_test",
            {
                "changed": metric_pair(
                    v36_3,
                    "semantic_changed_balanced_accuracy_val_mean",
                    "semantic_changed_balanced_accuracy_test_mean",
                ),
                "hard": metric_pair(v36_3, "semantic_hard_balanced_accuracy_val_mean", "semantic_hard_balanced_accuracy_test_mean"),
                "uncertainty": metric_pair(
                    v36_3,
                    "semantic_uncertainty_balanced_accuracy_val_mean",
                    "semantic_uncertainty_balanced_accuracy_test_mean",
                ),
            },
            v36_3.get("semantic_three_seed_passed"),
            "这是 semantic state / transition / uncertainty field，不是 full open-vocabulary dense semantic segmentation field。",
            "semantic hard/changed 仍是中等强度，需要 failure atlas 和 category breakdown 支撑。",
        ),
        claim_row(
            "claim_real_instance_pairwise_identity_field",
            "allowed_with_explicit_risk",
            "V36.3 在 real-instance subset 上 pairwise identity retrieval 三 seed 通过。",
            V36_3_DECISION,
            "identity_test_means",
            v36_3.get("identity_test_means"),
            v36_3.get("identity_real_instance_three_seed_passed"),
            "VSPW pseudo identity 只能 diagnostic-only；不能把 pseudo slot identity 写进 identity field claim。",
            "occlusion/reappear retrieval top1=0.0，是当前 identity field 的硬风险，必须显式报告。",
        ),
        claim_row(
            "claim_occlusion_reappear_identity_solved",
            "not_allowed",
            "不能 claim occlusion/reappear identity 已解决。",
            V36_3_DECISION,
            "occlusion_reappear_retrieval_top1",
            occlusion_value,
            "必须 >0 且跨 seed/category 稳定，当前为 0.0。",
            "不得把高 overall identity retrieval 掩盖 occlusion/reappear failure。",
            "这是最像 reviewer 会抓住的身份恒常性问题。",
        ),
        claim_row(
            "claim_full_cvpr_scale_complete_system",
            "not_allowed",
            "不能 claim full CVPR-scale complete world model success。",
            V36_3_DECISION,
            "full_cvpr_scale_claim_allowed",
            v36_3.get("full_cvpr_scale_claim_allowed"),
            "需要更大尺度、更多数据域、更强 occlusion identity、完整外部复验；当前明确 false。",
            "不得外推到任意分辨率、任意 horizon 或 full open-vocabulary semantic field。",
            "顶会叙述应是 bounded benchmark success + 清晰下一步，而不是过度包装。",
        ),
        claim_row(
            "claim_v34_continuous_teacher_delta_route",
            "not_allowed",
            "不能回到 V34 continuous teacher embedding delta writer/gate/prototype/local expert 路线或把它包装成 semantic field success。",
            V36_3_DECISION,
            "route_boundary",
            "V35/V36 已改为 observed-predictable semantic state targets",
            "semantic target 必须保持低维/离散/可观测可预测状态变量。",
            "不得声称 continuous teacher embedding delta 是主方法。",
            "这是前期负结果沉淀出的边界，不要倒车。",
        ),
    ]

    remaining_risks = [
        {
            "risk_id": "occlusion_reappear_identity",
            "severity": "hard_blocker_for_full_identity_claim",
            "evidence": rel(V36_3_DECISION),
            "current_value": occlusion_value,
            "说明": "overall identity retrieval 很高，但 occlusion/reappear top1=0.0；必须做专门 failure atlas 和修复路线。",
        },
        {
            "risk_id": "semantic_medium_margin",
            "severity": "reviewer_risk",
            "evidence": rel(V36_3_DECISION),
            "current_value": {
                "changed_test": v36_3.get("semantic_changed_balanced_accuracy_test_mean"),
                "hard_test": v36_3.get("semantic_hard_balanced_accuracy_test_mean"),
                "uncertainty_test": v36_3.get("semantic_uncertainty_balanced_accuracy_test_mean"),
            },
            "说明": "semantic state field 已过 gate，但 hard/changed 分数不是压倒性，需要 category atlas 解释成功/失败边界。",
        },
        {
            "risk_id": "teacher_trace_upper_bound_gap",
            "severity": "claim_boundary_risk",
            "evidence": rel(V36_3_DECISION),
            "current_value": v36_3.get("semantic_gap_vs_v35_49_teacher_trace_upper_bound"),
            "说明": "必须主动说明 V35.49 是 teacher-trace upper-bound，V36.3 是因果版本，两者存在 gap。",
        },
        {
            "risk_id": "packaging_vs_claim",
            "severity": "release_risk" if not packaging_complete else "controlled",
            "evidence": rel(REPORT),
            "current_value": {
                "missing_required_count": len(missing_required),
                "v36_slice_counts": v36_slice_counts,
                "v36_viz_figures_exist": v36_viz_figures_exist,
                "v35_viz_figures_exist": v35_viz_figures_exist,
            },
            "说明": "claim 成立与 release 包完整是两件事；本审计将二者拆开。",
        },
    ]

    recommended_next_step = (
        "run_v36_5_occlusion_reappear_identity_and_reviewer_risk_audit"
        if causal_claim_allowed
        else "fix_v36_causal_claim_boundary_or_rerun_causal_benchmark"
    )

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.4",
        "v36_causal_claim_boundary_audit_done": True,
        "v35_49_is_teacher_trace_upper_bound": bool(v36_audit.get("v35_49_is_teacher_trace_upper_bound")),
        "v35_49_causal_claim_forbidden": not bool(v36_audit.get("v35_49_is_causal_past_only_world_model")),
        "v35_49_claim_requires_rename": bool(v36_audit.get("claim_boundary_requires_rename")),
        "v36_3_is_causal_past_only_m128_h32_full325_benchmark": causal_claim_allowed,
        "m128_h32_causal_video_world_model_benchmark_claim_allowed": causal_claim_allowed,
        "m128_h32_teacher_trace_upper_bound_claim_allowed": teacher_upper_bound_claim_allowed,
        "full_cvpr_scale_claim_allowed": False,
        "future_trace_predicted_from_past_only": bool(v36_3.get("future_trace_predicted_from_past_only")),
        "future_teacher_trace_input_allowed": False,
        "future_teacher_embedding_input_allowed": False,
        "teacher_as_method_allowed": False,
        "v30_backbone_frozen": bool(v36_3.get("v30_backbone_frozen")),
        "future_leakage_detected": bool(v36_3.get("future_leakage_detected")),
        "trajectory_degraded": bool(v36_3.get("trajectory_degraded")),
        "pseudo_identity_excluded_from_claim": bool(v36_3.get("identity_pseudo_targets_excluded_from_claim")),
        "semantic_three_seed_passed": bool(v36_3.get("semantic_three_seed_passed")),
        "stable_preservation": bool(v36_3.get("stable_preservation")),
        "identity_real_instance_three_seed_passed": bool(v36_3.get("identity_real_instance_three_seed_passed")),
        "occlusion_reappear_identity_top1": occlusion_value,
        "occlusion_reappear_identity_hard_risk": occlusion_blocker,
        "artifact_packaging_complete_for_v36_claim": packaging_complete,
        "required_artifact_entries": artifact_entries,
        "missing_required_artifacts": missing_required,
        "v36_2c_downstream_slice_npz_counts": v36_slice_counts,
        "v36_2c_downstream_slice_count_ok": v36_slice_count_ok,
        "v36_visualization_manifest_exists": V36_VIZ.exists(),
        "v36_visualization_figures_exist": v36_viz_figures_exist,
        "v35_49_visualization_figures_exist": v35_viz_figures_exist,
        "machine_checkable_claim_table_path": rel(CLAIMS),
        "allowed_claim_count": sum(1 for c in claims if c["status"].startswith("allowed")),
        "not_allowed_claim_count": sum(1 for c in claims if c["status"] == "not_allowed"),
        "remaining_risks": remaining_risks,
        "forbidden_extrapolations": [
            "不得把 V35.49 写成 causal past-only world model。",
            "不得 claim full CVPR-scale complete system。",
            "不得 claim H64/H96、M512/M1024 或 1B。",
            "不得 claim full open-vocabulary dense semantic segmentation field。",
            "不得 claim occlusion/reappear identity 已解决。",
            "不得把 VSPW pseudo identity 纳入 identity claim gate。",
            "不得回到 V34 continuous teacher embedding delta 作为语义主线。",
        ],
        "recommended_next_step": recommended_next_step,
        "中文总结": (
            "V36.4 将 V35.49 改名为 teacher-trace upper-bound，并确认 V36.3 才是 causal past-only M128/H32 full-325 benchmark。"
            "当前允许 bounded M128/H32 causal video world model benchmark claim；不允许 full CVPR-scale claim。"
            "occlusion/reappear identity top1=0.0 是硬风险，必须进入下一轮 failure atlas/reviewer-risk audit。"
        ),
    }

    claim_table = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.4",
        "machine_checkable_claim_table_built": True,
        "claims": claims,
        "remaining_risks": remaining_risks,
        "recommended_next_step": recommended_next_step,
        "中文总结": "该表把允许 claim、不允许 claim、证据路径、指标、pass gate 和禁止外推边界逐项绑定，便于机器检查和 reviewer-risk 复核。",
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    CLAIMS.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    CLAIMS.write_text(json.dumps(claim_table, indent=2, ensure_ascii=False), encoding="utf-8")

    allowed_lines = "\n".join(
        f"- `{c['claim_id']}`: {c['claim_zh']} 证据: `{c['evidence_path']}`"
        for c in claims
        if c["status"].startswith("allowed")
    )
    blocked_lines = "\n".join(
        f"- `{c['claim_id']}`: {c['claim_zh']} 当前值: `{c['value']}`"
        for c in claims
        if c["status"] == "not_allowed"
    )
    risk_lines = "\n".join(f"- `{r['risk_id']}`: {r['说明']}" for r in remaining_risks)
    artifact_lines = "\n".join(
        f"- `{e['path']}`: exists={e['exists']}, kind={e['kind']}, file_count={e['file_count']}"
        for e in artifact_entries
    )
    DOC.write_text(
        "# STWM OSTF V36.4 Causal Claim Boundary and Packaging Audit\n\n"
        "## 中文总结\n"
        f"{audit['中文总结']}\n\n"
        "## 允许 claim\n"
        f"{allowed_lines}\n\n"
        "## 不允许 claim\n"
        f"{blocked_lines}\n\n"
        "## 关键剩余风险\n"
        f"{risk_lines}\n\n"
        "## Artifact / Packaging 检查\n"
        f"- artifact_packaging_complete_for_v36_claim: {packaging_complete}\n"
        f"- v36_2c_downstream_slice_npz_counts: {v36_slice_counts}\n"
        f"- missing_required_artifacts: {len(missing_required)}\n"
        f"{artifact_lines}\n\n"
        "## Claim 边界\n"
        "- V35.49 只能称为 teacher-trace upper-bound closure，因为 CoTracker/full-clip frontend 看见 future frames。\n"
        "- V36.3 才能称为 causal past-only M128/H32 full-325 benchmark，因为 future trace 来自 past-only observed trace 的 selector rollout。\n"
        "- 当前不能 claim full CVPR-scale complete system、H64/H96、M512/M1024、任意 horizon、full open-vocabulary dense segmentation 或 occlusion/reappear identity solved。\n\n"
        f"## 输出\n"
        f"- machine_checkable_claim_table: `{rel(CLAIMS)}`\n"
        f"- audit_report: `{rel(REPORT)}`\n"
        f"- log: `{rel(LOG)}`\n"
        f"- recommended_next_step: `{recommended_next_step}`\n",
        encoding="utf-8",
    )
    LOG.write_text(
        "\n".join(
            [
                f"[{datetime.now(timezone.utc).isoformat()}] V36.4 因果 claim 边界审计完成。",
                f"V35.49 teacher-trace upper-bound: {audit['v35_49_is_teacher_trace_upper_bound']}",
                f"V36.3 causal benchmark claim allowed: {causal_claim_allowed}",
                f"artifact packaging complete: {packaging_complete}",
                f"occlusion/reappear identity top1: {occlusion_value}",
                f"recommended_next_step: {recommended_next_step}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "中文状态": "V36.4 因果 claim 边界审计完成",
                "v35_49_is_teacher_trace_upper_bound": audit["v35_49_is_teacher_trace_upper_bound"],
                "m128_h32_causal_video_world_model_benchmark_claim_allowed": causal_claim_allowed,
                "full_cvpr_scale_claim_allowed": False,
                "occlusion_reappear_identity_hard_risk": occlusion_blocker,
                "artifact_packaging_complete_for_v36_claim": packaging_complete,
                "recommended_next_step": recommended_next_step,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if causal_claim_allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
