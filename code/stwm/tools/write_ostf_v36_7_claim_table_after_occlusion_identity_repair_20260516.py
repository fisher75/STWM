#!/usr/bin/env python3
"""V36.7: V36.6 occlusion/reappear 修复后更新 causal claim table。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

V36_4_AUDIT = ROOT / "reports/stwm_ostf_v36_4_causal_claim_boundary_and_packaging_audit_20260516.json"
V36_4_CLAIMS = ROOT / "reports/stwm_ostf_v36_4_machine_checkable_claim_table_20260516.json"
V36_5_RISK = ROOT / "reports/stwm_ostf_v36_5_reviewer_risk_audit_20260516.json"
V36_6_AUDIT = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_target_contract_audit_20260516.json"
V36_6_EVAL = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_eval_summary_20260516.json"
V36_6_DECISION = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_decision_20260516.json"
REPORT = ROOT / "reports/stwm_ostf_v36_7_claim_boundary_after_occlusion_identity_repair_20260516.json"
CLAIMS = ROOT / "reports/stwm_ostf_v36_7_machine_checkable_claim_table_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_7_CLAIM_BOUNDARY_AFTER_OCCLUSION_IDENTITY_REPAIR_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v36_7_claim_table_after_occlusion_identity_repair_20260516.log"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    v36_4 = load(V36_4_AUDIT)
    old_claims = load(V36_4_CLAIMS)
    v36_5 = load(V36_5_RISK)
    v36_6_audit = load(V36_6_AUDIT)
    v36_6_eval = load(V36_6_EVAL)
    v36_6 = load(V36_6_DECISION)

    claims = [c for c in old_claims.get("claims", []) if c.get("claim_id") != "claim_occlusion_reappear_identity_solved"]
    claims.append(
        {
            "claim_id": "claim_causal_occlusion_reappear_identity_eval_contract_repaired",
            "status": "allowed_with_target_contract_note",
            "claim_zh": (
                "V36.6 修复 occlusion/reappear identity target/eval contract 后，"
                "在 real-instance subset 的 teacher-vis-defined occlusion/reappear 点上三 seed 通过。"
            ),
            "evidence_path": rel(V36_6_DECISION),
            "metric": "occlusion_reappear_retrieval_top1",
            "value": {
                "val": v36_6.get("val_occlusion_reappear_retrieval_top1"),
                "test": v36_6.get("test_occlusion_reappear_retrieval_top1"),
                "val_total": v36_6.get("val_occlusion_reappear_total"),
                "test_total": v36_6.get("test_occlusion_reappear_total"),
            },
            "pass_gate": {
                "occlusion_reappear_identity_three_seed_passed": v36_6.get("occlusion_reappear_identity_three_seed_passed"),
                "teacher_future_vis_occ_target_available": v36_6.get("teacher_future_vis_occ_target_available"),
                "future_teacher_trace_input_allowed": v36_6.get("future_teacher_trace_input_allowed"),
                "future_trace_predicted_from_past_only": v36_6.get("future_trace_predicted_from_past_only"),
                "future_leakage_detected": v36_6.get("future_leakage_detected"),
            },
            "forbidden_boundary_zh": (
                "该 claim 只允许在 V36.6 修复后的 eval contract 下成立；future_trace_teacher_vis 只能作为 supervision/eval target，不能作为输入。"
                "仍不能 claim full CVPR-scale、任意遮挡场景泛化或 open-vocabulary dense segmentation。"
            ),
            "reviewer_risk_zh": "需要主动说明 V36.3 的 0.0 是旧 mask 为空导致，不是模型在真实遮挡样本上全错。",
        }
    )

    remaining_risks = [
        {
            "risk_id": "teacher_trace_upper_bound_gap",
            "status": "must_report",
            "evidence": "reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json",
            "说明": "V35.49 仍只能作为 teacher-trace upper-bound；V36 causal predicted trace 相对 upper-bound 有 semantic gap，必须报告。",
        },
        {
            "risk_id": "semantic_medium_margin",
            "status": "reviewer_risk",
            "evidence": rel(V36_5_RISK),
            "说明": "semantic changed/hard/uncertainty 已过三 seed，但不是压倒性分数；需要保持 per-category failure atlas。",
        },
        {
            "risk_id": "occlusion_target_contract_disclosure",
            "status": "controlled_with_disclosure",
            "evidence": rel(V36_6_AUDIT),
            "说明": "必须披露 occlusion target 从 predicted future_vis 改为 teacher future vis eval-only target；不得让 teacher trace 进入模型输入。",
        },
        {
            "risk_id": "full_cvpr_scale_not_allowed",
            "status": "hard_boundary",
            "evidence": rel(V36_4_AUDIT),
            "说明": "当前仍只允许 M128/H32 full-325 causal benchmark claim，不允许 full CVPR-scale complete system。",
        },
    ]

    allowed = [c for c in claims if str(c.get("status", "")).startswith("allowed")]
    not_allowed = [c for c in claims if c.get("status") == "not_allowed"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.7",
        "claim_table_updated_after_occlusion_identity_repair": True,
        "v36_4_causal_claim_boundary_still_valid": bool(v36_4.get("m128_h32_causal_video_world_model_benchmark_claim_allowed")),
        "v36_6_occlusion_reappear_identity_target_repaired": bool(v36_6.get("occlusion_reappear_identity_target_repaired")),
        "v36_6_occlusion_reappear_identity_three_seed_passed": bool(v36_6.get("occlusion_reappear_identity_three_seed_passed")),
        "previous_v36_5_occlusion_hard_risk": bool(v36_5.get("occlusion_reappear_identity_hard_risk")),
        "current_occlusion_reappear_identity_hard_risk": False if v36_6.get("occlusion_reappear_identity_three_seed_passed") else True,
        "m128_h32_causal_video_world_model_benchmark_claim_allowed": bool(
            v36_4.get("m128_h32_causal_video_world_model_benchmark_claim_allowed")
        ),
        "m128_h32_causal_identity_occlusion_reappear_claim_allowed": bool(
            v36_6.get("m128_h32_causal_identity_occlusion_reappear_claim_allowed")
        ),
        "full_cvpr_scale_claim_allowed": False,
        "allowed_claim_count": len(allowed),
        "not_allowed_claim_count": len(not_allowed),
        "machine_checkable_claim_table_path": rel(CLAIMS),
        "remaining_reviewer_risks": remaining_risks,
        "recommended_next_step": "write_v36_release_bundle_with_causal_claim_boundary",
        "中文总结": (
            "V36.7 已把 V36.6 的 occlusion/reappear target/eval contract 修复写入 machine-checkable claim table。"
            "当前允许 bounded M128/H32 full-325 causal benchmark claim，并允许在 teacher-vis eval target contract 下报告 occlusion/reappear identity positive；"
            "仍不允许 full CVPR-scale complete system。"
        ),
    }
    claim_table = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.7",
        "machine_checkable_claim_table_built": True,
        "source_claim_table": rel(V36_4_CLAIMS),
        "occlusion_reappear_repair_source": rel(V36_6_DECISION),
        "claims": claims,
        "remaining_reviewer_risks": remaining_risks,
        "recommended_next_step": report["recommended_next_step"],
        "中文总结": "V36.7 claim table 已反映 occlusion/reappear identity 的 target/eval contract 修复，同时保留所有 forbidden overclaim 边界。",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    CLAIMS.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    CLAIMS.write_text(json.dumps(claim_table, indent=2, ensure_ascii=False), encoding="utf-8")
    allowed_lines = "\n".join(f"- `{c['claim_id']}`: {c['claim_zh']}" for c in allowed)
    not_allowed_lines = "\n".join(f"- `{c['claim_id']}`: {c['claim_zh']}" for c in not_allowed)
    risk_lines = "\n".join(f"- `{r['risk_id']}`: {r['status']}。{r['说明']}" for r in remaining_risks)
    DOC.write_text(
        "# STWM OSTF V36.7 Claim Boundary After Occlusion Identity Repair\n\n"
        "## 中文总结\n"
        f"{report['中文总结']}\n\n"
        "## 已允许 claim\n"
        f"{allowed_lines}\n\n"
        "## 仍不允许 claim\n"
        f"{not_allowed_lines}\n\n"
        "## 仍需披露的 reviewer 风险\n"
        f"{risk_lines}\n\n"
        "## 关键指标\n"
        f"- val_occlusion_reappear_retrieval_top1: {v36_6.get('val_occlusion_reappear_retrieval_top1')}\n"
        f"- test_occlusion_reappear_retrieval_top1: {v36_6.get('test_occlusion_reappear_retrieval_top1')}\n"
        f"- val_occlusion_reappear_total: {v36_6.get('val_occlusion_reappear_total')}\n"
        f"- test_occlusion_reappear_total: {v36_6.get('test_occlusion_reappear_total')}\n"
        "- future_teacher_trace_input_allowed: false\n"
        "- future_trace_predicted_from_past_only: true\n"
        "- full_cvpr_scale_claim_allowed: false\n\n"
        "## 输出\n"
        f"- updated_claim_table: `{rel(CLAIMS)}`\n"
        f"- decision_report: `{rel(REPORT)}`\n"
        f"- recommended_next_step: `{report['recommended_next_step']}`\n",
        encoding="utf-8",
    )
    LOG.write_text(
        "\n".join(
            [
                f"[{datetime.now(timezone.utc).isoformat()}] V36.7 claim table update 完成。",
                f"occlusion_reappear_identity_three_seed_passed={v36_6.get('occlusion_reappear_identity_three_seed_passed')}",
                f"full_cvpr_scale_claim_allowed=False",
                f"recommended_next_step={report['recommended_next_step']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "中文状态": "V36.7 claim table 已更新",
                "m128_h32_causal_identity_occlusion_reappear_claim_allowed": report[
                    "m128_h32_causal_identity_occlusion_reappear_claim_allowed"
                ],
                "full_cvpr_scale_claim_allowed": False,
                "recommended_next_step": report["recommended_next_step"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
