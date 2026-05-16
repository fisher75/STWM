#!/usr/bin/env python3
"""V36.5: V36 causal failure atlas + reviewer-risk audit，不训练新模型。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402
from stwm.tools import eval_ostf_v35_46_per_category_failure_atlas_20260516 as base  # noqa: E402

base.SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_2c_conservative_selector_downstream_slice/M128_H32"
base.SUBSET_MANIFEST = ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json"
base.V35_45_DECISION = ROOT / "reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json"
base.EVAL_REPORT = ROOT / "reports/stwm_ostf_v36_5_causal_failure_atlas_eval_20260516.json"
base.DECISION_REPORT = ROOT / "reports/stwm_ostf_v36_5_causal_failure_atlas_decision_20260516.json"
base.DOC = ROOT / "docs/STWM_OSTF_V36_5_CAUSAL_FAILURE_ATLAS_DECISION_20260516.md"
base.LOG = ROOT / "outputs/logs/stwm_ostf_v36_5_causal_failure_atlas_20260516.log"

V36_3_DECISION = ROOT / "reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json"
V36_4_AUDIT = ROOT / "reports/stwm_ostf_v36_4_causal_claim_boundary_and_packaging_audit_20260516.json"
V36_4_CLAIMS = ROOT / "reports/stwm_ostf_v36_4_machine_checkable_claim_table_20260516.json"
REVIEWER_AUDIT = ROOT / "reports/stwm_ostf_v36_5_reviewer_risk_audit_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_5_CAUSAL_FAILURE_ATLAS_AND_REVIEWER_RISK_AUDIT_20260516.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def category_risk_rows(eval_report: dict[str, Any]) -> list[dict[str, Any]]:
    boundaries = eval_report.get("failure_boundaries", {})
    out: list[dict[str, Any]] = []
    for name in ("semantic_fragile_categories_test", "identity_fragile_categories_test"):
        for row in boundaries.get(name, []) or []:
            out.append(
                {
                    "source": name,
                    "split": row.get("split"),
                    "category": row.get("category"),
                    "sample_count": row.get("sample_count"),
                    "risk_metrics": row.get("risk_metrics"),
                }
            )
    return out


def postprocess(code: int) -> int:
    v36_3 = load(V36_3_DECISION)
    v36_4 = load(V36_4_AUDIT)
    claims = load(V36_4_CLAIMS)
    eval_report = load(base.EVAL_REPORT)
    decision = load(base.DECISION_REPORT)

    causal_claim_allowed = bool(v36_3.get("m128_h32_causal_video_world_model_benchmark_claim_allowed"))
    atlas_ready = bool(causal_claim_allowed and eval_report.get("category_summary") and len(eval_report.get("categories", [])) >= 10)
    occlusion_value = (v36_3.get("identity_test_means") or {}).get("occlusion_reappear_retrieval_top1")
    category_risks = category_risk_rows(eval_report)
    semantic_gap = v36_3.get("semantic_gap_vs_v35_49_teacher_trace_upper_bound", {})

    reviewer_risks = [
        {
            "risk_id": "future_trace_teacher_upper_bound_confusion",
            "status": "controlled_by_claim_boundary",
            "evidence": rel(V36_4_AUDIT),
            "说明": "V35.49 已被明确改名为 teacher-trace upper-bound；V36.3 才是 causal past-only benchmark。",
        },
        {
            "risk_id": "occlusion_reappear_identity_failure",
            "status": "hard_risk",
            "evidence": rel(V36_3_DECISION),
            "metric": "occlusion_reappear_retrieval_top1",
            "value": occlusion_value,
            "说明": "该项为 0.0，不能 claim occlusion/reappear identity solved；下一步应做 occlusion-aware identity memory/reassociation 修复。",
        },
        {
            "risk_id": "semantic_medium_margin",
            "status": "needs_failure_boundary_explanation",
            "evidence": rel(base.EVAL_REPORT),
            "metric": {
                "semantic_changed_test": v36_3.get("semantic_changed_balanced_accuracy_test_mean"),
                "semantic_hard_test": v36_3.get("semantic_hard_balanced_accuracy_test_mean"),
                "semantic_uncertainty_test": v36_3.get("semantic_uncertainty_balanced_accuracy_test_mean"),
            },
            "说明": "semantic state field 已过三 seed，但分数不是压倒性，需要以类别图谱解释 high-motion/VIPSeg/changed/hard 边界。",
        },
        {
            "risk_id": "teacher_trace_upper_bound_gap",
            "status": "must_report",
            "evidence": rel(V36_3_DECISION),
            "metric": semantic_gap,
            "说明": "必须报告 V36 causal predicted trace 相对 V35.49 teacher-trace upper-bound 的 gap；不能把 upper-bound 当因果结果。",
        },
        {
            "risk_id": "pseudo_identity_overclaim",
            "status": "controlled",
            "evidence": rel(V36_3_DECISION),
            "metric": "identity_pseudo_targets_excluded_from_claim",
            "value": v36_3.get("identity_pseudo_targets_excluded_from_claim"),
            "说明": "VSPW pseudo identity 仍 diagnostic-only，不进入 identity claim gate。",
        },
    ]

    recommended = "fix_occlusion_reappear_identity_field" if occlusion_value == 0.0 else "write_v36_release_bundle_with_causal_claim_boundary"

    if base.EVAL_REPORT.exists():
        eval_report["current_completed_version"] = "V36.5"
        eval_report["source_scale"] = "full_325_m128_h32_causal_selector_trace"
        eval_report["atlas_ready"] = atlas_ready
        eval_report["reviewer_risk_rows"] = reviewer_risks
        eval_report["category_risk_rows"] = category_risks
        eval_report["中文结论"] = (
            "V36.5 已在 V36.2c causal selector trace slice 上完成 per-category failure atlas；该 atlas 用于解释 V36.3 causal claim 的成功/失败边界。"
        )
        base.EVAL_REPORT.write_text(json.dumps(eval_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if base.DECISION_REPORT.exists():
        decision["current_completed_version"] = "V36.5"
        decision["causal_failure_atlas_done"] = True
        decision["atlas_ready"] = atlas_ready
        decision["m128_h32_causal_video_world_model_benchmark_claim_allowed"] = causal_claim_allowed
        decision["full_cvpr_scale_claim_allowed"] = False
        decision["occlusion_reappear_identity_top1"] = occlusion_value
        decision["occlusion_reappear_identity_hard_risk"] = bool(occlusion_value == 0.0)
        decision["reviewer_risk_audit_path"] = rel(REVIEWER_AUDIT)
        decision["recommended_next_step"] = recommended
        decision["中文结论"] = (
            "V36.5 不训练新模型；它把 V36.3 causal benchmark 的类别级失败边界和 reviewer 风险拆开。"
            "当前最大硬风险是 occlusion/reappear identity top1=0.0。"
        )
        base.DECISION_REPORT.write_text(json.dumps(decision, indent=2, ensure_ascii=False), encoding="utf-8")

    reviewer_audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.5",
        "reviewer_risk_audit_done": True,
        "causal_failure_atlas_done": bool(base.EVAL_REPORT.exists()),
        "atlas_ready": atlas_ready,
        "v36_3_causal_claim_allowed": causal_claim_allowed,
        "full_cvpr_scale_claim_allowed": False,
        "claim_table_path": rel(V36_4_CLAIMS),
        "claim_table_claim_count": len(claims.get("claims", [])),
        "reviewer_risks": reviewer_risks,
        "category_risk_rows": category_risks,
        "occlusion_reappear_identity_top1": occlusion_value,
        "occlusion_reappear_identity_hard_risk": bool(occlusion_value == 0.0),
        "semantic_gap_vs_teacher_trace_upper_bound": semantic_gap,
        "recommended_next_step": recommended,
        "中文总结": (
            "V36.5 reviewer-risk audit 确认：V36.3 causal M128/H32 full-325 benchmark claim 可以保留，但不能升级为 full CVPR-scale。"
            "最明确的下一步是修 occlusion/reappear identity field，而不是继续扩大 claim 或训练 V34 路线。"
        ),
    }
    REVIEWER_AUDIT.write_text(json.dumps(reviewer_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    risk_lines = "\n".join(
        f"- `{r['risk_id']}`: {r['status']}。{r['说明']}" for r in reviewer_risks
    )
    category_lines = "\n".join(
        f"- {r['split']} / {r['category']} / sample_count={r['sample_count']} / risk={r['risk_metrics']}"
        for r in category_risks[:24]
    ) or "- 当前 category_risk_rows 为空；仍需在后续图谱中继续放大高风险类别。"
    DOC.write_text(
        "# STWM OSTF V36.5 Causal Failure Atlas and Reviewer-Risk Audit\n\n"
        "## 中文总结\n"
        f"{reviewer_audit['中文总结']}\n\n"
        "## Reviewer 风险矩阵\n"
        f"{risk_lines}\n\n"
        "## 类别级风险摘录\n"
        f"{category_lines}\n\n"
        "## Claim 边界\n"
        "- 允许：M128/H32 full-325 causal video world model benchmark。\n"
        "- 不允许：full CVPR-scale complete system、H64/H96、M512/M1024、full open-vocabulary semantic segmentation、occlusion/reappear identity solved。\n"
        "- V35.49 只能作为 teacher-trace upper-bound，不是 causal result。\n\n"
        "## 输出\n"
        f"- causal_failure_atlas_eval: `{rel(base.EVAL_REPORT)}`\n"
        f"- causal_failure_atlas_decision: `{rel(base.DECISION_REPORT)}`\n"
        f"- reviewer_risk_audit: `{rel(REVIEWER_AUDIT)}`\n"
        f"- log: `{rel(base.LOG)}`\n"
        f"- recommended_next_step: `{recommended}`\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "中文状态": "V36.5 causal failure atlas + reviewer-risk audit 完成",
                "atlas_ready": atlas_ready,
                "occlusion_reappear_identity_hard_risk": bool(occlusion_value == 0.0),
                "full_cvpr_scale_claim_allowed": False,
                "recommended_next_step": recommended,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if atlas_ready else code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--postprocess-only", action="store_true")
    args = ap.parse_args()
    code = 0 if args.postprocess_only else base.main()
    return postprocess(code)


if __name__ == "__main__":
    raise SystemExit(main())
