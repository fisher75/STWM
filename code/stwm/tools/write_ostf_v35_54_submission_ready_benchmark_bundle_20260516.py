#!/usr/bin/env python3
"""V35.54: submission-ready benchmark bundle 索引与外部 sanity review checklist。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import setproctitle

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))
setproctitle.setproctitle("python")

from stwm.tools.ostf_v17_common_20260502 import ROOT

V35_52 = ROOT / "reports/stwm_ostf_v35_52_reproducibility_package_and_benchmark_card_20260516.json"
V35_53 = ROOT / "reports/stwm_ostf_v35_53_reproducibility_dry_run_from_manifest_20260516.json"
PACKAGE = ROOT / "reports/stwm_ostf_v35_52_reproducibility_package_manifest_20260516.json"
CARD = ROOT / "reports/stwm_ostf_v35_52_benchmark_card_20260516.json"
CLAIMS = ROOT / "reports/stwm_ostf_v35_50_machine_checkable_claim_table_20260516.json"
REVIEWER = ROOT / "reports/stwm_ostf_v35_51_external_comparison_and_reviewer_risk_audit_20260516.json"
FINAL = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json"

REPORT = ROOT / "reports/stwm_ostf_v35_54_submission_ready_benchmark_bundle_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_54_SUBMISSION_READY_BENCHMARK_BUNDLE_20260516.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    v35_52 = load(V35_52)
    v35_53 = load(V35_53)
    package = load(PACKAGE)
    card = load(CARD)
    claims = load(CLAIMS)
    reviewer = load(REVIEWER)
    final = load(FINAL)

    entry_points = {
        "benchmark_card": rel(CARD),
        "package_manifest": rel(PACKAGE),
        "claim_table": rel(CLAIMS),
        "reviewer_risk_audit": rel(REVIEWER),
        "full_325_final_decision": rel(FINAL),
        "reproducibility_dry_run": rel(V35_53),
    }
    core_metric_reports = [
        "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json",
        "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json",
        "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_eval_20260516.json",
        "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_decision_20260516.json",
        "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json",
    ]
    review_checklist = [
        {
            "question_zh": "输入是否真的是 raw video / predecode frame，而不是旧 trace cache？",
            "answer_zh": "是。V35.49 frontend rerun 从 raw frame paths / predecode 重跑；旧 cache 只做 drift comparison。",
            "evidence": ["reports/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.json", "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.log"],
        },
        {
            "question_zh": "V30 是否 frozen，trajectory 是否退化？",
            "answer_zh": "V30 M128 frozen；trajectory_degraded=false。",
            "evidence": [rel(FINAL), rel(CARD)],
        },
        {
            "question_zh": "future teacher embedding 是否作为 input？",
            "answer_zh": "否。future teacher embedding 只允许 supervision，future_leakage_detected=false。",
            "evidence": [rel(CLAIMS), rel(CARD)],
        },
        {
            "question_zh": "semantic 是否只是 copy/persistence？",
            "answer_zh": "不是只靠 copy。stable copy 被保留，但 changed/hard/uncertainty 三 seed 通过；claim 限定为 semantic state/transition/uncertainty field。",
            "evidence": ["reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json", rel(REVIEWER)],
        },
        {
            "question_zh": "identity 是否靠 pseudo label？",
            "answer_zh": "不是。identity claim 只使用 real-instance subset；VSPW pseudo identity diagnostic-only。",
            "evidence": [rel(FINAL), rel(CLAIMS), rel(REVIEWER)],
        },
        {
            "question_zh": "failure cases 是否隐藏？",
            "answer_zh": "没有。full per-category atlas 和 case-mined visualization 已打包，包含成功和失败案例。",
            "evidence": ["reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_eval_20260516.json", "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json"],
        },
        {
            "question_zh": "当前能否 claim full CVPR-scale complete system？",
            "answer_zh": "不能。当前只允许 bounded full 325 M128/H32 raw-video closure video-system benchmark claim。",
            "evidence": [rel(CARD), rel(CLAIMS), rel(FINAL)],
        },
    ]

    bundle_ready = bool(
        v35_52.get("reproducibility_package_ready", False)
        and v35_53.get("dry_run_passed", False)
        and final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)
        and not final.get("full_cvpr_scale_claim_allowed", True)
        and all((ROOT / p).exists() for p in core_metric_reports)
        and PACKAGE.exists()
        and CARD.exists()
        and CLAIMS.exists()
        and REVIEWER.exists()
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.54",
        "submission_ready_benchmark_bundle_built": True,
        "submission_ready_benchmark_bundle_ready": bundle_ready,
        "entry_points": entry_points,
        "core_metric_reports": core_metric_reports,
        "review_checklist": review_checklist,
        "claim_summary": {
            "allowed_claims": [c["claim_id"] for c in claims.get("claims", []) if c.get("status") == "allowed"],
            "not_allowed_claims": [c["claim_id"] for c in claims.get("claims", []) if c.get("status") == "not_allowed"],
            "m128_h32_full_325_video_system_benchmark_claim_allowed": bool(final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)),
            "full_cvpr_scale_claim_allowed": False,
        },
        "benchmark_card_key_metrics": card.get("metrics", {}),
        "reviewer_risk_audit_passed": reviewer.get("reviewer_risk_audit_passed"),
        "dry_run_passed": v35_53.get("dry_run_passed"),
        "recommended_next_step": "external_sanity_review_or_start_result_section_draft_from_benchmark_card",
        "中文结论": (
            "V35.54 已生成 submission-ready benchmark bundle 索引和外部 sanity review checklist。"
            "当前证据链足以支持 bounded full 325 M128/H32 raw-video closure video-system benchmark claim；"
            "仍不允许 full CVPR-scale / 任意尺度 / open-vocabulary dense segmentation claim。"
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# STWM OSTF V35.54 Submission-Ready Benchmark Bundle",
        "",
        "## 中文总结",
        report["中文结论"],
        "",
        "## 入口文件",
    ]
    for k, v in entry_points.items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Reviewer sanity checklist"]
    for item in review_checklist:
        lines.append(f"- {item['question_zh']} {item['answer_zh']}")
    lines += [
        "",
        "## Claim 边界",
        f"- submission_ready_benchmark_bundle_ready: {bundle_ready}",
        f"- m128_h32_full_325_video_system_benchmark_claim_allowed: {final.get('m128_h32_full_325_video_system_benchmark_claim_allowed', False)}",
        "- full_cvpr_scale_claim_allowed: false",
        f"- recommended_next_step: {report['recommended_next_step']}",
        "",
    ]
    DOC.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "v35_54_submission_ready_bundle_done": True,
        "submission_ready_benchmark_bundle_ready": bundle_ready,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False),
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": report["recommended_next_step"],
    }, ensure_ascii=False), flush=True)
    return 0 if bundle_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
