#!/usr/bin/env python3
"""V35.56: 冻结 V35.55 benchmark claim boundary，并生成 non-paper release bundle。"""
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

V35_49_FINAL = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json"
V35_50_AUDIT = ROOT / "reports/stwm_ostf_v35_50_cvpr_claim_boundary_and_packaging_audit_20260516.json"
V35_50_CLAIMS = ROOT / "reports/stwm_ostf_v35_50_machine_checkable_claim_table_20260516.json"
V35_51_REVIEWER = ROOT / "reports/stwm_ostf_v35_51_external_comparison_and_reviewer_risk_audit_20260516.json"
V35_52_CARD = ROOT / "reports/stwm_ostf_v35_52_benchmark_card_20260516.json"
V35_52_PACKAGE = ROOT / "reports/stwm_ostf_v35_52_reproducibility_package_manifest_20260516.json"
V35_53_DRY = ROOT / "reports/stwm_ostf_v35_53_reproducibility_dry_run_from_manifest_20260516.json"
V35_54_BUNDLE = ROOT / "reports/stwm_ostf_v35_54_submission_ready_benchmark_bundle_20260516.json"
V35_55_SANITY = ROOT / "reports/stwm_ostf_v35_55_external_sanity_review_package_20260516.json"

REPORT = ROOT / "reports/stwm_ostf_v35_56_claim_boundary_freeze_and_release_bundle_20260516.json"
FROZEN = ROOT / "reports/stwm_ostf_v35_56_frozen_claim_boundary_manifest_20260516.json"
RELEASE = ROOT / "reports/stwm_ostf_v35_56_non_paper_release_bundle_index_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_56_CLAIM_BOUNDARY_FREEZE_AND_RELEASE_BUNDLE_20260516.md"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    final = load(V35_49_FINAL)
    audit = load(V35_50_AUDIT)
    claims = load(V35_50_CLAIMS)
    reviewer = load(V35_51_REVIEWER)
    card = load(V35_52_CARD)
    package = load(V35_52_PACKAGE)
    dry = load(V35_53_DRY)
    bundle = load(V35_54_BUNDLE)
    sanity = load(V35_55_SANITY)

    claim_rows = list(claims.get("claims", []))
    allowed_claims = [row for row in claim_rows if row.get("status") == "allowed"]
    not_allowed_claims = [row for row in claim_rows if row.get("status") == "not_allowed"]

    frozen_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.56",
        "frozen_from_versions": ["V35.49", "V35.50", "V35.51", "V35.52", "V35.53", "V35.54", "V35.55"],
        "allowed_claims": allowed_claims,
        "not_allowed_claims": not_allowed_claims,
        "hard_boundaries": {
            "m128_h32_only": True,
            "selected_clip_count": final.get("selected_clip_count"),
            "full_325_raw_video_closure_only": True,
            "h64_h96_not_run": True,
            "m512_m1024_not_run": True,
            "full_cvpr_scale_claim_allowed": False,
            "open_vocabulary_dense_segmentation_claim_allowed": False,
            "v34_continuous_teacher_delta_route_claim_allowed": False,
            "teacher_as_method_allowed": False,
            "future_teacher_embedding_as_input_allowed": False,
            "pseudo_identity_claim_allowed": False,
        },
        "positive_claim_sentence_zh": (
            "STWM V35 在 full 325 M128/H32 raw-video closure benchmark 上，形成了 raw video / predecode frames → observed dense trace → "
            "frozen V30 future trace → future semantic state/transition/uncertainty → real-instance pairwise identity retrieval 的完整闭环。"
        ),
        "forbidden_overclaim_sentence_zh": (
            "当前不能声称任意分辨率、任意 horizon、H64/H96、M512/M1024、full open-vocabulary dense segmentation 或 full CVPR-scale complete system 已完成。"
        ),
        "evidence_entry_points": {
            "final_decision": rel(V35_49_FINAL),
            "claim_table": rel(V35_50_CLAIMS),
            "reviewer_risk_audit": rel(V35_51_REVIEWER),
            "benchmark_card": rel(V35_52_CARD),
            "package_manifest": rel(V35_52_PACKAGE),
            "dry_run": rel(V35_53_DRY),
            "submission_bundle": rel(V35_54_BUNDLE),
            "external_sanity_review": rel(V35_55_SANITY),
        },
    }

    release_bundle_index = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.56",
        "bundle_name": "stwm_v35_full325_m128_h32_raw_video_closure_non_paper_release_bundle",
        "bundle_ready": True,
        "entry_points": frozen_manifest["evidence_entry_points"],
        "minimum_files_to_share": [
            rel(V35_49_FINAL),
            rel(V35_50_CLAIMS),
            rel(V35_51_REVIEWER),
            rel(V35_52_CARD),
            rel(V35_52_PACKAGE),
            rel(V35_53_DRY),
            rel(V35_54_BUNDLE),
            rel(V35_55_SANITY),
            "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json",
            "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_eval_20260516.json",
            "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json",
            "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.log",
            "outputs/logs/stwm_ostf_v35_49_full_325_per_category_failure_atlas_20260516.log",
            "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_20260516.log",
        ],
        "recommended_read_order": [
            rel(V35_52_CARD),
            rel(V35_50_CLAIMS),
            rel(V35_55_SANITY),
            rel(V35_49_FINAL),
            rel(V35_51_REVIEWER),
            rel(V35_52_PACKAGE),
        ],
        "not_a_paper_draft": True,
        "notes_zh": "该 bundle 是证据索引，不是论文正文；用于外部 sanity review、内部复验和 claim 边界冻结。",
    }

    freeze_ready = bool(
        final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)
        and not final.get("full_cvpr_scale_claim_allowed", True)
        and audit.get("artifact_packaging_complete", False)
        and reviewer.get("reviewer_risk_audit_passed", False)
        and card.get("safety_and_claim_boundary", {}).get("full_cvpr_scale_claim_allowed") is False
        and package.get("reproducibility_package_ready", False)
        and dry.get("dry_run_passed", False)
        and bundle.get("submission_ready_benchmark_bundle_ready", False)
        and sanity.get("external_sanity_review_passed", False)
    )

    remaining_to_cvpr_scale = [
        {
            "gap": "scale_generalization",
            "status": "not_done",
            "说明": "尚未跑 H64/H96、M512/M1024 或更多真实视频域；当前只冻结 M128/H32 full 325。",
            "是否当前轮执行": False,
        },
        {
            "gap": "open_vocabulary_dense_semantic_segmentation",
            "status": "not_claimed",
            "说明": "当前 semantic 是 state/transition/uncertainty field，不是 full open-vocabulary dense segmentation。",
            "是否当前轮执行": False,
        },
        {
            "gap": "larger_real_instance_identity_provenance",
            "status": "optional_future_strengthening",
            "说明": "real-instance identity_count=121 已支撑当前 claim；若要更强 identity field scope，可继续扩大真实 instance provenance。",
            "是否当前轮执行": False,
        },
        {
            "gap": "independent_environment_replay",
            "status": "recommended_before_public_release",
            "说明": "V35.55 已完成本机 external sanity review；公开前最好在独立环境复验 package manifest。",
            "是否当前轮执行": False,
        },
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.56",
        "claim_boundary_freeze_done": True,
        "claim_boundary_freeze_ready": freeze_ready,
        "non_paper_release_bundle_index_built": True,
        "non_paper_release_bundle_ready": freeze_ready,
        "frozen_claim_boundary_manifest_path": rel(FROZEN),
        "non_paper_release_bundle_index_path": rel(RELEASE),
        "selected_clip_count": final.get("selected_clip_count"),
        "dataset_counts": final.get("dataset_counts"),
        "split_counts": final.get("split_counts"),
        "real_instance_identity_count": final.get("real_instance_identity_count"),
        "pseudo_identity_count": final.get("pseudo_identity_count"),
        "raw_frontend_rerun_success_rate": final.get("raw_frontend_rerun_success_rate"),
        "trace_drift_ok": final.get("trace_drift_ok"),
        "semantic_three_seed_passed": final.get("semantic_three_seed_passed"),
        "identity_real_instance_three_seed_passed": final.get("identity_real_instance_three_seed_passed"),
        "external_sanity_review_passed": sanity.get("external_sanity_review_passed"),
        "m128_h32_full_325_video_system_benchmark_claim_allowed": bool(
            freeze_ready and final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)
        ),
        "full_cvpr_scale_claim_allowed": False,
        "remaining_to_full_cvpr_scale": remaining_to_cvpr_scale,
        "recommended_next_step": "stop_and_return_to_claim_boundary_or_run_independent_environment_replay",
        "中文结论": (
            "V35.56 已冻结 V35.55 的 benchmark claim boundary，并生成 non-paper release bundle 索引。"
            "当前允许 bounded full 325 M128/H32 raw-video closure video-system benchmark claim；"
            "仍不允许 full CVPR-scale、任意尺度或 open-vocabulary dense segmentation claim。"
        ),
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(frozen_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    RELEASE.write_text(json.dumps(release_bundle_index, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    DOC.write_text(
        "# STWM OSTF V35.56 Claim Boundary Freeze and Release Bundle\n\n"
        "## 中文总结\n"
        + report["中文结论"]
        + "\n\n"
        "## 已冻结允许 claim\n"
        + "\n".join(f"- {row['claim_id']}: {row['claim_zh']}" for row in allowed_claims)
        + "\n\n"
        "## 已冻结不允许 claim\n"
        + "\n".join(f"- {row['claim_id']}: {row['claim_zh']}" for row in not_allowed_claims)
        + "\n\n"
        "## 当前顶会边界\n"
        f"- claim_boundary_freeze_ready: {freeze_ready}\n"
        f"- selected_clip_count: {final.get('selected_clip_count')}\n"
        f"- real_instance_identity_count: {final.get('real_instance_identity_count')}\n"
        f"- m128_h32_full_325_video_system_benchmark_claim_allowed: {report['m128_h32_full_325_video_system_benchmark_claim_allowed']}\n"
        "- full_cvpr_scale_claim_allowed: false\n"
        f"- frozen_claim_boundary_manifest: {rel(FROZEN)}\n"
        f"- non_paper_release_bundle_index: {rel(RELEASE)}\n"
        f"- recommended_next_step: {report['recommended_next_step']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "v35_56_claim_boundary_freeze_done": True,
        "claim_boundary_freeze_ready": freeze_ready,
        "non_paper_release_bundle_ready": freeze_ready,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": report["m128_h32_full_325_video_system_benchmark_claim_allowed"],
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": report["recommended_next_step"],
    }, ensure_ascii=False), flush=True)
    return 0 if freeze_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
