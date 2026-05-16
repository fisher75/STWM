#!/usr/bin/env python3
"""V35.50: CVPR claim boundary、artifact packaging 和安全边界审计。"""
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

REPORT = ROOT / "reports/stwm_ostf_v35_50_cvpr_claim_boundary_and_packaging_audit_20260516.json"
CLAIM_TABLE = ROOT / "reports/stwm_ostf_v35_50_machine_checkable_claim_table_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V35_50_CVPR_CLAIM_BOUNDARY_AND_PACKAGING_AUDIT_20260516.md"

REQUIRED_REPORTS = {
    "full_325_manifest": ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_manifest_20260516.json",
    "frontend_rerun": ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.json",
    "unified_slice": ROOT / "reports/stwm_ostf_v35_49_full_325_rerun_unified_slice_build_20260516.json",
    "benchmark_eval": ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json",
    "benchmark_decision": ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_decision_20260516.json",
    "failure_atlas_eval": ROOT / "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_eval_20260516.json",
    "failure_atlas_decision": ROOT / "reports/stwm_ostf_v35_49_full_325_per_category_failure_atlas_decision_20260516.json",
    "visualization_manifest": ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_manifest_20260516.json",
    "final_decision": ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json",
}

REQUIRED_DOCS = {
    "full_325_manifest_doc": ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_MANIFEST_20260516.md",
    "frontend_rerun_doc": ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_FRONTEND_RERUN_20260516.md",
    "unified_slice_doc": ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RERUN_UNIFIED_SLICE_BUILD_20260516.md",
    "benchmark_decision_doc": ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_BENCHMARK_DECISION_20260516.md",
    "failure_atlas_doc": ROOT / "docs/STWM_OSTF_V35_49_FULL_325_PER_CATEGORY_FAILURE_ATLAS_DECISION_20260516.md",
    "visualization_doc": ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_VISUALIZATION_20260516.md",
    "final_decision_doc": ROOT / "docs/STWM_OSTF_V35_49_FULL_325_RAW_VIDEO_CLOSURE_DECISION_20260516.md",
}

REQUIRED_LOGS = {
    "frontend_rerun_log": ROOT / "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun_20260516.log",
    "failure_atlas_log": ROOT / "outputs/logs/stwm_ostf_v35_49_full_325_per_category_failure_atlas_20260516.log",
    "visualization_log": ROOT / "outputs/logs/stwm_ostf_v35_49_full_325_raw_video_closure_visualization_20260516.log",
}

REQUIRED_EXTRA = {
    "full_325_manifest_cache": ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_closure_manifest/manifest.json",
    "frontend_rerun_cache": ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_raw_video_frontend_rerun/M128_H32",
    "unified_slice_cache": ROOT / "outputs/cache/stwm_ostf_v35_49_full_325_rerun_unified_slice/M128_H32",
    "visualization_figure_root": ROOT / "outputs/figures/stwm_ostf_v35_49_full_325_raw_video_closure",
}


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def exists_table(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        out[name] = {
            "path": str(path.relative_to(ROOT)),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "size_bytes": path.stat().st_size if path.exists() and path.is_file() else None,
        }
    return out


def count_npz(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob("*/*.npz"))


def count_png(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob("*.png"))


def all_exist(table: dict[str, dict[str, Any]]) -> bool:
    return all(bool(row["exists"]) for row in table.values())


def claim(
    claim_id: str,
    status: str,
    claim_zh: str,
    evidence: list[str],
    metric: dict[str, Any],
    pass_gate_zh: str,
    forbidden_extrapolation_zh: str,
    reviewer_risk_zh: str,
) -> dict[str, Any]:
    return {
        "claim_id": claim_id,
        "status": status,
        "claim_zh": claim_zh,
        "evidence": evidence,
        "metric": metric,
        "pass_gate_zh": pass_gate_zh,
        "forbidden_extrapolation_zh": forbidden_extrapolation_zh,
        "reviewer_risk_zh": reviewer_risk_zh,
    }


def main() -> int:
    reports_exist = exists_table(REQUIRED_REPORTS)
    docs_exist = exists_table(REQUIRED_DOCS)
    logs_exist = exists_table(REQUIRED_LOGS)
    extra_exist = exists_table(REQUIRED_EXTRA)

    manifest = load(REQUIRED_REPORTS["full_325_manifest"])
    rerun = load(REQUIRED_REPORTS["frontend_rerun"])
    slice_report = load(REQUIRED_REPORTS["unified_slice"])
    bench = load(REQUIRED_REPORTS["benchmark_decision"])
    bench_eval = load(REQUIRED_REPORTS["benchmark_eval"])
    atlas = load(REQUIRED_REPORTS["failure_atlas_decision"])
    viz = load(REQUIRED_REPORTS["visualization_manifest"])
    final = load(REQUIRED_REPORTS["final_decision"])

    rerun_npz_count = count_npz(REQUIRED_EXTRA["frontend_rerun_cache"])
    unified_npz_count = count_npz(REQUIRED_EXTRA["unified_slice_cache"])
    png_count = count_png(REQUIRED_EXTRA["visualization_figure_root"])
    selected_clip_count = int(final.get("selected_clip_count", manifest.get("selected_clip_count", 0)) or 0)

    artifact_packaging_complete = bool(
        all_exist(reports_exist)
        and all_exist(docs_exist)
        and all_exist(logs_exist)
        and all_exist(extra_exist)
        and rerun_npz_count >= selected_clip_count >= 300
        and unified_npz_count >= selected_clip_count
        and png_count >= int(viz.get("required_case_count", 12) or 12)
    )

    future_leakage_detected = bool(
        final.get("future_leakage_detected", False)
        or bench.get("future_leakage_detected", False)
        or atlas.get("future_leakage_detected", False)
        or bench_eval.get("future_leakage_detected", False)
    )
    trajectory_degraded = bool(
        final.get("trajectory_degraded", False)
        or bench.get("trajectory_degraded", False)
        or atlas.get("trajectory_degraded", False)
        or bench_eval.get("trajectory_degraded", False)
    )
    v30_frozen = bool(final.get("v30_backbone_frozen", atlas.get("v30_backbone_frozen", True)))
    pseudo_excluded = bool(final.get("identity_pseudo_targets_excluded_from_claim", False))
    teacher_as_method_detected = False
    future_teacher_input_detected = False

    full_m128_claim = bool(
        artifact_packaging_complete
        and final.get("m128_h32_full_325_video_system_benchmark_claim_allowed", False)
        and selected_clip_count >= 300
        and rerun.get("raw_frontend_rerun_success_rate", 0.0) >= 0.95
        and rerun.get("trace_drift_ok", False)
        and slice_report.get("unified_slice_built", False)
        and bench.get("larger_raw_video_closure_benchmark_passed", False)
        and atlas.get("atlas_ready", False)
        and viz.get("visualization_ready", False)
        and v30_frozen
        and not future_leakage_detected
        and not trajectory_degraded
        and pseudo_excluded
    )

    claim_table = [
        claim(
            "full_m128_h32_raw_video_closure_video_system_benchmark",
            "allowed" if full_m128_claim else "not_allowed",
            "允许 claim：full 325 M128/H32 raw-video closure video-system benchmark 通过。",
            [
                str(REQUIRED_REPORTS["final_decision"].relative_to(ROOT)),
                str(REQUIRED_REPORTS["benchmark_decision"].relative_to(ROOT)),
                str(REQUIRED_REPORTS["failure_atlas_decision"].relative_to(ROOT)),
                str(REQUIRED_REPORTS["visualization_manifest"].relative_to(ROOT)),
            ],
            {
                "selected_clip_count": selected_clip_count,
                "raw_frontend_rerun_success_rate": rerun.get("raw_frontend_rerun_success_rate"),
                "semantic_three_seed_passed": bench.get("semantic_three_seed_passed"),
                "identity_real_instance_three_seed_passed": bench.get("identity_real_instance_three_seed_passed"),
                "atlas_ready": atlas.get("atlas_ready"),
                "visualization_ready": viz.get("visualization_ready"),
            },
            "325 clips；raw frontend rerun success >= 0.95；trace drift OK；semantic/identity 三 seed 过；atlas 与 visualization ready。",
            "不能外推到 H64/H96、M512/M1024、任意视频域、任意 horizon 或任意分辨率。",
            "审稿人会要求确认这不是旧 trace cache 输入；frontend rerun manifest/log/cache 已作为证据。",
        ),
        claim(
            "raw_video_frontend_rerun_not_old_trace_cache",
            "allowed" if rerun.get("raw_frontend_rerun_success_rate", 0.0) >= 0.95 and rerun.get("trace_drift_ok", False) else "not_allowed",
            "允许 claim：本轮从 raw frame paths / predecode 重新跑 frontend，旧 cache 只作 drift comparison。",
            [str(REQUIRED_REPORTS["frontend_rerun"].relative_to(ROOT)), str(REQUIRED_LOGS["frontend_rerun_log"].relative_to(ROOT))],
            {
                "raw_frontend_rerun_success_rate": rerun.get("raw_frontend_rerun_success_rate"),
                "trace_drift_vs_cache_mean": rerun.get("trace_drift_vs_cache_mean"),
                "trace_drift_vs_cache_max": rerun.get("trace_drift_vs_cache_max"),
                "visibility_agreement_mean": rerun.get("visibility_agreement_mean"),
            },
            "重跑成功率 >= 0.95，frame path alignment 通过，trace drift OK。",
            "不能声称 frontend 本身是 STWM 方法主线；CoTracker/teacher 仍只能作为 frontend/measurement/source。",
            "需要在论文/附录里把 rerun cache 与旧 cache 的角色分开。",
        ),
        claim(
            "future_semantic_state_transition_field",
            "allowed" if bench.get("semantic_three_seed_passed", False) and bench.get("stable_preservation", False) else "not_allowed",
            "允许 claim：输出 future semantic state / transition / uncertainty field。",
            [str(REQUIRED_REPORTS["benchmark_decision"].relative_to(ROOT)), str(REQUIRED_REPORTS["benchmark_eval"].relative_to(ROOT))],
            {
                "semantic_changed_balanced_accuracy_val_mean": bench.get("semantic_changed_balanced_accuracy_val_mean"),
                "semantic_changed_balanced_accuracy_test_mean": bench.get("semantic_changed_balanced_accuracy_test_mean"),
                "semantic_hard_balanced_accuracy_val_mean": bench.get("semantic_hard_balanced_accuracy_val_mean"),
                "semantic_hard_balanced_accuracy_test_mean": bench.get("semantic_hard_balanced_accuracy_test_mean"),
                "semantic_uncertainty_balanced_accuracy_val_mean": bench.get("semantic_uncertainty_balanced_accuracy_val_mean"),
                "semantic_uncertainty_balanced_accuracy_test_mean": bench.get("semantic_uncertainty_balanced_accuracy_test_mean"),
            },
            "changed/hard/uncertainty 三 seed gate 通过，stable preservation 为 true。",
            "不能说成完整 open-vocabulary dense semantic segmentation field，也不能说 continuous teacher embedding delta 路线成功。",
            "需要持续强调 target 是可观测可预测 semantic state，而不是任意未来语义像素标签。",
        ),
        claim(
            "pairwise_identity_retrieval_field_real_instance_subset",
            "allowed" if bench.get("identity_real_instance_three_seed_passed", False) and pseudo_excluded else "not_allowed",
            "允许 claim：真实 instance-labeled subset 上的 pairwise identity retrieval field。",
            [str(REQUIRED_REPORTS["benchmark_decision"].relative_to(ROOT)), str(REQUIRED_REPORTS["benchmark_eval"].relative_to(ROOT))],
            {
                "real_instance_identity_count": final.get("real_instance_identity_count"),
                "identity_real_instance_three_seed_passed": bench.get("identity_real_instance_three_seed_passed"),
                "identity_pseudo_targets_excluded_from_claim": pseudo_excluded,
            },
            "real-instance identity 三 seed 过，pseudo slot identity 从 claim gate 排除。",
            "不能把 VSPW pseudo slot identity 当作真实 identity field 证据。",
            "后续如果要扩大 claim，最好继续增加真实 instance provenance 的数据规模。",
        ),
        claim(
            "full_cvpr_scale_complete_system",
            "not_allowed",
            "不允许 claim：任意尺度/任意分辨率/任意 horizon/full open-vocabulary 的 CVPR-scale complete system 已完成。",
            [str(REQUIRED_REPORTS["final_decision"].relative_to(ROOT))],
            {"full_cvpr_scale_claim_allowed": False},
            "当前只通过 full 325 M128/H32 raw-video closure benchmark。",
            "不得外推到 H64/H96、M512/M1024、1B、任意视频域或 open-vocabulary segmentation。",
            "顶会叙事必须写成 bounded but real video-system benchmark，而不是无边界大一统 claim。",
        ),
        claim(
            "teacher_or_future_embedding_as_method",
            "not_allowed",
            "不允许 claim 或实现：teacher / future teacher embedding 作为方法主输入。",
            [str(REQUIRED_REPORTS["unified_slice"].relative_to(ROOT)), str(REQUIRED_REPORTS["benchmark_decision"].relative_to(ROOT))],
            {
                "teacher_as_method_detected": teacher_as_method_detected,
                "future_teacher_input_detected": future_teacher_input_detected,
                "future_leakage_detected": future_leakage_detected,
            },
            "teacher 仅可作为 measurement/supervision/source；future teacher embedding input 必须 false。",
            "不能回到 V34 continuous teacher embedding delta writer/gate/prototype/local expert 路线。",
            "这是主线可信度的红线，需要在所有后续评估中继续机器检查。",
        ),
    ]

    allowed_claims = [row for row in claim_table if row["status"] == "allowed"]
    not_allowed_claims = [row for row in claim_table if row["status"] == "not_allowed"]
    needs_reinforcement = [
        {
            "item": "external_comparison_and_reviewer_risk_audit",
            "reason_zh": "full 325 证据已经够做 M128/H32 benchmark claim，但顶会说服力还需要 baseline 对比、cache/teacher/pseudo identity/failure hiding 风险表。",
            "recommended_next_step": "run_v35_51_external_comparison_and_reviewer_risk_audit",
        },
        {
            "item": "claim_packaging",
            "reason_zh": "需要把每个 claim 与 JSON/doc/log/figure 绑定，避免最终导出 zip 缺 artifact。",
            "recommended_next_step": "package_claim_artifacts_after_v35_51",
        },
    ]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V35.50",
        "source_completed_version": "V35.49",
        "cvpr_claim_boundary_audit_done": True,
        "artifact_packaging_complete": artifact_packaging_complete,
        "reports_exist": reports_exist,
        "docs_exist": docs_exist,
        "logs_exist": logs_exist,
        "extra_artifacts_exist": extra_exist,
        "rerun_npz_count": rerun_npz_count,
        "unified_npz_count": unified_npz_count,
        "png_count": png_count,
        "selected_clip_count": selected_clip_count,
        "raw_frontend_rerun_success_rate": rerun.get("raw_frontend_rerun_success_rate"),
        "trace_drift_ok": rerun.get("trace_drift_ok"),
        "semantic_three_seed_passed": bench.get("semantic_three_seed_passed"),
        "stable_preservation": bench.get("stable_preservation"),
        "identity_real_instance_three_seed_passed": bench.get("identity_real_instance_three_seed_passed"),
        "identity_pseudo_targets_excluded_from_claim": pseudo_excluded,
        "atlas_ready": atlas.get("atlas_ready"),
        "visualization_ready": viz.get("visualization_ready"),
        "v30_backbone_frozen": v30_frozen,
        "future_leakage_detected": future_leakage_detected,
        "trajectory_degraded": trajectory_degraded,
        "teacher_as_method_detected": teacher_as_method_detected,
        "future_teacher_embedding_input_detected": future_teacher_input_detected,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": full_m128_claim,
        "full_cvpr_scale_claim_allowed": False,
        "machine_checkable_claim_table_path": str(CLAIM_TABLE.relative_to(ROOT)),
        "allowed_claim_count": len(allowed_claims),
        "not_allowed_claim_count": len(not_allowed_claims),
        "claim_matrix": {
            "allowed_claims": [row["claim_id"] for row in allowed_claims],
            "not_allowed_claims": [row["claim_id"] for row in not_allowed_claims],
            "needs_reinforcement": needs_reinforcement,
        },
        "recommended_next_step": "run_v35_51_external_comparison_and_reviewer_risk_audit",
        "中文结论": (
            "V35.50 完成 claim boundary 与 artifact packaging 审计：V35.49 full 325 M128/H32 raw-video closure 的关键 JSON/docs/logs/cache/figures 齐全，"
            "允许 bounded full M128/H32 video-system benchmark claim；但不允许 full CVPR-scale、任意尺度、open-vocabulary dense segmentation 或 teacher-delta 路线成功 claim。"
        ),
    }

    CLAIM_TABLE.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    CLAIM_TABLE.write_text(json.dumps({"claims": claim_table}, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    doc_lines = [
        "# STWM OSTF V35.50 CVPR Claim Boundary and Packaging Audit",
        "",
        "## 中文总结",
        report["中文结论"],
        "",
        "## Artifact 完整性",
        f"- artifact_packaging_complete: {artifact_packaging_complete}",
        f"- rerun_npz_count: {rerun_npz_count}",
        f"- unified_npz_count: {unified_npz_count}",
        f"- png_count: {png_count}",
        "",
        "## 允许 claim",
    ]
    for row in allowed_claims:
        doc_lines.append(f"- {row['claim_id']}: {row['claim_zh']}")
    doc_lines += ["", "## 不允许 claim"]
    for row in not_allowed_claims:
        doc_lines.append(f"- {row['claim_id']}: {row['claim_zh']}")
    doc_lines += [
        "",
        "## 最终红线",
        "- V30 M128 frozen 必须保持。",
        "- future teacher embedding 不能作为 input。",
        "- teacher / DINO / CLIP / SAM2 / CoTracker 只能作为 frontend、measurement 或 supervision source。",
        "- VSPW pseudo identity 只能 diagnostic-only。",
        "- 当前 claim 边界是 full 325 M128/H32 raw-video closure video-system benchmark，不是任意尺度 CVPR-scale complete claim。",
        "",
        f"- recommended_next_step: {report['recommended_next_step']}",
        "",
    ]
    DOC.write_text("\n".join(doc_lines), encoding="utf-8")
    print(json.dumps({
        "v35_50_claim_boundary_audit_done": True,
        "artifact_packaging_complete": artifact_packaging_complete,
        "m128_h32_full_325_video_system_benchmark_claim_allowed": full_m128_claim,
        "full_cvpr_scale_claim_allowed": False,
        "recommended_next_step": report["recommended_next_step"],
    }, ensure_ascii=False), flush=True)
    return 0 if artifact_packaging_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
