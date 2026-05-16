#!/usr/bin/env python3
"""V36.8: 打包 V36 causal claim boundary release bundle，不写论文。"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT  # noqa: E402

V35_49_TEACHER_DECISION = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_decision_20260516.json"
V35_49_TEACHER_BENCH = ROOT / "reports/stwm_ostf_v35_49_full_325_raw_video_closure_benchmark_eval_summary_20260516.json"
V36_AUDIT = ROOT / "reports/stwm_ostf_v36_v35_49_causal_trace_contract_audit_20260516.json"
V36_PAST_ONLY_INPUT = ROOT / "reports/stwm_ostf_v36_past_only_observed_trace_input_build_20260516.json"
V36_V30_ROLLOUT = ROOT / "reports/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.json"
V36_SLICE = ROOT / "reports/stwm_ostf_v36_causal_unified_semantic_identity_slice_build_20260516.json"
V36_ORIG_DECISION = ROOT / "reports/stwm_ostf_v36_decision_20260516.json"
V36_1_DECISION = ROOT / "reports/stwm_ostf_v36_1_decision_20260516.json"
V36_2C_SELECTOR = ROOT / "reports/stwm_ostf_v36_2c_conservative_copy_default_selector_rollout_20260516.json"
V36_2C_DOWNSTREAM = ROOT / "reports/stwm_ostf_v36_2c_conservative_selector_downstream_gate_decision_20260516.json"
V36_3_EVAL = ROOT / "reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_eval_summary_20260516.json"
V36_3_DECISION = ROOT / "reports/stwm_ostf_v36_3_full_325_causal_benchmark_rerun_decision_20260516.json"
V36_4_AUDIT = ROOT / "reports/stwm_ostf_v36_4_causal_claim_boundary_and_packaging_audit_20260516.json"
V36_4_CLAIMS = ROOT / "reports/stwm_ostf_v36_4_machine_checkable_claim_table_20260516.json"
V36_5_ATLAS = ROOT / "reports/stwm_ostf_v36_5_causal_failure_atlas_eval_20260516.json"
V36_5_DECISION = ROOT / "reports/stwm_ostf_v36_5_causal_failure_atlas_decision_20260516.json"
V36_5_RISK = ROOT / "reports/stwm_ostf_v36_5_reviewer_risk_audit_20260516.json"
V36_6_AUDIT = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_target_contract_audit_20260516.json"
V36_6_EVAL = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_eval_summary_20260516.json"
V36_6_DECISION = ROOT / "reports/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_decision_20260516.json"
V36_7_DECISION = ROOT / "reports/stwm_ostf_v36_7_claim_boundary_after_occlusion_identity_repair_20260516.json"
V36_7_CLAIMS = ROOT / "reports/stwm_ostf_v36_7_machine_checkable_claim_table_20260516.json"

V36_2C_SLICE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_2c_conservative_selector_downstream_slice/M128_H32"
V36_6_OVERRIDE_ROOT = ROOT / "outputs/cache/stwm_ostf_v36_6_occlusion_reappear_identity_target_overrides/M128_H32"
V36_VIS = ROOT / "reports/stwm_ostf_v36_causal_past_only_world_model_visualization_manifest_20260516.json"
V36_FIG_DIR = ROOT / "reports/figures/stwm_ostf_v36_causal_past_only_world_model_20260516"

LOGS = [
    ROOT / "outputs/logs/stwm_ostf_v36_v30_past_only_future_trace_rollout_20260516.log",
    ROOT / "outputs/logs/stwm_ostf_v36_4_causal_claim_boundary_and_packaging_audit_20260516.log",
    ROOT / "outputs/logs/stwm_ostf_v36_5_causal_failure_atlas_20260516.log",
    ROOT / "outputs/logs/stwm_ostf_v36_6_occlusion_reappear_identity_field_repair_20260516.log",
    ROOT / "outputs/logs/stwm_ostf_v36_7_claim_table_after_occlusion_identity_repair_20260516.log",
]

REPORT = ROOT / "reports/stwm_ostf_v36_8_release_bundle_with_causal_claim_boundary_20260516.json"
FROZEN = ROOT / "reports/stwm_ostf_v36_8_frozen_causal_claim_boundary_manifest_20260516.json"
BUNDLE = ROOT / "reports/stwm_ostf_v36_8_non_paper_release_bundle_index_20260516.json"
DOC = ROOT / "docs/STWM_OSTF_V36_8_RELEASE_BUNDLE_WITH_CAUSAL_CLAIM_BOUNDARY_20260516.md"
LOG = ROOT / "outputs/logs/stwm_ostf_v36_8_release_bundle_with_causal_claim_boundary_20260516.log"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def entry(path: Path, kind: str, required: bool = True) -> dict[str, Any]:
    count = None
    if path.exists() and path.is_dir():
        count = sum(1 for p in path.rglob("*") if p.is_file())
    return {"path": rel(path), "kind": kind, "required": required, "exists": path.exists(), "file_count": count}


def split_counts(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        out[split] = len(list((root / split).glob("*.npz"))) if (root / split).exists() else 0
    out["all"] = sum(out.values())
    return out


def main() -> int:
    v36_3 = load(V36_3_DECISION)
    v36_4 = load(V36_4_AUDIT)
    v36_6 = load(V36_6_DECISION)
    v36_7 = load(V36_7_DECISION)
    claims = load(V36_7_CLAIMS)

    evidence_entries = [
        entry(V35_49_TEACHER_DECISION, "teacher_trace_upper_bound_report"),
        entry(V35_49_TEACHER_BENCH, "teacher_trace_upper_bound_eval"),
        entry(V36_AUDIT, "causal_contract_audit"),
        entry(V36_PAST_ONLY_INPUT, "past_only_input_build"),
        entry(V36_V30_ROLLOUT, "v30_past_only_rollout"),
        entry(V36_SLICE, "causal_unified_slice_build"),
        entry(V36_ORIG_DECISION, "original_causal_decision"),
        entry(V36_1_DECISION, "trace_failure_atlas_and_prior_downstream_decision"),
        entry(V36_2C_SELECTOR, "conservative_selector_rollout"),
        entry(V36_2C_DOWNSTREAM, "conservative_selector_downstream_gate"),
        entry(V36_3_EVAL, "causal_full325_eval_summary"),
        entry(V36_3_DECISION, "causal_full325_decision"),
        entry(V36_4_AUDIT, "causal_claim_boundary_audit"),
        entry(V36_4_CLAIMS, "old_machine_checkable_claim_table"),
        entry(V36_5_ATLAS, "causal_failure_atlas"),
        entry(V36_5_DECISION, "causal_failure_atlas_decision"),
        entry(V36_5_RISK, "reviewer_risk_audit"),
        entry(V36_6_AUDIT, "occlusion_target_contract_audit"),
        entry(V36_6_EVAL, "occlusion_identity_repair_eval"),
        entry(V36_6_DECISION, "occlusion_identity_repair_decision"),
        entry(V36_7_DECISION, "updated_claim_boundary_decision"),
        entry(V36_7_CLAIMS, "updated_machine_checkable_claim_table"),
        entry(V36_2C_SLICE_ROOT, "causal_selector_slice_cache"),
        entry(V36_6_OVERRIDE_ROOT, "occlusion_eval_target_override_cache"),
        entry(V36_VIS, "causal_visualization_manifest"),
        entry(V36_FIG_DIR, "causal_visualization_figures"),
    ] + [entry(p, "log") for p in LOGS]

    missing = [e for e in evidence_entries if e["required"] and not e["exists"]]
    causal_slice_counts = split_counts(V36_2C_SLICE_ROOT)
    occ_override_counts = split_counts(V36_6_OVERRIDE_ROOT)
    release_ready = bool(
        not missing
        and causal_slice_counts.get("all") == 325
        and occ_override_counts.get("all") == 325
        and v36_3.get("m128_h32_causal_video_world_model_benchmark_claim_allowed")
        and v36_6.get("m128_h32_causal_identity_occlusion_reappear_claim_allowed")
        and v36_7.get("m128_h32_causal_video_world_model_benchmark_claim_allowed")
        and not v36_7.get("full_cvpr_scale_claim_allowed", True)
    )
    allowed_claims = [c for c in claims.get("claims", []) if str(c.get("status", "")).startswith("allowed")]
    not_allowed_claims = [c for c in claims.get("claims", []) if c.get("status") == "not_allowed"]

    frozen_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.8",
        "frozen_from_versions": ["V35.49", "V36", "V36.1", "V36.2c", "V36.3", "V36.4", "V36.5", "V36.6", "V36.7"],
        "release_bundle_ready": release_ready,
        "allowed_claims": allowed_claims,
        "not_allowed_claims": not_allowed_claims,
        "hard_boundaries": {
            "m128_h32_only": True,
            "selected_clip_count": v36_3.get("selected_clip_count"),
            "causal_past_only_future_trace_required": True,
            "v35_49_teacher_trace_upper_bound_only": True,
            "h64_h96_not_run": True,
            "m512_m1024_not_run": True,
            "one_b_not_run": True,
            "full_cvpr_scale_claim_allowed": False,
            "open_vocabulary_dense_segmentation_claim_allowed": False,
            "v34_continuous_teacher_delta_route_claim_allowed": False,
            "teacher_as_method_allowed": False,
            "future_teacher_embedding_as_input_allowed": False,
            "future_teacher_trace_as_input_allowed": False,
            "pseudo_identity_claim_allowed": False,
        },
        "positive_claim_sentence_zh": (
            "STWM V36 在 full 325 M128/H32 causal benchmark 上，形成了 past-only observed dense trace → "
            "frozen V30/prior-selector future trace → future semantic state/transition/uncertainty → real-instance pairwise identity retrieval 的闭环。"
        ),
        "occlusion_identity_sentence_zh": (
            "V36.6 修复 occlusion/reappear identity target/eval contract 后，在 teacher-vis-defined eval target 上 real-instance occlusion/reappear 三 seed 通过；"
            "teacher future trace 仅作 evaluation/supervision target，不作为输入。"
        ),
        "forbidden_overclaim_sentence_zh": (
            "当前不能声称 full CVPR-scale complete system、任意分辨率、任意 horizon、H64/H96、M512/M1024、1B 或 full open-vocabulary dense segmentation 已完成。"
        ),
        "entry_points": {
            "updated_claim_table": rel(V36_7_CLAIMS),
            "v36_3_causal_benchmark_decision": rel(V36_3_DECISION),
            "v36_6_occlusion_identity_repair": rel(V36_6_DECISION),
            "v36_4_claim_boundary_audit": rel(V36_4_AUDIT),
            "v36_5_reviewer_risk_audit": rel(V36_5_RISK),
            "v35_49_teacher_trace_upper_bound_audit": rel(V36_AUDIT),
        },
    }
    bundle_index = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.8",
        "bundle_name": "stwm_v36_full325_m128_h32_causal_video_world_model_non_paper_release_bundle",
        "release_bundle_ready": release_ready,
        "not_a_paper_draft": True,
        "evidence_entries": evidence_entries,
        "minimum_files_to_share": [e["path"] for e in evidence_entries if e["required"]],
        "recommended_read_order": [
            rel(V36_7_CLAIMS),
            rel(V36_3_DECISION),
            rel(V36_6_DECISION),
            rel(V36_4_AUDIT),
            rel(V36_5_RISK),
            rel(V36_AUDIT),
            rel(V36_2C_SELECTOR),
        ],
        "notes_zh": "该 bundle 是 V36 因果 benchmark 证据索引，不是论文正文；用于内部复验、外部 sanity review 和 claim boundary 冻结。",
    }
    remaining_to_cvpr = [
        {
            "gap": "scale_generalization",
            "status": "not_done",
            "说明": "尚未跑 H64/H96、M512/M1024 或更长 horizon；当前只允许 M128/H32 full-325。",
        },
        {
            "gap": "open_vocabulary_dense_semantic_segmentation",
            "status": "not_claimed",
            "说明": "当前 semantic 是 state/transition/uncertainty field，不是 full open-vocabulary dense segmentation。",
        },
        {
            "gap": "independent_environment_replay",
            "status": "recommended_before_public_claim",
            "说明": "release bundle 已可复验索引化；公开前最好在独立环境按 bundle 入口重放。",
        },
        {
            "gap": "stronger_video_domain_generalization",
            "status": "future_work",
            "说明": "当前 full325 是 bounded benchmark，不代表任意视频域。",
        },
    ]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_completed_version": "V36.8",
        "v36_release_bundle_built": True,
        "v36_release_bundle_ready": release_ready,
        "frozen_causal_claim_boundary_manifest_path": rel(FROZEN),
        "non_paper_release_bundle_index_path": rel(BUNDLE),
        "selected_clip_count": v36_3.get("selected_clip_count"),
        "causal_slice_counts": causal_slice_counts,
        "occlusion_override_counts": occ_override_counts,
        "artifact_missing_count": len(missing),
        "missing_required_artifacts": missing,
        "m128_h32_causal_video_world_model_benchmark_claim_allowed": bool(
            release_ready and v36_7.get("m128_h32_causal_video_world_model_benchmark_claim_allowed")
        ),
        "m128_h32_causal_identity_occlusion_reappear_claim_allowed": bool(
            release_ready and v36_7.get("m128_h32_causal_identity_occlusion_reappear_claim_allowed")
        ),
        "m128_h32_teacher_trace_upper_bound_claim_allowed": True,
        "full_cvpr_scale_claim_allowed": False,
        "v35_49_is_teacher_trace_upper_bound": True,
        "future_trace_predicted_from_past_only": bool(v36_3.get("future_trace_predicted_from_past_only")),
        "semantic_three_seed_passed": bool(v36_3.get("semantic_three_seed_passed")),
        "stable_preservation": bool(v36_3.get("stable_preservation")),
        "identity_real_instance_three_seed_passed": bool(v36_3.get("identity_real_instance_three_seed_passed")),
        "occlusion_reappear_identity_three_seed_passed": bool(v36_6.get("occlusion_reappear_identity_three_seed_passed")),
        "future_leakage_detected": bool(v36_3.get("future_leakage_detected")),
        "trajectory_degraded": bool(v36_3.get("trajectory_degraded")),
        "remaining_to_full_cvpr_scale": remaining_to_cvpr,
        "recommended_next_step": "independent_environment_replay_or_prepare_result_section_from_v36_bundle",
        "中文总结": (
            "V36.8 已打包 V36.3 causal benchmark、V36.6 occlusion identity repair 和 V36.7 claim table。"
            "当前允许 bounded M128/H32 full-325 causal video world model benchmark claim；仍不允许 full CVPR-scale complete system claim。"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    DOC.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(json.dumps(frozen_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    BUNDLE.write_text(json.dumps(bundle_index, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    allowed_lines = "\n".join(f"- `{c['claim_id']}`: {c['claim_zh']}" for c in allowed_claims)
    blocked_lines = "\n".join(f"- `{c['claim_id']}`: {c['claim_zh']}" for c in not_allowed_claims)
    missing_lines = "\n".join(f"- `{e['path']}`" for e in missing) or "- 无缺失 required artifacts。"
    DOC.write_text(
        "# STWM OSTF V36.8 Release Bundle with Causal Claim Boundary\n\n"
        "## 中文总结\n"
        f"{report['中文总结']}\n\n"
        "## 允许 claim\n"
        f"{allowed_lines}\n\n"
        "## 不允许 claim\n"
        f"{blocked_lines}\n\n"
        "## Release bundle 状态\n"
        f"- v36_release_bundle_ready: {release_ready}\n"
        f"- selected_clip_count: {report['selected_clip_count']}\n"
        f"- causal_slice_counts: {causal_slice_counts}\n"
        f"- occlusion_override_counts: {occ_override_counts}\n"
        f"- artifact_missing_count: {len(missing)}\n"
        f"{missing_lines}\n\n"
        "## Claim 边界\n"
        "- V35.49 只能作为 teacher-trace upper-bound。\n"
        "- V36.3 是 causal past-only M128/H32 full-325 benchmark。\n"
        "- V36.6 的 occlusion/reappear target 来自 teacher future visibility，仅作为 eval/supervision target，不作为输入。\n"
        "- full_cvpr_scale_claim_allowed: false\n\n"
        "## 输出\n"
        f"- frozen_causal_claim_boundary_manifest: `{rel(FROZEN)}`\n"
        f"- non_paper_release_bundle_index: `{rel(BUNDLE)}`\n"
        f"- release_report: `{rel(REPORT)}`\n"
        f"- log: `{rel(LOG)}`\n"
        f"- recommended_next_step: `{report['recommended_next_step']}`\n",
        encoding="utf-8",
    )
    LOG.write_text(
        "\n".join(
            [
                f"[{datetime.now(timezone.utc).isoformat()}] V36.8 release bundle 生成完成。",
                f"release_bundle_ready={release_ready}",
                f"artifact_missing_count={len(missing)}",
                f"m128_h32_causal_video_world_model_benchmark_claim_allowed={report['m128_h32_causal_video_world_model_benchmark_claim_allowed']}",
                "full_cvpr_scale_claim_allowed=False",
                f"recommended_next_step={report['recommended_next_step']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "中文状态": "V36.8 release bundle 已生成",
                "v36_release_bundle_ready": release_ready,
                "artifact_missing_count": len(missing),
                "m128_h32_causal_video_world_model_benchmark_claim_allowed": report[
                    "m128_h32_causal_video_world_model_benchmark_claim_allowed"
                ],
                "full_cvpr_scale_claim_allowed": False,
                "recommended_next_step": report["recommended_next_step"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if release_ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
