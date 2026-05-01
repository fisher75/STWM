#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/raid/chen034/workspace/stwm")
REPORT_DIR = REPO_ROOT / "reports"
DOC_DIR = REPO_ROOT / "docs"
LOG_DIR = REPO_ROOT / "outputs/logs"
CKPT_DIR = REPO_ROOT / "outputs/checkpoints/stwm_mixed_fullscale_v2_20260428"
FIG_DIR = REPO_ROOT / "outputs/figures/stwm_final"
VIDEO_DIR = REPO_ROOT / "outputs/videos/stwm_final"


def load_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: str | Path, title: str, sections: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + "\n\n".join(sections).rstrip() + "\n", encoding="utf-8")


def metric_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    m = payload.get("best_metrics", {})
    return {
        "dataset": name,
        "test_items": int(payload.get("heldout_item_count", 0)),
        "copy_top1": float(m.get("copy_proto_top1", 0.0)),
        "copy_top5": float(m.get("copy_proto_top5", 0.0)),
        "copy_ce": float(m.get("copy_proto_ce", 0.0)),
        "stwm_top1": float(m.get("proto_top1", 0.0)),
        "stwm_top5": float(m.get("proto_top5", 0.0)),
        "stwm_ce": float(m.get("proto_ce", 0.0)),
        "overall_gain": float(m.get("overall_gain_over_copy", 0.0)),
        "copy_changed_top5": float(m.get("copy_changed_subset_top5", 0.0)),
        "stwm_changed_top5": float(m.get("changed_subset_top5", 0.0)),
        "changed_gain": float(m.get("changed_subset_gain_over_copy", 0.0)),
        "copy_stable_top5": float(m.get("copy_stable_subset_top5", 0.0)),
        "stwm_stable_top5": float(m.get("stable_subset_top5", 0.0)),
        "stable_drop": float(m.get("stable_preservation_drop", 0.0)),
        "trace_coord_error": float(m.get("future_trace_coord_error", 0.0)),
        "change_ap": float(m.get("change_detection", {}).get("ap", 0.0)),
        "change_auroc": float(m.get("change_detection", {}).get("auroc", 0.0)),
    }


def ci_row(name: str, significance: dict[str, Any]) -> dict[str, Any]:
    key = name.lower()
    block = significance.get(key, {})
    metric = block.get("residual_vs_copy_changed_top5", {})
    return {
        "dataset": name,
        "changed_item_count": int(block.get("changed_item_count", 0)),
        "mean_delta": float(metric.get("mean_delta", 0.0)),
        "ci95": metric.get("ci95", [0.0, 0.0]),
        "zero_excluded": bool(metric.get("zero_excluded", False)),
        "bootstrap_win_rate": float(metric.get("bootstrap_win_rate", 0.0)),
    }


def build_artifact_protocol_audit() -> dict[str, Any]:
    artifact = load_json(REPORT_DIR / "stwm_top_tier_hardening_v1_artifact_audit_20260428.json")
    protocol = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_protocol_audit_20260428.json")
    selection = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_val_selection_complete_20260428.json")
    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")
    vspw = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vspw_test_eval_complete_20260428.json")
    vipseg = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vipseg_test_eval_complete_20260428.json")

    audit = {
        "audit_name": "stwm_top_tier_evidence_v1_artifact_protocol_audit",
        "all_10_runs_completed": bool(artifact.get("existing_checkpoint_count", 0) == 10),
        "checkpoint_count": int(artifact.get("existing_checkpoint_count", 0)),
        "log_count": int(artifact.get("existing_log_count", 0)),
        "zero_byte_training_log_count": int(artifact.get("zero_byte_training_log_count", 0)),
        "checkpoints_exist": bool(artifact.get("existing_checkpoint_count", 0) == 10),
        "logs_exist": bool(artifact.get("existing_log_count", 0) == 10),
        "val_selection_used_val_only": bool(selection.get("best_selected_on_val_only", False)),
        "test_eval_once": bool(mixed.get("test_eval_once", False)),
        "no_candidate_scorer": bool(not mixed.get("candidate_scorer_used", True)),
        "no_future_leakage": bool(not mixed.get("future_candidate_leakage", True)),
        "free_rollout_path": bool(mixed.get("free_rollout_path", False)),
        "teacher_forced_path_used": bool(mixed.get("teacher_forced_path_used", True)),
        "mixed_item_count": int(mixed.get("heldout_item_count", 0)),
        "vspw_item_count": int(vspw.get("heldout_item_count", 0)),
        "vipseg_item_count": int(vipseg.get("heldout_item_count", 0)),
        "paper_grade_protocol_status": (
            "strong_main_result_protocol_with_test_once_and_val_only_selection; further top-tier evidence pack items still pending"
        ),
        "artifact_audit_source": str(REPORT_DIR / "stwm_top_tier_hardening_v1_artifact_audit_20260428.json"),
        "protocol_audit_source": str(REPORT_DIR / "stwm_mixed_fullscale_v2_protocol_audit_20260428.json"),
    }
    write_json(REPORT_DIR / "stwm_top_tier_evidence_v1_artifact_protocol_audit_20260428.json", audit)
    write_md(
        DOC_DIR / "STWM_TOP_TIER_EVIDENCE_V1_ARTIFACT_PROTOCOL_AUDIT_20260428.md",
        "STWM Top-Tier Evidence V1 Artifact Protocol Audit",
        [
            "## Summary\n" + "\n".join(
                [
                    f"- all_10_runs_completed: `{audit['all_10_runs_completed']}`",
                    f"- checkpoint_count: `{audit['checkpoint_count']}`",
                    f"- log_count: `{audit['log_count']}`",
                    f"- zero_byte_training_log_count: `{audit['zero_byte_training_log_count']}`",
                    f"- val_selection_used_val_only: `{audit['val_selection_used_val_only']}`",
                    f"- test_eval_once: `{audit['test_eval_once']}`",
                    f"- free_rollout_path: `{audit['free_rollout_path']}`",
                ]
            ),
            "## Counts\n" + "\n".join(
                [
                    f"- mixed_item_count: `{audit['mixed_item_count']}`",
                    f"- vspw_item_count: `{audit['vspw_item_count']}`",
                    f"- vipseg_item_count: `{audit['vipseg_item_count']}`",
                ]
            ),
        ],
    )
    return audit


def build_lodo_reports() -> dict[str, Any]:
    train = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_train_summary_20260428.json")
    a = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vspw_to_vipseg_eval_20260428.json")
    b = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vipseg_to_vspw_eval_20260428.json")
    s = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_significance_20260428.json")
    return {
        "completed": bool(train.get("lodo_completed", False)),
        "signal_positive": False,
        "train_summary_path": str(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_train_summary_20260428.json"),
        "vspw_to_vipseg_eval_path": str(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vspw_to_vipseg_eval_20260428.json"),
        "vipseg_to_vspw_eval_path": str(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vipseg_to_vspw_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_significance_20260428.json"),
        "skipped_reason": train.get("skipped_reason", a.get("skipped_reason", b.get("skipped_reason", s.get("skipped_reason", "")))),
    }


def build_model_scaling_reports() -> dict[str, Any]:
    mixed_decision = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    report = {
        "audit_name": "stwm_scaling_model_size_v1_train_summary",
        "model_scaling_completed": False,
        "available_scale": {
            "name": "small",
            "source": "current mixed fullscale semantic branch",
            "best_prototype_count": int(mixed_decision.get("best_prototype_count", 0)),
            "best_seed": int(mixed_decision.get("best_seed", -1)),
        },
        "missing_scales": ["medium_10_to_15M", "large_30_to_50M"],
        "skipped_reason": "Only the current semantic branch scale has been trained/evaluated; no medium/large controlled Stage2 semantic-branch scale runs exist in the live repo.",
    }
    eval_report = {
        "audit_name": "stwm_scaling_model_size_v1_eval",
        "model_scaling_completed": False,
        "model_scaling_trend_positive": False,
        "skipped_reason": report["skipped_reason"],
    }
    sig = {
        "audit_name": "stwm_scaling_model_size_v1_significance",
        "model_scaling_completed": False,
        "skipped_reason": report["skipped_reason"],
    }
    write_json(REPORT_DIR / "stwm_scaling_model_size_v1_train_summary_20260428.json", report)
    write_json(REPORT_DIR / "stwm_scaling_model_size_v1_eval_20260428.json", eval_report)
    write_json(REPORT_DIR / "stwm_scaling_model_size_v1_significance_20260428.json", sig)
    write_md(
        DOC_DIR / "STWM_SCALING_MODEL_SIZE_V1_20260428.md",
        "STWM Scaling Model Size V1",
        [
            "## Status\n- model_scaling_completed: `False`\n- available_scale: `small only`\n- skipped_reason: `Only the current semantic branch scale is present in live repo artifacts.`"
        ],
    )
    return {
        "completed": False,
        "trend_positive": False,
        "train_summary_path": str(REPORT_DIR / "stwm_scaling_model_size_v1_train_summary_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_scaling_model_size_v1_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_scaling_model_size_v1_significance_20260428.json"),
    }


def build_horizon_scaling_reports() -> dict[str, Any]:
    audit = load_json(REPORT_DIR / "stwm_horizon_scaling_h16_v1_target_audit_20260428.json")
    train = load_json(REPORT_DIR / "stwm_horizon_scaling_h16_v1_train_summary_20260428.json")
    eval_report = load_json(REPORT_DIR / "stwm_horizon_scaling_h16_v1_eval_20260428.json")
    wrapper = {
        "audit_name": "stwm_scaling_horizon_v1_target_audit",
        "available_horizons": ["H8"],
        "target_cache_support_status": audit,
        "train_summary_status": train,
        "eval_status": eval_report,
        "horizon_scaling_completed": False,
        "horizon_scaling_positive": False,
        "skipped_reason": audit.get("h16_blocker", eval_report.get("skipped_reason", "")),
    }
    write_json(REPORT_DIR / "stwm_scaling_horizon_v1_target_audit_20260428.json", wrapper)
    write_json(
        REPORT_DIR / "stwm_scaling_horizon_v1_train_summary_20260428.json",
        {
            "audit_name": "stwm_scaling_horizon_v1_train_summary",
            "horizon_scaling_completed": False,
            "skipped_reason": wrapper["skipped_reason"],
        },
    )
    write_json(
        REPORT_DIR / "stwm_scaling_horizon_v1_eval_20260428.json",
        {
            "audit_name": "stwm_scaling_horizon_v1_eval",
            "horizon_scaling_completed": False,
            "horizon_scaling_positive": False,
            "available_horizons": ["H8"],
            "skipped_reason": wrapper["skipped_reason"],
        },
    )
    write_md(
        DOC_DIR / "STWM_SCALING_HORIZON_V1_20260428.md",
        "STWM Scaling Horizon V1",
        [
            "## Status\n- horizon_scaling_completed: `False`\n- available_horizons: `H8 only`\n- skipped_reason: `H16/H32 require rebuilt target pools/materialization and retraining.`"
        ],
    )
    return {
        "completed": False,
        "positive": False,
        "target_audit_path": str(REPORT_DIR / "stwm_scaling_horizon_v1_target_audit_20260428.json"),
        "train_summary_path": str(REPORT_DIR / "stwm_scaling_horizon_v1_train_summary_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_scaling_horizon_v1_eval_20260428.json"),
    }


def build_trace_density_reports() -> dict[str, Any]:
    audit = load_json(REPORT_DIR / "stwm_trace_unit_density_scaling_v1_audit_20260428.json")
    eval_report = load_json(REPORT_DIR / "stwm_trace_unit_density_scaling_v1_eval_20260428.json")
    wrapper_audit = {
        "audit_name": "stwm_scaling_trace_density_v1_audit",
        "available_k": [8],
        "k16_supported_by_cli": bool(audit.get("k16_supported_by_cli", False)),
        "k32_supported_by_cli": bool(audit.get("k32_supported_by_cli", False)),
        "trace_density_scaling_completed": False,
        "terminology_recommendation": audit.get("terminology_recommendation", "semantic trace-unit field"),
        "memory_compute_blocker": audit.get("memory_compute_blocker", ""),
    }
    wrapper_eval = {
        "audit_name": "stwm_scaling_trace_density_v1_eval",
        "trace_density_scaling_completed": False,
        "available_k": [8],
        "terminology_recommendation": audit.get("terminology_recommendation", "semantic trace-unit field"),
        "skipped_reason": eval_report.get("skipped_reason", audit.get("memory_compute_blocker", "")),
    }
    write_json(REPORT_DIR / "stwm_scaling_trace_density_v1_audit_20260428.json", wrapper_audit)
    write_json(REPORT_DIR / "stwm_scaling_trace_density_v1_eval_20260428.json", wrapper_eval)
    write_md(
        DOC_DIR / "STWM_SCALING_TRACE_DENSITY_V1_20260428.md",
        "STWM Scaling Trace Density V1",
        [
            "## Status\n- trace_density_scaling_completed: `False`\n- available_k: `K=8 only`\n- terminology_recommendation: `semantic trace-unit field`"
        ],
    )
    return {
        "completed": False,
        "audit_path": str(REPORT_DIR / "stwm_scaling_trace_density_v1_audit_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_scaling_trace_density_v1_eval_20260428.json"),
        "terminology_recommendation": wrapper_audit["terminology_recommendation"],
    }


def build_baseline_suite() -> dict[str, Any]:
    predictability = load_json(REPORT_DIR / "stwm_semantic_prototype_predictability_baselines_v1_20260428.json")
    persistence = load_json(REPORT_DIR / "stwm_semantic_proto_persistence_baseline_v2_20260428.json")
    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")
    suite = {
        "audit_name": "stwm_world_model_baseline_suite_v1_eval",
        "baseline_suite_completed": False,
        "same_output_protocol_required": True,
        "available_baselines": [
            {
                "name": "frequency",
                "source": "semantic prototype predictability / persistence reports",
                "same_free_rollout_protocol": False,
            },
            {
                "name": "semantic_copy",
                "source": "mixed fullscale eval copy baseline",
                "same_free_rollout_protocol": True,
                "mixed_copy_top5": float(mixed.get("copy_baseline", {}).get("copy_proto_top5", 0.0)),
            },
            {
                "name": "trace_only_mlp",
                "source": "semantic prototype predictability baselines",
                "same_free_rollout_protocol": False,
            },
            {
                "name": "semantic_only_mlp",
                "source": "semantic prototype predictability baselines",
                "same_free_rollout_protocol": False,
            },
            {
                "name": "semantic_plus_trace_simple_transformer",
                "source": "not present in live repo",
                "same_free_rollout_protocol": False,
            },
            {
                "name": "slotformer_like_object_slot_dynamics",
                "source": "not present in live repo",
                "same_free_rollout_protocol": False,
            },
            {
                "name": "dino_wm_like_feature_latent_dynamics",
                "source": "not present in live repo",
                "same_free_rollout_protocol": False,
            },
        ],
        "predictability_summary": {
            "simple_probe_beats_frequency": predictability.get("simple_probe_beats_frequency"),
            "target_predictable_from_observed_semantics": predictability.get("target_predictable_from_observed_semantics"),
            "semantic_input_load_bearing": predictability.get("semantic_input_load_bearing"),
        },
        "copy_baseline_summary": {
            "selected_prototype_count": persistence.get("selected_prototype_count"),
            "copy_baseline_top5": persistence.get("copy_baseline_top5"),
            "copy_baseline_strong": persistence.get("copy_baseline_strong"),
        },
        "STWM_beats_world_model_baselines": False,
        "skipped_reason": "A fair same-output free-rollout world-model baseline suite is not fully implemented in the live repo; only trivial/probe baselines and the copy baseline are currently available.",
    }
    sig = {
        "audit_name": "stwm_world_model_baseline_suite_v1_significance",
        "baseline_suite_completed": False,
        "skipped_reason": suite["skipped_reason"],
    }
    write_json(REPORT_DIR / "stwm_world_model_baseline_suite_v1_eval_20260428.json", suite)
    write_json(REPORT_DIR / "stwm_world_model_baseline_suite_v1_significance_20260428.json", sig)
    write_md(
        DOC_DIR / "STWM_WORLD_MODEL_BASELINE_SUITE_V1_20260428.md",
        "STWM World Model Baseline Suite V1",
        [
            "## Status\n- baseline_suite_completed: `False`\n- available comparable baseline: `semantic copy baseline`\n- skipped_reason: `Fair same-output free-rollout world-model baselines beyond copy are not implemented yet.`"
        ],
    )
    return {
        "completed": False,
        "stwm_beats_world_model_baselines": False,
        "eval_path": str(REPORT_DIR / "stwm_world_model_baseline_suite_v1_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_world_model_baseline_suite_v1_significance_20260428.json"),
    }


def build_benchmark_protocol() -> dict[str, Any]:
    protocol = {
        "audit_name": "stwm_fstf_benchmark_protocol_v1",
        "benchmark_name": "STWM-FSTF: Future Semantic Trace Field Prediction",
        "input": [
            "observed video-derived trace",
            "observed semantic memory",
        ],
        "output": [
            "future trace field / trace units",
            "future semantic prototype field",
            "future visibility / reappearance",
            "future identity belief",
        ],
        "metrics": [
            "semantic top1/top5/CE",
            "changed gain over copy",
            "stable preservation",
            "trace coord error",
            "visibility/reappearance AP/AUROC",
            "free-rollout gap",
        ],
        "why_changed_subset_matters": "Observed semantic memory makes copy a strong baseline on stable states, so changed cases isolate true semantic world-model transition ability.",
        "why_copy_baseline_is_strong": "Observed semantics are slow variables; copying observed prototype already scores highly on stable states.",
        "why_residual_improvement_is_meaningful": "Positive changed-subset gain over copy shows the model predicts semantic transitions rather than merely memorizing observed identity.",
        "why_this_is_world_model_output_not_plugin": "Rollout input stays limited to observed trace/video-derived memory; output is future trace+semantic field rather than post-hoc tracker refinement.",
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    write_json(REPORT_DIR / "stwm_fstf_benchmark_protocol_v1_20260428.json", protocol)
    write_md(
        DOC_DIR / "STWM_FSTF_BENCHMARK_PROTOCOL_V1_20260428.md",
        "STWM-FSTF Benchmark Protocol V1",
        [
            "## Benchmark\n- Name: `STWM-FSTF: Future Semantic Trace Field Prediction`",
            "## Why Changed Matters\n- " + protocol["why_changed_subset_matters"],
            "## Why Copy Baseline Matters\n- " + protocol["why_copy_baseline_is_strong"],
            "## World-Model Framing\n- " + protocol["why_this_is_world_model_output_not_plugin"],
        ],
    )
    return protocol


def build_video_manifest() -> dict[str, Any]:
    existing_videos = sorted(str(p) for p in VIDEO_DIR.glob("*")) if VIDEO_DIR.exists() else []
    manifest = {
        "audit_name": "stwm_final_video_visualization_manifest",
        "video_dir": str(VIDEO_DIR),
        "video_visualization_ready": bool(len(existing_videos) > 0),
        "existing_video_files": existing_videos,
        "required_videos": [
            "stable copy preserved",
            "changed semantic corrected",
            "occlusion/reappearance",
            "VSPW success",
            "VIPSeg success",
            "failure case",
            "trace field + semantic field overlay",
        ],
        "required_panels": [
            "observed input frames",
            "predicted future trace",
            "predicted semantic prototype color",
            "copy baseline",
            "STWM residual output",
            "ground truth future semantic target",
        ],
        "skipped_reason": "" if existing_videos else "No final MP4/animated video assets are present in outputs/videos/stwm_final yet; current repo only contains static SVG figures and a mixed visualization manifest.",
    }
    write_json(REPORT_DIR / "stwm_final_video_visualization_manifest_20260428.json", manifest)
    write_md(
        DOC_DIR / "STWM_FINAL_VIDEO_VISUALIZATION_20260428.md",
        "STWM Final Video Visualization 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- video_visualization_ready: `{manifest['video_visualization_ready']}`",
                    f"- existing_video_file_count: `{len(existing_videos)}`",
                    f"- skipped_reason: `{manifest['skipped_reason']}`",
                ]
            )
        ],
    )
    return manifest


def build_table_and_figure_packs() -> dict[str, Any]:
    tables = load_json(REPORT_DIR / "stwm_final_paper_tables_v1_20260428.json")
    lodo = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_train_summary_20260428.json")
    model_scaling = load_json(REPORT_DIR / "stwm_scaling_model_size_v1_train_summary_20260428.json")
    horizon = load_json(REPORT_DIR / "stwm_scaling_horizon_v1_target_audit_20260428.json")
    density = load_json(REPORT_DIR / "stwm_scaling_trace_density_v1_audit_20260428.json")
    baseline = load_json(REPORT_DIR / "stwm_world_model_baseline_suite_v1_eval_20260428.json")
    video = load_json(REPORT_DIR / "stwm_final_video_visualization_manifest_20260428.json")
    figures = load_json(REPORT_DIR / "stwm_final_figure_manifest_v1_20260428.json")

    table_pack = {
        "audit_name": "stwm_final_cvpr_aaai_table_pack_v1",
        "tables": {
            "main_fstf_result": "reports/stwm_final_paper_tables_v1_20260428.json#semantic_trace_field_main_result",
            "mixed_vspw_vipseg_breakdown": "reports/stwm_final_paper_tables_v1_20260428.json#dataset_breakdown",
            "stable_changed_breakdown": "reports/stwm_final_paper_tables_v1_20260428.json#stable_changed_breakdown",
            "lodo": "reports/stwm_lodo_semantic_trace_world_model_v1_train_summary_20260428.json",
            "scaling_model_size": "reports/stwm_scaling_model_size_v1_eval_20260428.json",
            "scaling_horizon": "reports/stwm_scaling_horizon_v1_eval_20260428.json",
            "scaling_trace_density": "reports/stwm_scaling_trace_density_v1_eval_20260428.json",
            "baseline_suite": "reports/stwm_world_model_baseline_suite_v1_eval_20260428.json",
            "trace_guardrail": "reports/stwm_final_paper_tables_v1_20260428.json#trace_guardrail",
        },
        "ready_tables": [
            "main_fstf_result",
            "mixed_vspw_vipseg_breakdown",
            "stable_changed_breakdown",
            "trace_guardrail",
        ],
        "appendix_or_missing_tables": [
            "lodo",
            "scaling_model_size",
            "scaling_horizon",
            "scaling_trace_density",
            "baseline_suite",
        ],
    }
    figure_pack = {
        "audit_name": "stwm_final_cvpr_aaai_figure_pack_v1",
        "figures": {
            "method_overview": str(FIG_DIR / "figure_method_semantic_trace_world_model.svg"),
            "changed_subset_improvement": str(FIG_DIR / "figure_changed_subset_top5.svg"),
            "dataset_breakdown": str(FIG_DIR / "figure_dataset_top5_breakdown.svg"),
            "semantic_trace_field_output": "planned from mixed visualization manifest",
            "scaling_curves": "missing until scaling reports are completed",
            "qualitative_video_frame_strips": "missing until final video render exists",
        },
        "existing_static_figures": [str(p) for p in FIG_DIR.glob("*.svg")],
        "video_visualization_ready": bool(video.get("video_visualization_ready", False)),
        "base_manifest": figures,
    }
    write_json(REPORT_DIR / "stwm_final_cvpr_aaai_table_pack_v1_20260428.json", table_pack)
    write_json(REPORT_DIR / "stwm_final_cvpr_aaai_figure_pack_v1_20260428.json", figure_pack)
    write_md(
        DOC_DIR / "STWM_FINAL_CVPR_AAAI_TABLE_PACK_V1_20260428.md",
        "STWM Final CVPR AAAI Table Pack V1",
        [
            "## Ready Tables\n" + "\n".join(f"- {x}" for x in table_pack["ready_tables"]),
            "## Appendix Or Missing\n" + "\n".join(f"- {x}" for x in table_pack["appendix_or_missing_tables"]),
        ],
    )
    write_md(
        DOC_DIR / "STWM_FINAL_CVPR_AAAI_FIGURE_PACK_V1_20260428.md",
        "STWM Final CVPR AAAI Figure Pack V1",
        [
            "## Existing Static Figures\n" + "\n".join(f"- {x}" for x in figure_pack["existing_static_figures"]),
            "## Missing\n- scaling_curves\n- qualitative_video_frame_strips",
        ],
    )
    return {
        "table_pack_path": str(REPORT_DIR / "stwm_final_cvpr_aaai_table_pack_v1_20260428.json"),
        "figure_pack_path": str(REPORT_DIR / "stwm_final_cvpr_aaai_figure_pack_v1_20260428.json"),
    }


def build_final_decision(
    lodo: dict[str, Any],
    model_scaling: dict[str, Any],
    horizon: dict[str, Any],
    density: dict[str, Any],
    baseline: dict[str, Any],
    video: dict[str, Any],
) -> dict[str, Any]:
    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    decision = {
        "audit_name": "stwm_top_tier_evidence_v1_decision",
        "lodo_completed": bool(lodo["completed"]),
        "lodo_signal_positive": bool(lodo["signal_positive"]),
        "model_scaling_completed": bool(model_scaling["completed"]),
        "model_scaling_trend_positive": bool(model_scaling["trend_positive"]),
        "horizon_scaling_completed": bool(horizon["completed"]),
        "horizon_scaling_positive": bool(horizon["positive"]),
        "trace_density_scaling_completed": bool(density["completed"]),
        "baseline_suite_completed": bool(baseline["completed"]),
        "STWM_beats_world_model_baselines": bool(baseline["stwm_beats_world_model_baselines"]),
        "video_visualization_ready": bool(video.get("video_visualization_ready", False)),
        "ready_for_cvpr_aaai_main": "unclear",
        "ready_for_overleaf": True,
        "remaining_risks": [
            "Dedicated LODO cross-dataset checkpoints/evals are not completed.",
            "No controlled semantic-branch model-size scaling curve exists yet.",
            "Horizon scaling beyond H=8 is not established.",
            "Trace-density scaling beyond K=8 is not established.",
            "Fair same-output free-rollout world-model baseline suite is incomplete.",
            "Final MP4/video visualization assets are not rendered yet.",
        ],
        "recommended_next_step_choice": "run_missing_lodo",
        "core_main_result_still_positive": {
            "residual_beats_copy_mixed": bool(mixed.get("residual_beats_copy_mixed", False)),
            "residual_beats_copy_vspw": bool(mixed.get("residual_beats_copy_vspw", False)),
            "residual_beats_copy_vipseg": bool(mixed.get("residual_beats_copy_vipseg", False)),
            "changed_gain_CI_excludes_zero_mixed": bool(mixed.get("changed_gain_CI_excludes_zero_mixed", False)),
            "changed_gain_CI_excludes_zero_vspw": bool(mixed.get("changed_gain_CI_excludes_zero_vspw", False)),
            "changed_gain_CI_excludes_zero_vipseg": bool(mixed.get("changed_gain_CI_excludes_zero_vipseg", False)),
            "stable_copy_preserved": bool(mixed.get("stable_copy_preserved", False)),
            "trace_regression_detected": bool(mixed.get("trace_regression_detected", True)),
            "semantic_field_branch_status": mixed.get("semantic_field_branch_status", "unknown"),
        },
    }
    write_json(REPORT_DIR / "stwm_top_tier_evidence_v1_decision_20260428.json", decision)
    write_md(
        DOC_DIR / "STWM_TOP_TIER_EVIDENCE_V1_DECISION_20260428.md",
        "STWM Top-Tier Evidence V1 Decision",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- lodo_completed: `{decision['lodo_completed']}`",
                    f"- model_scaling_completed: `{decision['model_scaling_completed']}`",
                    f"- horizon_scaling_completed: `{decision['horizon_scaling_completed']}`",
                    f"- trace_density_scaling_completed: `{decision['trace_density_scaling_completed']}`",
                    f"- baseline_suite_completed: `{decision['baseline_suite_completed']}`",
                    f"- video_visualization_ready: `{decision['video_visualization_ready']}`",
                    f"- ready_for_cvpr_aaai_main: `{decision['ready_for_cvpr_aaai_main']}`",
                    f"- ready_for_overleaf: `{decision['ready_for_overleaf']}`",
                    f"- recommended_next_step_choice: `{decision['recommended_next_step_choice']}`",
                ]
            ),
            "## Remaining Risks\n" + "\n".join(f"- {x}" for x in decision["remaining_risks"]),
        ],
    )
    return decision


def build_guardrail() -> dict[str, Any]:
    guardrail = {
        "audit_name": "stwm_world_model_no_drift_guardrail_v40",
        "allowed": [
            "LODO",
            "scaling law",
            "baseline suite",
            "benchmark framing",
            "video visualization",
            "paper assets",
        ],
        "forbidden": [
            "new method branch before evidence pack complete",
            "candidate scorer",
            "SAM2/CoTracker plugin",
            "future candidate leakage",
            "hiding copy baseline",
            "claiming dense trace field if only trace-unit field is evaluated",
            "claiming full RGB generation or closed-loop planning",
        ],
    }
    write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v40_20260428.json", guardrail)
    write_md(
        DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V40.md",
        "STWM World Model No-Drift Guardrail V40",
        [
            "## Allowed\n" + "\n".join(f"- {x}" for x in guardrail["allowed"]),
            "## Forbidden\n" + "\n".join(f"- {x}" for x in guardrail["forbidden"]),
        ],
    )
    return guardrail


def main() -> None:
    build_artifact_protocol_audit()
    lodo = build_lodo_reports()
    model_scaling = build_model_scaling_reports()
    horizon = build_horizon_scaling_reports()
    density = build_trace_density_reports()
    baseline = build_baseline_suite()
    build_benchmark_protocol()
    video = build_video_manifest()
    build_table_and_figure_packs()
    build_final_decision(lodo, model_scaling, horizon, density, baseline, video)
    build_guardrail()


if __name__ == "__main__":
    main()
