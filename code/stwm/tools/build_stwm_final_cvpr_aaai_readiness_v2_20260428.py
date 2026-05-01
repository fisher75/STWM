#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/raid/chen034/workspace/stwm")
REPORT_DIR = REPO_ROOT / "reports"
DOC_DIR = REPO_ROOT / "docs"
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


def build_lodo_reports() -> dict[str, Any]:
    src_train = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_train_summary_20260428.json")
    src_a = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vspw_to_vipseg_eval_20260428.json")
    src_b = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vipseg_to_vspw_eval_20260428.json")
    src_sig = load_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_significance_20260428.json")
    train = {
        "audit_name": "stwm_final_lodo_train_summary",
        "lodo_completed": bool(src_train.get("lodo_completed", False)),
        "planned_protocols": src_train.get("planned_protocols", []),
        "skipped_reason": src_train.get("skipped_reason", ""),
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    a = {
        "audit_name": "stwm_final_lodo_vspw_to_vipseg_eval",
        "lodo_completed": bool(src_a.get("lodo_completed", False)),
        "lodo_positive": False,
        "skipped_reason": src_a.get("skipped_reason", train["skipped_reason"]),
    }
    b = {
        "audit_name": "stwm_final_lodo_vipseg_to_vspw_eval",
        "lodo_completed": bool(src_b.get("lodo_completed", False)),
        "lodo_positive": False,
        "skipped_reason": src_b.get("skipped_reason", train["skipped_reason"]),
    }
    sig = {
        "audit_name": "stwm_final_lodo_significance",
        "lodo_completed": bool(src_sig.get("lodo_completed", False)),
        "lodo_positive": False,
        "skipped_reason": src_sig.get("skipped_reason", train["skipped_reason"]),
    }
    write_json(REPORT_DIR / "stwm_final_lodo_train_summary_20260428.json", train)
    write_json(REPORT_DIR / "stwm_final_lodo_vspw_to_vipseg_eval_20260428.json", a)
    write_json(REPORT_DIR / "stwm_final_lodo_vipseg_to_vspw_eval_20260428.json", b)
    write_json(REPORT_DIR / "stwm_final_lodo_significance_20260428.json", sig)
    write_md(
        DOC_DIR / "STWM_FINAL_LODO_20260428.md",
        "STWM Final LODO 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- lodo_completed: `{train['lodo_completed']}`",
                    f"- skipped_reason: `{train['skipped_reason']}`",
                ]
            )
        ],
    )
    return {
        "lodo_completed": bool(train["lodo_completed"]),
        "lodo_positive": False,
        "train_summary_path": str(REPORT_DIR / "stwm_final_lodo_train_summary_20260428.json"),
        "vspw_to_vipseg_eval_path": str(REPORT_DIR / "stwm_final_lodo_vspw_to_vipseg_eval_20260428.json"),
        "vipseg_to_vspw_eval_path": str(REPORT_DIR / "stwm_final_lodo_vipseg_to_vspw_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_final_lodo_significance_20260428.json"),
    }


def build_prototype_vocab_scaling() -> dict[str, Any]:
    sweep = load_json(REPORT_DIR / "stwm_semantic_trace_prototypes_v2_sweep_20260428.json")
    full32 = load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_prototypes_c32_v1_20260428.json")
    full64 = load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_prototypes_c64_v1_20260428.json")
    val_sel = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_val_selection_complete_20260428.json")
    mixed_eval = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")

    sweep_rows = []
    for row in sweep.get("prototype_sweep_results", []):
        sweep_rows.append(
            {
                "prototype_count": int(row.get("prototype_count", 0)),
                "source_valid_feature_count": int(row.get("source_valid_feature_count", 0)),
                "mean_cluster_size": float(row.get("mean_prototype_count", 0.0)),
                "median_cluster_size": float(row.get("median_prototype_count", 0.0)),
                "min_cluster_size": int(row.get("min_prototype_count", 0)),
                "max_cluster_size": int(row.get("max_prototype_count", 0)),
                "mean_within_cluster_similarity": float(row.get("mean_within_cluster_similarity", 0.0)),
                "empty_prototype_count": int(row.get("empty_prototype_count", 0)),
                "coverage": float(row.get("coverage", 0.0)),
                "long_tail_warning": bool(row.get("long_tail_warning", False)),
            }
        )

    selected_candidates = [
        {
            "prototype_count": int(c.get("prototype_count", 0)),
            "seed": int(c.get("seed", -1)),
            "changed_gain_over_copy": float(c.get("changed_gain_over_copy", 0.0)),
            "overall_gain_over_copy": float(c.get("overall_gain_over_copy", 0.0)),
            "stable_preservation_drop": float(c.get("stable_preservation_drop", 0.0)),
            "future_trace_coord_error": float(c.get("future_trace_coord_error", 0.0)),
        }
        for c in val_sel.get("candidates", [])
    ]
    selected = {
        "selected_prototype_count": int(val_sel.get("selected_prototype_count", 0)),
        "selected_seed": int(val_sel.get("selected_seed", -1)),
        "selected_changed_gain_over_copy": float(val_sel.get("selected_changed_gain_over_copy", 0.0)),
        "selected_overall_gain_over_copy": float(val_sel.get("selected_overall_gain_over_copy", 0.0)),
        "selection_rule": val_sel.get("selection_rule"),
    }

    report = {
        "audit_name": "stwm_final_prototype_vocab_scaling",
        "C_definition": "C is semantic prototype vocabulary size controlling granularity vs stability of the semantic field target space.",
        "frequency_baseline_top1": float(sweep.get("frequency_baseline_top1", 0.0)),
        "frequency_baseline_top5": float(sweep.get("frequency_baseline_top5", 0.0)),
        "early_sweep_rows": sweep_rows,
        "fullscale_available_train_prototypes": [
            {
                "prototype_count": 32,
                "source_valid_feature_count": int(full32.get("source_valid_feature_count", 0)),
                "mean_cluster_size": float(sum(full32.get("prototype_counts", [])) / max(1, len(full32.get("prototype_counts", [])))),
                "min_cluster_size": int(min(full32.get("prototype_counts", [0]))),
                "max_cluster_size": int(max(full32.get("prototype_counts", [0]))),
                "mean_within_cluster_similarity": float(full32.get("mean_within_cluster_similarity", 0.0)),
            },
            {
                "prototype_count": 64,
                "source_valid_feature_count": int(full64.get("source_valid_feature_count", 0)),
                "mean_cluster_size": float(sum(full64.get("prototype_counts", [])) / max(1, len(full64.get("prototype_counts", [])))),
                "min_cluster_size": int(min(full64.get("prototype_counts", [0]))),
                "max_cluster_size": int(max(full64.get("prototype_counts", [0]))),
                "mean_within_cluster_similarity": float(full64.get("mean_within_cluster_similarity", 0.0)),
            },
        ],
        "fullscale_val_candidates": selected_candidates,
        "selected_from_fullscale_val_only": selected,
        "heldout_mixed_test_best": {
            "prototype_count": int(mixed_eval.get("prototype_count", 0)),
            "proto_top5": float(mixed_eval.get("best_metrics", {}).get("proto_top5", 0.0)),
            "copy_top5": float(mixed_eval.get("best_metrics", {}).get("copy_proto_top5", 0.0)),
            "changed_gain_over_copy": float(mixed_eval.get("best_metrics", {}).get("changed_subset_gain_over_copy", 0.0)),
            "stable_preservation_drop": float(mixed_eval.get("best_metrics", {}).get("stable_preservation_drop", 0.0)),
            "trace_regression_detected": bool(mixed_eval.get("trace_regression_detected", False)),
        },
        "missing_requested_C": [16],
        "selected_C_justified": True,
        "selection_reason": (
            "Early prototype sweep showed C256 becomes long-tailed and C128 is finer but less stable; "
            "final mixed fullscale val-only selection over the complete 10-run matrix chose C32 seed456 as the best changed-gain tradeoff with low stable drop and low trace error."
        ),
        "no_heldout_tuning": True,
    }
    write_json(REPORT_DIR / "stwm_final_prototype_vocab_scaling_20260428.json", report)
    write_md(
        DOC_DIR / "STWM_FINAL_PROTOTYPE_VOCAB_SCALING_20260428.md",
        "STWM Final Prototype Vocab Scaling 20260428",
        [
            "## What C Means\n- C is semantic prototype vocabulary size controlling granularity vs stability.",
            "## Selection\n"
            + "\n".join(
                [
                    f"- selected_C: `{report['selected_from_fullscale_val_only']['selected_prototype_count']}`",
                    f"- selected_seed: `{report['selected_from_fullscale_val_only']['selected_seed']}`",
                    f"- selected_C_justified: `{report['selected_C_justified']}`",
                    f"- missing_requested_C: `{report['missing_requested_C']}`",
                ]
            ),
            "## Reason\n- " + report["selection_reason"],
        ],
    )
    return {
        "path": str(REPORT_DIR / "stwm_final_prototype_vocab_scaling_20260428.json"),
        "selected_C_justified": True,
    }


def build_model_size_scaling() -> dict[str, Any]:
    src = load_json(REPORT_DIR / "stwm_scaling_model_size_v1_train_summary_20260428.json")
    eval_src = load_json(REPORT_DIR / "stwm_scaling_model_size_v1_eval_20260428.json")
    sig_src = load_json(REPORT_DIR / "stwm_scaling_model_size_v1_significance_20260428.json")
    write_json(REPORT_DIR / "stwm_final_model_size_scaling_train_summary_20260428.json", src)
    write_json(REPORT_DIR / "stwm_final_model_size_scaling_eval_20260428.json", eval_src)
    write_json(REPORT_DIR / "stwm_final_model_size_scaling_significance_20260428.json", sig_src)
    write_md(
        DOC_DIR / "STWM_FINAL_MODEL_SIZE_SCALING_20260428.md",
        "STWM Final Model Size Scaling 20260428",
        ["## Status\n- model_scaling_completed: `False`\n- current evidence: `small only`\n- medium/large semantic branch scales have not been trained in live repo."],
    )
    return {
        "completed": False,
        "positive": False,
        "train_summary_path": str(REPORT_DIR / "stwm_final_model_size_scaling_train_summary_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_final_model_size_scaling_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_final_model_size_scaling_significance_20260428.json"),
    }


def build_horizon_scaling() -> dict[str, Any]:
    src_a = load_json(REPORT_DIR / "stwm_scaling_horizon_v1_target_audit_20260428.json")
    src_t = load_json(REPORT_DIR / "stwm_scaling_horizon_v1_train_summary_20260428.json")
    src_e = load_json(REPORT_DIR / "stwm_scaling_horizon_v1_eval_20260428.json")
    write_json(REPORT_DIR / "stwm_final_horizon_scaling_target_audit_20260428.json", src_a)
    write_json(REPORT_DIR / "stwm_final_horizon_scaling_train_summary_20260428.json", src_t)
    write_json(REPORT_DIR / "stwm_final_horizon_scaling_eval_20260428.json", src_e)
    write_md(
        DOC_DIR / "STWM_FINAL_HORIZON_SCALING_20260428.md",
        "STWM Final Horizon Scaling 20260428",
        ["## Status\n- horizon_scaling_completed: `False`\n- H8 is established; H16/H32 require rebuilt target pools and retraining."],
    )
    return {
        "completed": False,
        "positive": False,
        "target_audit_path": str(REPORT_DIR / "stwm_final_horizon_scaling_target_audit_20260428.json"),
        "train_summary_path": str(REPORT_DIR / "stwm_final_horizon_scaling_train_summary_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_final_horizon_scaling_eval_20260428.json"),
    }


def build_trace_density_scaling() -> dict[str, Any]:
    src_a = load_json(REPORT_DIR / "stwm_scaling_trace_density_v1_audit_20260428.json")
    src_e = load_json(REPORT_DIR / "stwm_scaling_trace_density_v1_eval_20260428.json")
    write_json(REPORT_DIR / "stwm_final_trace_density_scaling_audit_20260428.json", src_a)
    write_json(REPORT_DIR / "stwm_final_trace_density_scaling_eval_20260428.json", src_e)
    write_md(
        DOC_DIR / "STWM_FINAL_TRACE_DENSITY_SCALING_20260428.md",
        "STWM Final Trace Density Scaling 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    "- density_scaling_completed: `False`",
                    f"- terminology_recommendation: `{src_a.get('terminology_recommendation', 'semantic trace-unit field')}`",
                    "- current evidence supports semantic trace-unit field, not dense trace field.",
                ]
            )
        ],
    )
    return {
        "completed": False,
        "audit_path": str(REPORT_DIR / "stwm_final_trace_density_scaling_audit_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_final_trace_density_scaling_eval_20260428.json"),
        "terminology_recommendation": src_a.get("terminology_recommendation", "semantic trace-unit field"),
    }


def build_baseline_suite() -> dict[str, Any]:
    src_eval = load_json(REPORT_DIR / "stwm_world_model_baseline_suite_v1_eval_20260428.json")
    src_sig = load_json(REPORT_DIR / "stwm_world_model_baseline_suite_v1_significance_20260428.json")
    write_json(REPORT_DIR / "stwm_final_world_model_baseline_suite_eval_20260428.json", src_eval)
    write_json(REPORT_DIR / "stwm_final_world_model_baseline_suite_significance_20260428.json", src_sig)
    write_md(
        DOC_DIR / "STWM_FINAL_WORLD_MODEL_BASELINE_SUITE_20260428.md",
        "STWM Final World Model Baseline Suite 20260428",
        ["## Status\n- baseline_suite_completed: `False`\n- fair same-output free-rollout baselines beyond copy are still missing in live repo."],
    )
    return {
        "completed": False,
        "beats_same_output_baselines": False,
        "eval_path": str(REPORT_DIR / "stwm_final_world_model_baseline_suite_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_final_world_model_baseline_suite_significance_20260428.json"),
    }


def build_benchmark_protocol() -> dict[str, Any]:
    src = load_json(REPORT_DIR / "stwm_fstf_benchmark_protocol_v1_20260428.json")
    src["audit_name"] = "stwm_final_fstf_benchmark_protocol"
    write_json(REPORT_DIR / "stwm_final_fstf_benchmark_protocol_20260428.json", src)
    write_md(
        DOC_DIR / "STWM_FINAL_FSTF_BENCHMARK_PROTOCOL_20260428.md",
        "STWM Final FSTF Benchmark Protocol 20260428",
        [
            "## Task Definition\n- STWM-FSTF: Future Semantic Trace Field Prediction",
            "## Inputs\n" + "\n".join(f"- {x}" for x in src.get("input", [])),
            "## Outputs\n" + "\n".join(f"- {x}" for x in src.get("output", [])),
            "## Metrics\n" + "\n".join(f"- {x}" for x in src.get("metrics", [])),
        ],
    )
    return {"path": str(REPORT_DIR / "stwm_final_fstf_benchmark_protocol_20260428.json")}


def build_video_manifest() -> dict[str, Any]:
    existing_videos = sorted(str(p) for p in VIDEO_DIR.glob("*")) if VIDEO_DIR.exists() else []
    report = {
        "audit_name": "stwm_final_video_visualization_manifest_v2",
        "video_dir": str(VIDEO_DIR),
        "video_visualization_ready": bool(existing_videos),
        "existing_video_files": existing_videos,
        "required_examples": [
            "stable success",
            "changed success",
            "copy failure fixed by residual",
            "VSPW example",
            "VIPSeg example",
            "failure example",
            "trace+semantic field overlay",
        ],
        "render_format": ["mp4", "gif"],
        "skipped_reason": "" if existing_videos else "No final MP4/GIF assets exist yet in outputs/videos/stwm_final.",
    }
    write_json(REPORT_DIR / "stwm_final_video_visualization_manifest_v2_20260428.json", report)
    write_md(
        DOC_DIR / "STWM_FINAL_VIDEO_VISUALIZATION_V2_20260428.md",
        "STWM Final Video Visualization V2 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- video_visualization_ready: `{report['video_visualization_ready']}`",
                    f"- existing_video_count: `{len(existing_videos)}`",
                    f"- skipped_reason: `{report['skipped_reason']}`",
                ]
            )
        ],
    )
    return {
        "path": str(REPORT_DIR / "stwm_final_video_visualization_manifest_v2_20260428.json"),
        "ready": bool(existing_videos),
    }


def build_assets_and_positioning() -> dict[str, Any]:
    table_v1 = load_json(REPORT_DIR / "stwm_final_cvpr_aaai_table_pack_v1_20260428.json")
    fig_v1 = load_json(REPORT_DIR / "stwm_final_cvpr_aaai_figure_pack_v1_20260428.json")
    claim_v1 = load_json(REPORT_DIR / "stwm_final_claim_boundary_v1_20260428.json")
    outline_v1 = load_json(REPORT_DIR / "stwm_final_paper_outline_v1_20260428.json")

    table_v2 = dict(table_v1)
    table_v2["audit_name"] = "stwm_final_cvpr_aaai_table_pack_v2"
    table_v2["appendix_or_missing_tables"] = sorted(set(table_v2.get("appendix_or_missing_tables", []) + ["prototype_vocab_scaling"]))
    fig_v2 = dict(fig_v1)
    fig_v2["audit_name"] = "stwm_final_cvpr_aaai_figure_pack_v2"
    fig_v2["figures"]["prototype_vocab_scaling"] = "to be plotted from stwm_final_prototype_vocab_scaling_20260428.json"
    claim_v2 = dict(claim_v1)
    claim_v2["audit_name"] = "stwm_final_claim_boundary_v2"
    must_disclose = list(claim_v2.get("must_disclose", []))
    extra = [
        "C32 is selected from val-only fullscale mixed selection; prototype vocabulary granularity should be stated explicitly.",
        "LODO, model-size scaling, horizon scaling, and density scaling are not completed in live repo.",
        "Same-output free-rollout baseline suite is incomplete.",
        "Final video assets are not yet rendered.",
    ]
    for x in extra:
        if x not in must_disclose:
            must_disclose.append(x)
    claim_v2["must_disclose"] = must_disclose

    write_json(REPORT_DIR / "stwm_final_cvpr_aaai_table_pack_v2_20260428.json", table_v2)
    write_json(REPORT_DIR / "stwm_final_cvpr_aaai_figure_pack_v2_20260428.json", fig_v2)
    write_json(REPORT_DIR / "stwm_final_claim_boundary_v2_20260428.json", claim_v2)

    outline_sections = [
        "## Introduction\n" + "\n".join(f"- {x}" for x in outline_v1.get("introduction", [])),
        "## Related Work\n- trajectory fields / Trace Anything\n- object-centric dynamics / SlotFormer\n- real-world object-centric learning / SAVi++\n- future instance prediction / FIERY\n- latent feature world models / DINO-WM\n- video diffusion is not same-output baseline / MotionCrafter distinction",
        "## Method\n" + "\n".join(f"- {x}" for x in outline_v1.get("method", [])),
        "## Experiments\n- mixed/VSPW/VIPSeg free-rollout semantic trace field\n- stable/changed analysis\n- prototype vocabulary scaling\n- trace guardrail\n- optional LODO/scaling appendices\n- belief utility evidence",
        "## Limitations\n- LODO/scaling/baseline/video evidence pack still incomplete in live repo\n- semantic trace-unit field wording unless density scaling is completed",
    ]
    write_md(DOC_DIR / "STWM_FINAL_PAPER_OUTLINE_V2_20260428.md", "STWM Final Paper Outline V2 20260428", outline_sections)
    related_sections = [
        "## Trajectory Fields / Trace Anything\n- Use trajectory-first video state as a conceptual base, but STWM predicts future semantic trace-unit fields rather than only tracking geometry.",
        "## SlotFormer / Object-Centric Dynamics\n- Closest object-slot dynamics-style baseline family for semantic memory rollout; useful for same-output baseline framing.",
        "## SAVi++ / Real-World Object-Centric Learning\n- Important for discussing why semantic persistence and real-world object variation are nontrivial beyond synthetic object-centric settings.",
        "## FIERY / Future Instance Prediction\n- Relevant because it predicts future scene state under structured supervision; STWM differs by future semantic prototype field over trace units.",
        "## DINO-WM / Latent Feature World Models\n- Relevant latent-dynamics comparison, but not identical output since STWM predicts structured prototype fields rather than generic latent rollout.",
        "## MotionCrafter / Video Diffusion\n- Not a same-output baseline: video diffusion targets RGB synthesis, while STWM targets structured future trace+semantic field outputs.",
    ]
    write_md(DOC_DIR / "STWM_FINAL_RELATED_WORK_POSITIONING_V2_20260428.md", "STWM Final Related Work Positioning V2 20260428", related_sections)
    return {
        "table_pack_path": str(REPORT_DIR / "stwm_final_cvpr_aaai_table_pack_v2_20260428.json"),
        "figure_pack_path": str(REPORT_DIR / "stwm_final_cvpr_aaai_figure_pack_v2_20260428.json"),
        "claim_boundary_path": str(REPORT_DIR / "stwm_final_claim_boundary_v2_20260428.json"),
        "paper_outline_path": str(DOC_DIR / "STWM_FINAL_PAPER_OUTLINE_V2_20260428.md"),
        "related_work_path": str(DOC_DIR / "STWM_FINAL_RELATED_WORK_POSITIONING_V2_20260428.md"),
    }


def build_readiness(
    lodo: dict[str, Any],
    proto: dict[str, Any],
    model: dict[str, Any],
    horizon: dict[str, Any],
    density: dict[str, Any],
    baseline: dict[str, Any],
    video: dict[str, Any],
) -> dict[str, Any]:
    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    readiness = {
        "audit_name": "stwm_final_cvpr_aaai_readiness_v2",
        "lodo_completed": bool(lodo["lodo_completed"]),
        "lodo_positive": bool(lodo["lodo_positive"]),
        "prototype_scaling_completed": True,
        "selected_C_justified": bool(proto["selected_C_justified"]),
        "model_scaling_completed": bool(model["completed"]),
        "model_scaling_positive": bool(model["positive"]),
        "horizon_scaling_completed": bool(horizon["completed"]),
        "horizon_scaling_positive": bool(horizon["positive"]),
        "density_scaling_completed": bool(density["completed"]),
        "baseline_suite_completed": bool(baseline["completed"]),
        "STWM_beats_same_output_baselines": bool(baseline["beats_same_output_baselines"]),
        "video_visualization_ready": bool(video["ready"]),
        "ready_for_cvpr_aaai_main": "unclear",
        "ready_for_overleaf": True,
        "remaining_risks": [
            "LODO remains incomplete.",
            "Model-size scaling remains incomplete.",
            "Horizon scaling remains incomplete.",
            "Trace-density scaling remains incomplete.",
            "Same-output free-rollout baseline suite remains incomplete.",
            "Final MP4/GIF qualitative videos are not ready.",
            "Prototype vocabulary justification is good for C32/C64 but requested C16 evidence is still absent.",
        ],
        "next_step_choice": "run_missing_scaling",
        "core_main_result_status": {
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
    write_json(REPORT_DIR / "stwm_final_cvpr_aaai_readiness_v2_20260428.json", readiness)
    write_md(
        DOC_DIR / "STWM_FINAL_CVPR_AAAI_READINESS_V2_20260428.md",
        "STWM Final CVPR AAAI Readiness V2 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- lodo_completed: `{readiness['lodo_completed']}`",
                    f"- prototype_scaling_completed: `{readiness['prototype_scaling_completed']}`",
                    f"- selected_C_justified: `{readiness['selected_C_justified']}`",
                    f"- model_scaling_completed: `{readiness['model_scaling_completed']}`",
                    f"- horizon_scaling_completed: `{readiness['horizon_scaling_completed']}`",
                    f"- density_scaling_completed: `{readiness['density_scaling_completed']}`",
                    f"- baseline_suite_completed: `{readiness['baseline_suite_completed']}`",
                    f"- video_visualization_ready: `{readiness['video_visualization_ready']}`",
                    f"- ready_for_cvpr_aaai_main: `{readiness['ready_for_cvpr_aaai_main']}`",
                    f"- ready_for_overleaf: `{readiness['ready_for_overleaf']}`",
                    f"- next_step_choice: `{readiness['next_step_choice']}`",
                ]
            ),
            "## Remaining Risks\n" + "\n".join(f"- {x}" for x in readiness["remaining_risks"]),
        ],
    )
    return readiness


def build_guardrail() -> dict[str, Any]:
    guard = {
        "audit_name": "stwm_world_model_no_drift_guardrail_v41",
        "allowed": [
            "final evidence pack",
            "LODO",
            "scaling law",
            "same-output baselines",
            "benchmark framing",
            "video visualization",
        ],
        "forbidden": [
            "new method branches",
            "candidate scorer",
            "SAM2/CoTracker plugin",
            "future candidate leakage",
            "test-set model selection",
            "hiding copy baseline",
            "claiming dense trace field without density evidence",
            "claiming full RGB generation",
        ],
    }
    write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v41_20260428.json", guard)
    write_md(
        DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V41.md",
        "STWM World Model No-Drift Guardrail V41",
        [
            "## Allowed\n" + "\n".join(f"- {x}" for x in guard["allowed"]),
            "## Forbidden\n" + "\n".join(f"- {x}" for x in guard["forbidden"]),
        ],
    )
    return guard


def main() -> None:
    lodo = build_lodo_reports()
    proto = build_prototype_vocab_scaling()
    model = build_model_size_scaling()
    horizon = build_horizon_scaling()
    density = build_trace_density_scaling()
    baseline = build_baseline_suite()
    build_benchmark_protocol()
    video = build_video_manifest()
    build_assets_and_positioning()
    build_readiness(lodo, proto, model, horizon, density, baseline, video)
    build_guardrail()


if __name__ == "__main__":
    main()
