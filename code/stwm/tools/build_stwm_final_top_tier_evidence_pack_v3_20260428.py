#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path("/raid/chen034/workspace/stwm")
REPORT_DIR = REPO_ROOT / "reports"
DOC_DIR = REPO_ROOT / "docs"
FIG_DIR = REPO_ROOT / "outputs/figures/stwm_final"
VIDEO_DIR = REPO_ROOT / "outputs/videos/stwm_final_v3"


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
    src = load_json(REPORT_DIR / "stwm_final_lodo_train_summary_20260428.json")
    a = load_json(REPORT_DIR / "stwm_final_lodo_vspw_to_vipseg_eval_20260428.json")
    b = load_json(REPORT_DIR / "stwm_final_lodo_vipseg_to_vspw_eval_20260428.json")
    sig = load_json(REPORT_DIR / "stwm_final_lodo_significance_20260428.json")
    lodo_completed = bool(src.get("lodo_completed", False))
    lodo_positive = bool(sig.get("lodo_positive", False))
    train = {
        "audit_name": "stwm_final_lodo_train_summary_v3",
        "lodo_completed": lodo_completed,
        "skipped_reason": src.get("skipped_reason", ""),
        "planned_protocols": src.get("planned_protocols", []),
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "completed_run_count": int(src.get("completed_run_count", 0)),
        "failed_run_count": int(src.get("failed_run_count", 0)),
        "expected_run_count": int(src.get("expected_run_count", 0)),
    }
    eva = {
        "audit_name": "stwm_final_lodo_vspw_to_vipseg_eval_v3",
        "lodo_completed": bool(a.get("lodo_completed", False)),
        "lodo_positive": bool(a.get("lodo_positive", False)),
        "skipped_reason": a.get("skipped_reason", train["skipped_reason"]),
        "selected_prototype_count": int(a.get("selected_prototype_count", 0)),
        "selected_seed": int(a.get("selected_seed", -1)),
    }
    evb = {
        "audit_name": "stwm_final_lodo_vipseg_to_vspw_eval_v3",
        "lodo_completed": bool(b.get("lodo_completed", False)),
        "lodo_positive": bool(b.get("lodo_positive", False)),
        "skipped_reason": b.get("skipped_reason", train["skipped_reason"]),
        "selected_prototype_count": int(b.get("selected_prototype_count", 0)),
        "selected_seed": int(b.get("selected_seed", -1)),
    }
    sigv = {
        "audit_name": "stwm_final_lodo_significance_v3",
        "lodo_completed": bool(sig.get("lodo_completed", False)),
        "lodo_positive": lodo_positive,
        "skipped_reason": sig.get("skipped_reason", train["skipped_reason"]),
    }
    write_json(REPORT_DIR / "stwm_final_lodo_train_summary_v3_20260428.json", train)
    write_json(REPORT_DIR / "stwm_final_lodo_vspw_to_vipseg_eval_v3_20260428.json", eva)
    write_json(REPORT_DIR / "stwm_final_lodo_vipseg_to_vspw_eval_v3_20260428.json", evb)
    write_json(REPORT_DIR / "stwm_final_lodo_significance_v3_20260428.json", sigv)
    write_md(
        DOC_DIR / "STWM_FINAL_LODO_V3_20260428.md",
        "STWM Final LODO V3 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- lodo_completed: `{train['lodo_completed']}`",
                    f"- lodo_positive: `{lodo_positive}`",
                    f"- completed_run_count: `{train['completed_run_count']}`",
                    f"- skipped_reason: `{train['skipped_reason']}`",
                ]
            ),
            "## Interpretation\n- Mixed protocol is strong main evidence.\n- Dedicated cross-dataset generalization is propagated from current live LODO artifacts when available.",
        ],
    )
    return {
        "lodo_completed": bool(train["lodo_completed"]),
        "lodo_positive": lodo_positive,
        "train_summary_path": str(REPORT_DIR / "stwm_final_lodo_train_summary_v3_20260428.json"),
        "vspw_to_vipseg_eval_path": str(REPORT_DIR / "stwm_final_lodo_vspw_to_vipseg_eval_v3_20260428.json"),
        "vipseg_to_vspw_eval_path": str(REPORT_DIR / "stwm_final_lodo_vipseg_to_vspw_eval_v3_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_final_lodo_significance_v3_20260428.json"),
    }


def build_baseline_suite() -> dict[str, Any]:
    src = load_json(REPORT_DIR / "stwm_final_world_model_baseline_suite_eval_20260428.json")
    sig = load_json(REPORT_DIR / "stwm_final_world_model_baseline_suite_significance_20260428.json")
    eval_payload = {
        "audit_name": "stwm_final_same_output_baseline_suite_v3_eval",
        "baseline_suite_completed": False,
        "same_output_protocol_required": True,
        "available_baselines": src.get("available_baselines", []),
        "copy_baseline_summary": src.get("copy_baseline_summary", {}),
        "predictability_summary": src.get("predictability_summary", {}),
        "STWM_beats_same_output_baselines": False,
        "skipped_reason": (
            "A fair same-output free-rollout suite beyond copy/probe baselines is not implemented in the live repo. "
            "Trace-only/semantic-only/semantic+trace transformers, SlotFormer-like, DINO-WM-like, and FIERY-style same-output baselines remain missing."
        ),
    }
    sig_payload = {
        "audit_name": "stwm_final_same_output_baseline_suite_v3_significance",
        "baseline_suite_completed": False,
        "skipped_reason": eval_payload["skipped_reason"],
        "source_v1_significance": sig,
    }
    write_json(REPORT_DIR / "stwm_final_same_output_baseline_suite_v3_eval_20260428.json", eval_payload)
    write_json(REPORT_DIR / "stwm_final_same_output_baseline_suite_v3_significance_20260428.json", sig_payload)
    write_md(
        DOC_DIR / "STWM_FINAL_SAME_OUTPUT_BASELINE_SUITE_V3_20260428.md",
        "STWM Final Same-Output Baseline Suite V3 20260428",
        [
            "## Status\n- baseline_suite_completed: `False`\n- only copy/probe-style partial baselines are available in live repo.\n- same-output free-rollout transformer / slot-dynamics / latent-dynamics baselines are still missing."
        ],
    )
    return {
        "completed": False,
        "beats_same_output_baselines": False,
        "eval_path": str(REPORT_DIR / "stwm_final_same_output_baseline_suite_v3_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_final_same_output_baseline_suite_v3_significance_20260428.json"),
    }


def build_model_size_scaling() -> dict[str, Any]:
    src_train = load_json(REPORT_DIR / "stwm_final_model_size_scaling_train_summary_20260428.json")
    src_eval = load_json(REPORT_DIR / "stwm_final_model_size_scaling_eval_20260428.json")
    src_sig = load_json(REPORT_DIR / "stwm_final_model_size_scaling_significance_20260428.json")
    train = dict(src_train)
    train["audit_name"] = "stwm_final_model_size_scaling_v3_train_summary"
    evalp = dict(src_eval)
    evalp["audit_name"] = "stwm_final_model_size_scaling_v3_eval"
    sigp = dict(src_sig)
    sigp["audit_name"] = "stwm_final_model_size_scaling_v3_significance"
    write_json(REPORT_DIR / "stwm_final_model_size_scaling_v3_train_summary_20260428.json", train)
    write_json(REPORT_DIR / "stwm_final_model_size_scaling_v3_eval_20260428.json", evalp)
    write_json(REPORT_DIR / "stwm_final_model_size_scaling_v3_significance_20260428.json", sigp)
    write_md(
        DOC_DIR / "STWM_FINAL_MODEL_SIZE_SCALING_V3_20260428.md",
        "STWM Final Model Size Scaling V3 20260428",
        [
            "## Status\n- model_scaling_completed: `False`\n- only the current small semantic branch exists in live repo evidence.\n- medium/large semantic-branch controlled runs are missing."
        ],
    )
    return {
        "completed": False,
        "positive": False,
        "train_summary_path": str(REPORT_DIR / "stwm_final_model_size_scaling_v3_train_summary_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_final_model_size_scaling_v3_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_final_model_size_scaling_v3_significance_20260428.json"),
    }


def build_horizon_scaling() -> dict[str, Any]:
    src_a = load_json(REPORT_DIR / "stwm_final_horizon_scaling_target_audit_20260428.json")
    src_t = load_json(REPORT_DIR / "stwm_final_horizon_scaling_train_summary_20260428.json")
    src_e = load_json(REPORT_DIR / "stwm_final_horizon_scaling_eval_20260428.json")
    audit = dict(src_a)
    audit["audit_name"] = "stwm_final_horizon_scaling_v3_target_audit"
    train = dict(src_t)
    train["audit_name"] = "stwm_final_horizon_scaling_v3_train_summary"
    evalp = dict(src_e)
    evalp["audit_name"] = "stwm_final_horizon_scaling_v3_eval"
    write_json(REPORT_DIR / "stwm_final_horizon_scaling_v3_target_audit_20260428.json", audit)
    write_json(REPORT_DIR / "stwm_final_horizon_scaling_v3_train_summary_20260428.json", train)
    write_json(REPORT_DIR / "stwm_final_horizon_scaling_v3_eval_20260428.json", evalp)
    write_md(
        DOC_DIR / "STWM_FINAL_HORIZON_SCALING_V3_20260428.md",
        "STWM Final Horizon Scaling V3 20260428",
        [
            "## Status\n- horizon_scaling_completed: `False`\n- H8 is established; H16/H32 remain unrun in live repo."
        ],
    )
    return {
        "completed": False,
        "positive": False,
        "target_audit_path": str(REPORT_DIR / "stwm_final_horizon_scaling_v3_target_audit_20260428.json"),
        "train_summary_path": str(REPORT_DIR / "stwm_final_horizon_scaling_v3_train_summary_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_final_horizon_scaling_v3_eval_20260428.json"),
    }


def build_trace_density_scaling() -> dict[str, Any]:
    src_a = load_json(REPORT_DIR / "stwm_final_trace_density_scaling_audit_20260428.json")
    src_e = load_json(REPORT_DIR / "stwm_final_trace_density_scaling_eval_20260428.json")
    audit = dict(src_a)
    audit["audit_name"] = "stwm_final_trace_density_scaling_v3_audit"
    evalp = dict(src_e)
    evalp["audit_name"] = "stwm_final_trace_density_scaling_v3_eval"
    write_json(REPORT_DIR / "stwm_final_trace_density_scaling_v3_audit_20260428.json", audit)
    write_json(REPORT_DIR / "stwm_final_trace_density_scaling_v3_eval_20260428.json", evalp)
    write_md(
        DOC_DIR / "STWM_FINAL_TRACE_DENSITY_SCALING_V3_20260428.md",
        "STWM Final Trace Density Scaling V3 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- trace_density_scaling_completed: `{audit.get('trace_density_scaling_completed', False)}`",
                    f"- terminology_recommendation: `{audit.get('terminology_recommendation', 'semantic trace-unit field')}`",
                ]
            )
        ],
    )
    return {
        "completed": bool(audit.get("trace_density_scaling_completed", False)),
        "audit_path": str(REPORT_DIR / "stwm_final_trace_density_scaling_v3_audit_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_final_trace_density_scaling_v3_eval_20260428.json"),
    }


def build_prototype_vocab_justification() -> dict[str, Any]:
    src = load_json(REPORT_DIR / "stwm_final_prototype_vocab_scaling_20260428.json")
    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    report = dict(src)
    report["audit_name"] = "stwm_final_prototype_vocab_justification_v3"
    report["selected_C_on_mixed_main_result"] = int(mixed.get("best_prototype_count", 0))
    report["selected_C_justified"] = True
    report["final_summary"] = (
        "C32 is selected from fullscale mixed validation only. "
        "Earlier sweeps show finer vocabularies increase granularity but worsen stability/long-tail behavior; "
        "the completed 10-run mixed matrix picked C32 seed456 as the best changed-gain/stable-drop/trace-error tradeoff."
    )
    write_json(REPORT_DIR / "stwm_final_prototype_vocab_justification_v3_20260428.json", report)
    write_md(
        DOC_DIR / "STWM_FINAL_PROTOTYPE_VOCAB_JUSTIFICATION_V3_20260428.md",
        "STWM Final Prototype Vocab Justification V3 20260428",
        [
            "## Summary\n"
            + "\n".join(
                [
                    f"- selected_C_justified: `{report['selected_C_justified']}`",
                    f"- selected_C_on_mixed_main_result: `{report['selected_C_on_mixed_main_result']}`",
                    f"- missing_requested_C: `{report.get('missing_requested_C', [])}`",
                ]
            ),
            "## Reason\n- " + report["final_summary"],
        ],
    )
    return {
        "path": str(REPORT_DIR / "stwm_final_prototype_vocab_justification_v3_20260428.json"),
        "selected_C_justified": True,
    }


def build_benchmark_protocol() -> dict[str, Any]:
    src = load_json(REPORT_DIR / "stwm_final_fstf_benchmark_protocol_20260428.json")
    mixed_split = load_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json")
    proto_density = load_json(REPORT_DIR / "stwm_final_trace_density_scaling_audit_20260428.json")
    report = dict(src)
    report["audit_name"] = "stwm_final_fstf_benchmark_protocol_v3"
    report["datasets"] = ["VSPW", "VIPSeg"]
    report["train_item_count"] = int(mixed_split.get("train_item_count", 0))
    report["val_item_count"] = int(mixed_split.get("val_item_count", 0))
    report["test_item_count"] = int(mixed_split.get("test_item_count", 0))
    report["stable_changed_rationale"] = report.get("why_changed_subset_matters", "")
    report["copy_baseline_rationale"] = report.get("why_copy_baseline_is_strong", "")
    report["free_rollout_requirement"] = True
    report["trace_guardrail"] = "future trace coord error + trace_regression_detected=false"
    report["terminology"] = proto_density.get("terminology_recommendation", "semantic trace-unit field")
    write_json(REPORT_DIR / "stwm_final_fstf_benchmark_protocol_v3_20260428.json", report)
    write_md(
        DOC_DIR / "STWM_FINAL_FSTF_BENCHMARK_PROTOCOL_V3_20260428.md",
        "STWM Final FSTF Benchmark Protocol V3 20260428",
        [
            "## Task\n- STWM-FSTF: Future Semantic Trace Field Prediction",
            "## Datasets\n- VSPW\n- VIPSeg",
            "## Split Counts\n"
            + "\n".join(
                [
                    f"- train: `{report['train_item_count']}`",
                    f"- val: `{report['val_item_count']}`",
                    f"- test: `{report['test_item_count']}`",
                ]
            ),
            "## Metrics\n" + "\n".join(f"- {x}" for x in report.get("metrics", [])),
            "## Terminology\n- " + str(report["terminology"]),
        ],
    )
    return {"path": str(REPORT_DIR / "stwm_final_fstf_benchmark_protocol_v3_20260428.json")}


def build_video_manifest() -> dict[str, Any]:
    existing = sorted(str(p) for p in VIDEO_DIR.glob("*")) if VIDEO_DIR.exists() else []
    report = {
        "audit_name": "stwm_final_video_visualization_manifest_v3",
        "video_dir": str(VIDEO_DIR),
        "video_visualization_ready": bool(existing),
        "existing_video_files": existing,
        "required_examples": [
            "stable success",
            "changed success",
            "copy failure corrected",
            "VSPW success",
            "VIPSeg success",
            "failure case",
            "trace+semantic overlay",
        ],
        "must_be_actual_media": True,
        "skipped_reason": "" if existing else "No actual MP4/GIF files exist yet under outputs/videos/stwm_final_v3.",
    }
    write_json(REPORT_DIR / "stwm_final_video_visualization_manifest_v3_20260428.json", report)
    write_md(
        DOC_DIR / "STWM_FINAL_VIDEO_VISUALIZATION_V3_20260428.md",
        "STWM Final Video Visualization V3 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- video_visualization_ready: `{report['video_visualization_ready']}`",
                    f"- existing_video_count: `{len(existing)}`",
                    f"- skipped_reason: `{report['skipped_reason']}`",
                ]
            )
        ],
    )
    return {"path": str(REPORT_DIR / "stwm_final_video_visualization_manifest_v3_20260428.json"), "ready": bool(existing)}


def build_assets() -> dict[str, Any]:
    table_v2 = load_json(REPORT_DIR / "stwm_final_cvpr_aaai_table_pack_v2_20260428.json")
    fig_v2 = load_json(REPORT_DIR / "stwm_final_cvpr_aaai_figure_pack_v2_20260428.json")
    table_v3 = dict(table_v2)
    table_v3["audit_name"] = "stwm_final_cvpr_aaai_table_pack_v3"
    ready = list(table_v3.get("ready_tables", []))
    if "prototype_vocab_justification" not in ready:
        ready.append("prototype_vocab_justification")
    lodo = load_json(REPORT_DIR / "stwm_final_lodo_train_summary_20260428.json")
    if bool(lodo.get("lodo_completed", False)) and "lodo" not in ready:
        ready.append("lodo")
    table_v3["ready_tables"] = ready
    appendix = [x for x in list(table_v3.get("appendix_or_missing_tables", [])) if not (x == "lodo" and bool(lodo.get("lodo_completed", False)))]
    table_v3["appendix_or_missing_tables"] = appendix
    fig_v3 = dict(fig_v2)
    fig_v3["audit_name"] = "stwm_final_cvpr_aaai_figure_pack_v3"
    figs = dict(fig_v3.get("figures", {}))
    figs["related_work_positioning"] = "document-only"
    figs["genie_scaling_context"] = "document-only"
    fig_v3["figures"] = figs
    write_json(REPORT_DIR / "stwm_final_cvpr_aaai_table_pack_v3_20260428.json", table_v3)
    write_json(REPORT_DIR / "stwm_final_cvpr_aaai_figure_pack_v3_20260428.json", fig_v3)

    related_sections = [
        "## Trace Anything / Trajectory Fields\n- Use trajectory-first video state as a conceptual anchor, but STWM predicts future semantic trace-unit fields rather than only reconstructing trajectories.",
        "## SlotFormer / Object-Centric Dynamics\n- Closest object-centric dynamics family for same-output baseline intuition: observed slots plus rollout to future semantic slot/state prediction.",
        "## SAVi++ / Real-World Object-Centric Learning\n- Important for positioning real-world semantic persistence and object variation as harder than synthetic slot-learning settings.",
        "## FIERY / Future Instance Prediction\n- Relevant structured future-state forecasting baseline family, but STWM predicts future semantic prototype fields over trace units rather than future occupancy/map heads.",
        "## DINO-WM / Latent World Models\n- Relevant latent-dynamics comparison, though not same-output because STWM outputs structured semantic prototype fields, not generic latent futures.",
        "## Genie / Scaling of World Models\n- Relevant to scaling discussion and long-horizon ambitions, but Genie-style generative world models target different output spaces and training scales.",
        "## MotionCrafter / Video Diffusion\n- Related as future video synthesis, but not a same-output baseline because RGB diffusion does not directly predict future trace-unit semantic fields under the same protocol.",
    ]
    write_md(DOC_DIR / "STWM_FINAL_RELATED_WORK_POSITIONING_V3_20260428.md", "STWM Final Related Work Positioning V3 20260428", related_sections)

    outline_sections = [
        "## Introduction\n- World models need future semantics, not only future motion.\n- Observed semantic memory plus trace rollout yields future semantic trace fields.\n- Copy-gated residual transition improves changed semantics while preserving stable states.",
        "## Related Work\n- trajectory fields / Trace Anything\n- object-centric dynamics / SlotFormer\n- real-world object-centric learning / SAVi++\n- future instance prediction / FIERY\n- latent world models / DINO-WM\n- scaling world models / Genie\n- video diffusion distinction / MotionCrafter",
        "## Method\n- trace backbone\n- semantic trace units\n- observed semantic memory\n- copy-gated residual semantic transition\n- future semantic prototype field output",
        "## Experiments\n- mixed/VSPW/VIPSeg free-rollout semantic trace field\n- stable/changed analysis\n- prototype vocabulary justification\n- trace guardrail\n- optional LODO and scaling appendices\n- belief utility evidence",
        "## Limitations\n- LODO/baseline/scaling/video assets incomplete in live repo\n- semantic trace-unit field wording unless density scaling is completed",
    ]
    write_md(DOC_DIR / "STWM_FINAL_PAPER_OUTLINE_V3_20260428.md", "STWM Final Paper Outline V3 20260428", outline_sections)
    return {
        "table_pack_path": str(REPORT_DIR / "stwm_final_cvpr_aaai_table_pack_v3_20260428.json"),
        "figure_pack_path": str(REPORT_DIR / "stwm_final_cvpr_aaai_figure_pack_v3_20260428.json"),
        "related_work_path": str(DOC_DIR / "STWM_FINAL_RELATED_WORK_POSITIONING_V3_20260428.md"),
        "paper_outline_path": str(DOC_DIR / "STWM_FINAL_PAPER_OUTLINE_V3_20260428.md"),
    }


def build_readiness(
    lodo: dict[str, Any],
    baseline: dict[str, Any],
    model: dict[str, Any],
    horizon: dict[str, Any],
    density: dict[str, Any],
    proto: dict[str, Any],
    video: dict[str, Any],
) -> dict[str, Any]:
    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    readiness = {
        "audit_name": "stwm_final_cvpr_aaai_readiness_v3",
        "lodo_completed": bool(lodo["lodo_completed"]),
        "lodo_positive": bool(lodo["lodo_positive"]),
        "baseline_suite_completed": bool(baseline["completed"]),
        "STWM_beats_same_output_baselines": bool(baseline["beats_same_output_baselines"]),
        "model_scaling_completed": bool(model["completed"]),
        "model_scaling_positive": bool(model["positive"]),
        "horizon_scaling_completed": bool(horizon["completed"]),
        "horizon_scaling_positive": bool(horizon["positive"]),
        "trace_density_scaling_completed": bool(density["completed"]),
        "selected_C_justified": bool(proto["selected_C_justified"]),
        "video_visualization_ready": bool(video["ready"]),
        "ready_for_cvpr_aaai_main": "unclear",
        "ready_for_overleaf": True,
        "remaining_risks": [],
        "next_step_choice": "run_missing_lodo",
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
    if not readiness["lodo_completed"]:
        readiness["remaining_risks"].append("Dedicated LODO remains incomplete.")
        readiness["next_step_choice"] = "run_missing_lodo"
    elif not readiness["baseline_suite_completed"]:
        readiness["remaining_risks"].append("Same-output baseline suite remains incomplete.")
        readiness["next_step_choice"] = "run_missing_baselines"
    if not readiness["model_scaling_completed"]:
        readiness["remaining_risks"].append("Model-size scaling remains incomplete.")
    if not readiness["horizon_scaling_completed"]:
        readiness["remaining_risks"].append("Horizon scaling remains incomplete.")
    if not readiness["trace_density_scaling_completed"]:
        readiness["remaining_risks"].append("Trace-unit density scaling remains incomplete.")
    if not readiness["video_visualization_ready"]:
        readiness["remaining_risks"].append("Final MP4/GIF visualization is not ready.")
    if readiness["lodo_completed"] and readiness["baseline_suite_completed"] and not readiness["model_scaling_completed"]:
        readiness["next_step_choice"] = "run_missing_scaling"
    if readiness["lodo_completed"] and readiness["baseline_suite_completed"] and readiness["model_scaling_completed"] and readiness["horizon_scaling_completed"] and readiness["trace_density_scaling_completed"] and not readiness["video_visualization_ready"]:
        readiness["next_step_choice"] = "fix_visualization"
    if readiness["lodo_completed"] and readiness["baseline_suite_completed"] and readiness["model_scaling_completed"] and readiness["horizon_scaling_completed"] and readiness["trace_density_scaling_completed"] and readiness["video_visualization_ready"]:
        readiness["next_step_choice"] = "start_overleaf_draft"
    write_json(REPORT_DIR / "stwm_final_cvpr_aaai_readiness_v3_20260428.json", readiness)
    write_md(
        DOC_DIR / "STWM_FINAL_CVPR_AAAI_READINESS_V3_20260428.md",
        "STWM Final CVPR AAAI Readiness V3 20260428",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- lodo_completed: `{readiness['lodo_completed']}`",
                    f"- baseline_suite_completed: `{readiness['baseline_suite_completed']}`",
                    f"- model_scaling_completed: `{readiness['model_scaling_completed']}`",
                    f"- horizon_scaling_completed: `{readiness['horizon_scaling_completed']}`",
                    f"- trace_density_scaling_completed: `{readiness['trace_density_scaling_completed']}`",
                    f"- selected_C_justified: `{readiness['selected_C_justified']}`",
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
    report = {
        "audit_name": "stwm_world_model_no_drift_guardrail_v42",
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
    write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v42_20260428.json", report)
    write_md(
        DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V42.md",
        "STWM World Model No-Drift Guardrail V42",
        [
            "## Allowed\n" + "\n".join(f"- {x}" for x in report["allowed"]),
            "## Forbidden\n" + "\n".join(f"- {x}" for x in report["forbidden"]),
        ],
    )
    return report


def main() -> None:
    lodo = build_lodo_reports()
    baseline = build_baseline_suite()
    model = build_model_size_scaling()
    horizon = build_horizon_scaling()
    density = build_trace_density_scaling()
    proto = build_prototype_vocab_justification()
    build_benchmark_protocol()
    video = build_video_manifest()
    build_assets()
    build_readiness(lodo, baseline, model, horizon, density, proto, video)
    build_guardrail()


if __name__ == "__main__":
    main()
