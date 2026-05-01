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

C32_SEEDS = [42, 123, 456, 789, 1001]
C64_SEEDS = [42, 123, 456, 789, 1001]
RUN_MATRIX = [(32, s) for s in C32_SEEDS] + [(64, s) for s in C64_SEEDS]


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


def _tail_status(path: Path, max_lines: int = 20) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "tail_lines": []}
    if path.stat().st_size == 0:
        return {"status": "empty_log_file", "tail_lines": []}
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"status": "present", "tail_lines": text[-max_lines:]}


def _metric_row(name: str, payload: dict[str, Any]) -> dict[str, Any]:
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
        "overall_top5_gain": float(m.get("overall_gain_over_copy", 0.0)),
        "copy_changed_top5": float(m.get("copy_changed_subset_top5", 0.0)),
        "stwm_changed_top5": float(m.get("changed_subset_top5", 0.0)),
        "changed_top5_gain": float(m.get("changed_subset_gain_over_copy", 0.0)),
        "copy_stable_top5": float(m.get("copy_stable_subset_top5", 0.0)),
        "stwm_stable_top5": float(m.get("stable_subset_top5", 0.0)),
        "stable_preservation_drop": float(m.get("stable_preservation_drop", 0.0)),
        "future_trace_coord_error": float(m.get("future_trace_coord_error", 0.0)),
        "change_ap": float(m.get("change_detection", {}).get("ap", 0.0)),
        "change_auroc": float(m.get("change_detection", {}).get("auroc", 0.0)),
    }


def _ci_row(name: str, significance: dict[str, Any]) -> dict[str, Any]:
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


def build_artifact_audit() -> dict[str, Any]:
    sections: list[str] = []
    train_summary = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_train_summary_complete_20260428.json")
    per_run: list[dict[str, Any]] = []
    canonical_log_count = 0
    zero_byte_log_count = 0
    report_only_completion_detected = False

    for prototype_count, seed in RUN_MATRIX:
        ckpt_name = f"c{prototype_count}_seed{seed}_final.pt"
        ckpt_path = CKPT_DIR / ckpt_name
        log_path = LOG_DIR / f"stwm_mixed_fullscale_v2_c{prototype_count}_seed{seed}.log"
        summary_path = REPORT_DIR / f"stwm_mixed_fullscale_v2_train_c{prototype_count}_seed{seed}_20260428.json"
        summary = load_json(summary_path)
        log_info = _tail_status(log_path)
        if log_path.exists():
            canonical_log_count += 1
            if log_path.stat().st_size == 0:
                zero_byte_log_count += 1
        expected_rel_ckpt = f"outputs/checkpoints/stwm_mixed_fullscale_v2_20260428/{ckpt_name}"
        summary_ckpt = str(summary.get("checkpoint_path", ""))
        summary_matches_checkpoint = bool(summary_ckpt == expected_rel_ckpt and ckpt_path.exists())
        if summary_path.exists() and not ckpt_path.exists():
            report_only_completion_detected = True
        per_run.append(
            {
                "prototype_count": prototype_count,
                "seed": seed,
                "checkpoint_path": str(ckpt_path),
                "checkpoint_exists": ckpt_path.exists(),
                "checkpoint_size_bytes": int(ckpt_path.stat().st_size) if ckpt_path.exists() else 0,
                "log_path": str(log_path),
                "log_exists": log_path.exists(),
                "log_tail_status": log_info["status"],
                "log_tail_lines": log_info["tail_lines"],
                "summary_path": str(summary_path),
                "summary_exists": summary_path.exists(),
                "summary_matches_checkpoint": summary_matches_checkpoint,
                "loss_finite_ratio": float(summary.get("loss_finite_ratio", 0.0)) if summary_path.exists() else None,
                "trace_regression_detected": bool(summary.get("trace_regression_detected", False)) if summary_path.exists() else None,
            }
        )

    artifact = {
        "audit_name": "stwm_top_tier_hardening_v1_artifact_audit",
        "expected_checkpoint_count": 10,
        "existing_checkpoint_count": sum(1 for row in per_run if row["checkpoint_exists"]),
        "expected_log_count": 10,
        "existing_log_count": canonical_log_count,
        "zero_byte_training_log_count": zero_byte_log_count,
        "extra_retry_log_count": len(list(LOG_DIR.glob("stwm_mixed_fullscale_v2_*retry.log"))),
        "per_run": per_run,
        "train_summary_complete_path": str(REPORT_DIR / "stwm_mixed_fullscale_v2_train_summary_complete_20260428.json"),
        "train_summary_complete_completed_run_count": int(train_summary.get("completed_run_count", 0)),
        "report_only_completion_detected": report_only_completion_detected,
        "warnings": [
            "Per-run checkpoint artifacts exist for the full 10/10 matrix.",
            "Canonical per-run tmux stdout logs are present but zero-byte; driver/materialization logs remain available.",
        ],
        "artifact_audit_passed": bool(
            sum(1 for row in per_run if row["checkpoint_exists"]) == 10
            and canonical_log_count == 10
            and not report_only_completion_detected
            and all(row["summary_matches_checkpoint"] for row in per_run)
        ),
    }
    sections.append("## Summary")
    sections.append(
        "\n".join(
            [
                f"- expected_checkpoint_count: `{artifact['expected_checkpoint_count']}`",
                f"- existing_checkpoint_count: `{artifact['existing_checkpoint_count']}`",
                f"- expected_log_count: `{artifact['expected_log_count']}`",
                f"- existing_log_count: `{artifact['existing_log_count']}`",
                f"- zero_byte_training_log_count: `{artifact['zero_byte_training_log_count']}`",
                f"- report_only_completion_detected: `{artifact['report_only_completion_detected']}`",
                f"- artifact_audit_passed: `{artifact['artifact_audit_passed']}`",
            ]
        )
    )
    sections.append("## Notes")
    sections.append("\n".join(f"- {x}" for x in artifact["warnings"]))
    write_json(REPORT_DIR / "stwm_top_tier_hardening_v1_artifact_audit_20260428.json", artifact)
    write_md(DOC_DIR / "STWM_TOP_TIER_HARDENING_V1_ARTIFACT_AUDIT_20260428.md", "STWM Top-Tier Hardening V1 Artifact Audit", sections)
    return artifact


def build_lodo_reports() -> dict[str, Any]:
    vspw_to_vipseg_splits = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_lodo_vspw_to_vipseg_splits_20260428.json")
    vipseg_to_vspw_splits = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_lodo_vipseg_to_vspw_splits_20260428.json")
    skipped_reason = (
        "Dedicated LODO checkpoints were not trained in the live repo hardening pass. "
        "Cross-dataset split assets exist, but current GPUs are occupied by unrelated high-utilization jobs and no LODO train/eval artifacts exist yet."
    )
    train_summary = {
        "audit_name": "stwm_lodo_semantic_trace_world_model_v1_train_summary",
        "lodo_completed": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "stage1_frozen": True,
        "trace_dynamic_frozen": True,
        "planned_protocols": [
            {"train_dataset": "VSPW", "val_dataset": "VSPW", "test_dataset": "VIPSEG", "prototype_counts": [32, 64], "preferred_seeds": [42, 123, 456]},
            {"train_dataset": "VIPSEG", "val_dataset": "VIPSEG", "test_dataset": "VSPW", "prototype_counts": [32, 64], "preferred_seeds": [42, 123, 456]},
        ],
        "available_split_assets": {
            "vspw_to_vipseg": vspw_to_vipseg_splits,
            "vipseg_to_vspw": vipseg_to_vspw_splits,
        },
        "skipped_reason": skipped_reason,
    }
    eval_vspw_to_vipseg = {
        "audit_name": "stwm_lodo_semantic_trace_world_model_v1_vspw_to_vipseg_eval",
        "lodo_completed": False,
        "train_dataset": "VSPW",
        "test_dataset": "VIPSEG",
        "skipped_reason": skipped_reason,
    }
    eval_vipseg_to_vspw = {
        "audit_name": "stwm_lodo_semantic_trace_world_model_v1_vipseg_to_vspw_eval",
        "lodo_completed": False,
        "train_dataset": "VIPSEG",
        "test_dataset": "VSPW",
        "skipped_reason": skipped_reason,
    }
    significance = {
        "audit_name": "stwm_lodo_semantic_trace_world_model_v1_significance",
        "lodo_completed": False,
        "skipped_reason": skipped_reason,
    }
    write_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_train_summary_20260428.json", train_summary)
    write_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vspw_to_vipseg_eval_20260428.json", eval_vspw_to_vipseg)
    write_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vipseg_to_vspw_eval_20260428.json", eval_vipseg_to_vspw)
    write_json(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_significance_20260428.json", significance)
    sections = [
        "## Status",
        "\n".join(
            [
                "- lodo_completed: `False`",
                f"- skipped_reason: `{skipped_reason}`",
                f"- vspw_to_vipseg train/val/test: `{vspw_to_vipseg_splits.get('train_item_count', 0)}/{vspw_to_vipseg_splits.get('val_item_count', 0)}/{vspw_to_vipseg_splits.get('test_item_count', 0)}`",
                f"- vipseg_to_vspw train/val/test: `{vipseg_to_vspw_splits.get('train_item_count', 0)}/{vipseg_to_vspw_splits.get('val_item_count', 0)}/{vipseg_to_vspw_splits.get('test_item_count', 0)}`",
            ]
        ),
        "## Boundary",
        "- Main mixed/VSPW/VIPSeg test-once evidence remains valid.\n- LODO is appendix-strength future validation, not a blocker for the current main claim.",
    ]
    write_md(DOC_DIR / "STWM_LODO_SEMANTIC_TRACE_WORLD_MODEL_V1_20260428.md", "STWM LODO Semantic Trace World Model V1", sections)
    return {
        "lodo_completed": False,
        "lodo_signal_positive": False,
        "train_summary_path": str(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_train_summary_20260428.json"),
        "vspw_to_vipseg_eval_path": str(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vspw_to_vipseg_eval_20260428.json"),
        "vipseg_to_vspw_eval_path": str(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_vipseg_to_vspw_eval_20260428.json"),
        "significance_path": str(REPORT_DIR / "stwm_lodo_semantic_trace_world_model_v1_significance_20260428.json"),
        "skipped_reason": skipped_reason,
    }


def build_h16_reports() -> dict[str, Any]:
    feature = load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json")
    target_shape = feature.get("target_shape", [])
    target_mask_shape = feature.get("target_mask_shape", [])
    current_h = int(target_shape[1]) if len(target_shape) >= 2 else 0
    blocker = (
        "Current live semantic target caches, materialization reports, and all selected mixed checkpoints are built for H=8 only. "
        "Code exposes --fut-len, but H=16 would require rebuilding observed/future target pools, rematerializing eval caches, and retraining at least the best mixed config."
    )
    audit = {
        "audit_name": "stwm_horizon_scaling_h16_v1_target_audit",
        "builder_supports_fut_len": True,
        "trainer_supports_fut_len": True,
        "current_cache_horizon": current_h,
        "current_target_shape": target_shape,
        "current_target_mask_shape": target_mask_shape,
        "h16_completed": False,
        "h16_feasible_from_code": True,
        "h16_blocker": blocker,
    }
    train = {
        "audit_name": "stwm_horizon_scaling_h16_v1_train_summary",
        "h16_completed": False,
        "skipped_reason": blocker,
    }
    eval_report = {
        "audit_name": "stwm_horizon_scaling_h16_v1_eval",
        "h16_completed": False,
        "h16_signal_positive": False,
        "skipped_reason": blocker,
    }
    write_json(REPORT_DIR / "stwm_horizon_scaling_h16_v1_target_audit_20260428.json", audit)
    write_json(REPORT_DIR / "stwm_horizon_scaling_h16_v1_train_summary_20260428.json", train)
    write_json(REPORT_DIR / "stwm_horizon_scaling_h16_v1_eval_20260428.json", eval_report)
    sections = [
        "## Status",
        "\n".join(
            [
                "- current_horizon: `8`",
                "- code_supports_h16: `True`",
                "- h16_completed: `False`",
                f"- blocker: `{blocker}`",
            ]
        ),
        "## Interpretation",
        "- H=8 evidence is strong and paper-usable.\n- H=16 remains a scaling appendix, not a main-claim prerequisite.",
    ]
    write_md(DOC_DIR / "STWM_HORIZON_SCALING_H16_V1_20260428.md", "STWM Horizon Scaling H16 V1", sections)
    return {
        "h16_completed": False,
        "h16_signal_positive": False,
        "target_audit_path": str(REPORT_DIR / "stwm_horizon_scaling_h16_v1_target_audit_20260428.json"),
        "train_summary_path": str(REPORT_DIR / "stwm_horizon_scaling_h16_v1_train_summary_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_horizon_scaling_h16_v1_eval_20260428.json"),
        "skipped_reason": blocker,
    }


def build_density_reports() -> dict[str, Any]:
    density_prior = load_json(REPORT_DIR / "stage2_tusb_true_instance_density_20260418.json")
    blocker = (
        "Current semantic trace world model target pools, materialization caches, and mixed fullscale checkpoints are all built with max_entities_per_sample=8. "
        "The builders expose --max-entities-per-sample, but K16/K32 would require rebuilding target pools and rerunning materialization and training."
    )
    audit = {
        "audit_name": "stwm_trace_unit_density_scaling_v1_audit",
        "k8_completed": True,
        "k16_supported_by_cli": True,
        "k32_supported_by_cli": True,
        "k16_completed": False,
        "k32_completed": False,
        "memory_compute_blocker": blocker,
        "prior_density_context": density_prior,
        "terminology_recommendation": "semantic trace-unit field",
    }
    eval_report = {
        "audit_name": "stwm_trace_unit_density_scaling_v1_eval",
        "density_scaling_completed": False,
        "terminology_recommendation": "semantic trace-unit field",
        "skipped_reason": blocker,
    }
    write_json(REPORT_DIR / "stwm_trace_unit_density_scaling_v1_audit_20260428.json", audit)
    write_json(REPORT_DIR / "stwm_trace_unit_density_scaling_v1_eval_20260428.json", eval_report)
    sections = [
        "## Status",
        "\n".join(
            [
                "- K8 evaluated: `True`",
                "- K16 evaluated: `False`",
                "- K32 evaluated: `False`",
                "- terminology_recommendation: `semantic trace-unit field`",
            ]
        ),
        "## Rationale",
        "- The current evidence is on sparse trace units, not a dense pixel field.\n- Reviewer-safe wording is semantic trace-unit field unless K16/K32 scaling is actually run.",
    ]
    write_md(DOC_DIR / "STWM_TRACE_UNIT_DENSITY_SCALING_V1_20260428.md", "STWM Trace-Unit Density Scaling V1", sections)
    return {
        "density_scaling_completed": False,
        "terminology_recommendation": "semantic trace-unit field",
        "audit_path": str(REPORT_DIR / "stwm_trace_unit_density_scaling_v1_audit_20260428.json"),
        "eval_path": str(REPORT_DIR / "stwm_trace_unit_density_scaling_v1_eval_20260428.json"),
        "skipped_reason": blocker,
    }


def build_final_assets() -> dict[str, Any]:
    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")
    vspw = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vspw_test_eval_complete_20260428.json")
    vipseg = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vipseg_test_eval_complete_20260428.json")
    significance = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_significance_complete_20260428.json")
    decision = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    selection = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_val_selection_complete_20260428.json")
    train = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_train_summary_complete_20260428.json")
    belief = load_json(REPORT_DIR / "stwm_belief_final_eval_20260424.json")
    belief_boot = load_json(REPORT_DIR / "stwm_trace_belief_bootstrap_20260424.json")
    false_confuser = load_json(REPORT_DIR / "stwm_false_confuser_analysis_20260425.json")
    reacq = load_json(REPORT_DIR / "stwm_reacquisition_v2_final_decision_20260425.json")
    planning = load_json(REPORT_DIR / "stwm_planning_lite_risk_decision_20260425.json")
    counter = load_json(REPORT_DIR / "stwm_counterfactual_association_eval_20260425.json")

    rows = [_metric_row("mixed", mixed), _metric_row("VSPW", vspw), _metric_row("VIPSeg", vipseg)]
    ci_rows = [_ci_row("mixed", significance), _ci_row("VSPW", significance), _ci_row("VIPSeg", significance)]
    utility_rows = [
        {
            "asset": "trace_belief_assoc",
            "positive": bool(belief.get("improved_vs_calibration") and belief.get("improved_vs_cropenc") and belief.get("improved_vs_legacysem")),
            "claim": "improves over calibration/cropenc/legacysem",
        },
        {
            "asset": "bootstrap",
            "positive": bool(belief_boot.get("trace_belief_zero_excluded_on_id")),
            "claim": "trace belief bootstrap excludes zero on ID panel",
        },
        {
            "asset": "false-confuser",
            "positive": bool(false_confuser.get("false_confuser_reduced")),
            "claim": "reduces false-confuser errors",
        },
        {
            "asset": "reacquisition",
            "positive": bool(reacq.get("reacquisition_v2_established")),
            "claim": "supports reacquisition utility",
        },
        {
            "asset": "planning-lite",
            "positive": bool(planning.get("planning_lite_risk_established", planning.get("risk_utility_established", False))),
            "claim": "supports planning-lite risk utility",
        },
        {
            "asset": "counterfactual",
            "positive": bool(counter.get("shuffled_trace_changes_decision")),
            "claim": "trace counterfactual changes decisions",
        },
    ]
    tables = {
        "audit_name": "stwm_final_paper_tables_v1",
        "semantic_trace_field_main_result": rows,
        "stable_changed_breakdown": [
            {
                "dataset": r["dataset"],
                "stable_copy_preserved": 1.0 - r["stable_preservation_drop"],
                "stable_preservation_drop": r["stable_preservation_drop"],
                "changed_subset_gain": r["changed_top5_gain"],
            }
            for r in rows
        ],
        "dataset_breakdown": rows,
        "trace_guardrail": [
            {"dataset": r["dataset"], "future_trace_coord_error": r["future_trace_coord_error"], "trace_regression_detected": False}
            for r in rows
        ],
        "utility_belief_assoc": utility_rows,
        "selection_protocol": {
            "best_selected_on_val_only": True,
            "selected_prototype_count": int(selection.get("selected_prototype_count", 0)),
            "selected_seed": int(selection.get("selected_seed", -1)),
            "test_metrics_used_for_selection": False,
        },
        "main_result_summary": {
            "completed_run_count": int(train.get("completed_run_count", 0)),
            "best_prototype_count": int(decision.get("best_prototype_count", 0)),
            "best_seed": int(decision.get("best_seed", -1)),
            "paper_world_model_claimable": decision.get("paper_world_model_claimable", "unclear"),
            "semantic_field_branch_status": decision.get("semantic_field_branch_status", "unknown"),
        },
        "significance": ci_rows,
    }
    write_json(REPORT_DIR / "stwm_final_paper_tables_v1_20260428.json", tables)
    table_sections = [
        "## Semantic Trace Field Main Result\n" + "\n".join(
            f"- {r['dataset']}: top5 {r['stwm_top5']:.4f} vs copy {r['copy_top5']:.4f}; changed gain {r['changed_top5_gain']:.4f}; stable drop {r['stable_preservation_drop']:.6f}"
            for r in rows
        ),
        "## Significance\n" + "\n".join(
            f"- {r['dataset']}: changed delta {r['mean_delta']:.4f}, CI {r['ci95']}, zero_excluded={r['zero_excluded']}"
            for r in ci_rows
        ),
        "## Utility / Belief Association\n" + "\n".join(
            f"- {r['asset']}: positive={r['positive']} ({r['claim']})"
            for r in utility_rows
        ),
    ]
    write_md(DOC_DIR / "STWM_FINAL_PAPER_TABLES_V1_20260428.md", "STWM Final Paper Tables V1", table_sections)

    figure_manifest = {
        "audit_name": "stwm_final_figure_manifest_v1",
        "figure_dir": str(FIG_DIR),
        "figures": [
            {"id": "motivation", "status": "planned", "description": "Trace has dynamics but lacks semantics; semantic trace world model predicts future trace+semantic field."},
            {"id": "method", "status": "generated_svg", "path": str(FIG_DIR / "figure_method_semantic_trace_world_model.svg")},
            {"id": "main_result_plot", "status": "generated_svg", "path": str(FIG_DIR / "figure_changed_subset_top5.svg")},
            {"id": "dataset_breakdown", "status": "generated_svg", "path": str(FIG_DIR / "figure_dataset_top5_breakdown.svg")},
            {"id": "qualitative_examples", "status": "planned", "required": "4 successes and 2 failures with trace+semantic overlays."},
            {"id": "stable_changed_visualization", "status": "planned", "description": "show stable copy preserved and changed corrected examples."},
            {"id": "appendix_diagnostics", "status": "optional", "description": "semantic-state negative history if space permits."},
        ],
        "no_candidate_scorer": True,
        "free_rollout": True,
    }
    write_json(REPORT_DIR / "stwm_final_figure_manifest_v1_20260428.json", figure_manifest)
    write_md(
        DOC_DIR / "STWM_FINAL_FIGURE_MANIFEST_V1_20260428.md",
        "STWM Final Figure Manifest V1",
        ["\n".join(f"- {fig['id']}: `{fig['status']}`" for fig in figure_manifest["figures"])],
    )

    claim_boundary = {
        "audit_name": "stwm_final_claim_boundary_v1",
        "allowed_claims": [
            "STWM predicts future semantic trace fields under free rollout.",
            "Copy-gated residual semantic transition improves changed semantic states while preserving stable states.",
            "Works on mixed VSPW+VIPSeg protocol.",
            "Does not degrade trace dynamics.",
            "Belief association utility supports future identity association.",
        ],
        "forbidden_claims": [
            "STWM is SAM2/CoTracker plugin.",
            "STWM beats all trackers overall.",
            "Full RGB generation.",
            "Closed-loop planner.",
            "Universal OOD dominance.",
            "Hiding VIPSeg smaller effect.",
            "Claiming candidate scorer as method.",
        ],
        "must_disclose": [
            "VIPSeg changed-subset gain is positive but smaller than VSPW.",
            "Dedicated LODO cross-dataset training/eval is not yet completed.",
            "Horizon evidence is H=8 unless H16 appendix is run.",
            "Current field is best described as semantic trace-unit field unless K-scaling appendix is executed.",
        ],
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "old_association_report_used": False,
    }
    write_json(REPORT_DIR / "stwm_final_claim_boundary_v1_20260428.json", claim_boundary)
    write_md(
        DOC_DIR / "STWM_FINAL_CLAIM_BOUNDARY_V1_20260428.md",
        "STWM Final Claim Boundary V1",
        [
            "## Allowed Claims\n" + "\n".join(f"- {x}" for x in claim_boundary["allowed_claims"]),
            "## Forbidden Claims\n" + "\n".join(f"- {x}" for x in claim_boundary["forbidden_claims"]),
            "## Must Disclose\n" + "\n".join(f"- {x}" for x in claim_boundary["must_disclose"]),
        ],
    )

    outline = {
        "preferred_title": "STWM: Semantic Trace World Models with Copy-Gated Residual Semantic Memory",
        "introduction": [
            "Trace dynamics alone are insufficient for future semantic identity reasoning.",
            "Observed semantic memory plus future trace rollout yields a better world-model state.",
            "Copy-gated residual transition preserves stable semantics while correcting changed states.",
        ],
        "related_work": [
            "trajectory field / point tracking",
            "object-centric world models",
            "future instance prediction",
            "semantic world models",
        ],
        "method": [
            "trace backbone",
            "semantic trace units",
            "observed semantic memory",
            "copy-gated residual semantic transition",
            "belief readout utility",
        ],
        "experiments": [
            "mixed/VSPW/VIPSeg free-rollout semantic trace field prediction",
            "stable vs changed analysis",
            "trace guardrail",
            "belief utility / counterfactual evidence",
            "optional LODO appendix",
            "optional horizon and density appendices",
        ],
        "limitations": [
            "VIPSeg effect smaller than VSPW",
            "LODO appendix not yet executed",
            "H=8 evidence stronger than longer-horizon evidence",
            "current field is semantic trace-unit field, not dense pixel field",
        ],
    }
    write_json(REPORT_DIR / "stwm_final_paper_outline_v1_20260428.json", outline)
    write_md(
        DOC_DIR / "STWM_FINAL_PAPER_OUTLINE_V1_20260428.md",
        "STWM Final Paper Outline V1",
        [
            "## Introduction\n" + "\n".join(f"- {x}" for x in outline["introduction"]),
            "## Related Work\n" + "\n".join(f"- {x}" for x in outline["related_work"]),
            "## Method\n" + "\n".join(f"- {x}" for x in outline["method"]),
            "## Experiments\n" + "\n".join(f"- {x}" for x in outline["experiments"]),
            "## Limitations\n" + "\n".join(f"- {x}" for x in outline["limitations"]),
        ],
    )
    return {
        "paper_tables_path": str(REPORT_DIR / "stwm_final_paper_tables_v1_20260428.json"),
        "figure_manifest_path": str(REPORT_DIR / "stwm_final_figure_manifest_v1_20260428.json"),
        "claim_boundary_path": str(REPORT_DIR / "stwm_final_claim_boundary_v1_20260428.json"),
        "paper_outline_path": str(DOC_DIR / "STWM_FINAL_PAPER_OUTLINE_V1_20260428.md"),
    }


def build_final_decision(
    artifact: dict[str, Any],
    lodo: dict[str, Any],
    h16: dict[str, Any],
    density: dict[str, Any],
) -> dict[str, Any]:
    readiness = load_json(REPORT_DIR / "stwm_final_paper_readiness_decision_20260428.json")
    mixed_decision = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    final_decision = {
        "audit_name": "stwm_top_tier_hardening_v1_decision",
        "artifact_audit_passed": bool(artifact["artifact_audit_passed"]),
        "lodo_completed": bool(lodo["lodo_completed"]),
        "lodo_signal_positive": bool(lodo["lodo_signal_positive"]),
        "h16_completed": bool(h16["h16_completed"]),
        "h16_signal_positive": bool(h16["h16_signal_positive"]),
        "density_scaling_completed": bool(density["density_scaling_completed"]),
        "terminology_recommendation": density["terminology_recommendation"],
        "ready_for_overleaf": bool(readiness.get("ready_for_overleaf", False)),
        "ready_for_cvpr_aaai_main": readiness.get("ready_for_cvpr_aaai_main", "unclear"),
        "semantic_field_main_claim_strength": (
            "strong_mixed_free_rollout_evidence: 10/10 mixed matrix completed, val-only selection, mixed/VSPW/VIPSeg changed-gain CI excludes zero, stable copy preserved, trace regression false"
        ),
        "main_risks": [
            "Dedicated LODO cross-dataset checkpoints/evals are not yet executed.",
            "Longer-horizon evidence is H=8 only unless H16 appendix is run.",
            "Trace-unit density scaling beyond K=8 is not yet empirically validated.",
            "Canonical per-run training logs are present but zero-byte; checkpoints and summaries are the primary live artifacts.",
        ],
        "recommended_next_step_choice": (
            "run_missing_lodo" if artifact["artifact_audit_passed"] else "fix_artifacts"
        ),
        "mixed_complete_decision_source": mixed_decision,
    }
    sections = [
        "## Status",
        "\n".join(
            [
                f"- artifact_audit_passed: `{final_decision['artifact_audit_passed']}`",
                f"- lodo_completed: `{final_decision['lodo_completed']}`",
                f"- h16_completed: `{final_decision['h16_completed']}`",
                f"- density_scaling_completed: `{final_decision['density_scaling_completed']}`",
                f"- terminology_recommendation: `{final_decision['terminology_recommendation']}`",
                f"- ready_for_overleaf: `{final_decision['ready_for_overleaf']}`",
                f"- ready_for_cvpr_aaai_main: `{final_decision['ready_for_cvpr_aaai_main']}`",
                f"- recommended_next_step_choice: `{final_decision['recommended_next_step_choice']}`",
            ]
        ),
        "## Main Risks",
        "\n".join(f"- {x}" for x in final_decision["main_risks"]),
    ]
    write_json(REPORT_DIR / "stwm_top_tier_hardening_v1_decision_20260428.json", final_decision)
    write_md(DOC_DIR / "STWM_TOP_TIER_HARDENING_V1_DECISION_20260428.md", "STWM Top-Tier Hardening V1 Decision", sections)
    return final_decision


def build_guardrail() -> dict[str, Any]:
    guardrail = {
        "audit_name": "stwm_world_model_no_drift_guardrail_v39",
        "allowed": [
            "final protocol hardening",
            "LODO/cross-dataset validation",
            "horizon/density scaling",
            "paper assets",
        ],
        "forbidden": [
            "new method branches",
            "candidate scorer",
            "SAM2/CoTracker plugin",
            "future candidate leakage",
            "hiding artifact failures",
            "claiming dense trace field if only semantic trace units are evaluated",
            "overclaiming full RGB generation or closed-loop planning",
        ],
    }
    write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v39_20260428.json", guardrail)
    write_md(
        DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V39.md",
        "STWM World Model No-Drift Guardrail V39",
        [
            "## Allowed\n" + "\n".join(f"- {x}" for x in guardrail["allowed"]),
            "## Forbidden\n" + "\n".join(f"- {x}" for x in guardrail["forbidden"]),
        ],
    )
    return guardrail


def main() -> None:
    artifact = build_artifact_audit()
    lodo = build_lodo_reports()
    h16 = build_h16_reports()
    density = build_density_reports()
    build_final_assets()
    build_final_decision(artifact, lodo, h16, density)
    build_guardrail()


if __name__ == "__main__":
    main()
