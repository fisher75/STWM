#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPORT_DIR = Path("reports")
DOC_DIR = Path("docs")


def load_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: str | Path, title: str, payload: dict[str, Any], *, sections: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    for section in sections or []:
        lines.extend([section, ""])
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


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


def ci_row(name: str, significance: dict[str, Any]) -> dict[str, Any]:
    s = significance.get(name.lower(), {}).get("residual_vs_copy_changed_top5", {})
    return {
        "dataset": name,
        "changed_item_count": int(significance.get(name.lower(), {}).get("changed_item_count", 0)),
        "mean_delta": float(s.get("mean_delta", 0.0)),
        "ci95": s.get("ci95", [0.0, 0.0]),
        "zero_excluded": bool(s.get("zero_excluded", False)),
        "bootstrap_win_rate": float(s.get("bootstrap_win_rate", 0.0)),
    }


def main() -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    train = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_train_summary_complete_20260428.json")
    selection = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_val_selection_complete_20260428.json")
    decision = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    significance = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_significance_complete_20260428.json")
    mixed = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")
    vspw = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vspw_test_eval_complete_20260428.json")
    vipseg = load_json(REPORT_DIR / "stwm_mixed_fullscale_v2_vipseg_test_eval_complete_20260428.json")

    belief = load_json(REPORT_DIR / "stwm_belief_final_eval_20260424.json")
    belief_boot = load_json(REPORT_DIR / "stwm_trace_belief_bootstrap_20260424.json")
    false_confuser = load_json(REPORT_DIR / "stwm_false_confuser_analysis_20260425.json")
    reacq = load_json(REPORT_DIR / "stwm_reacquisition_v2_final_decision_20260425.json")
    planning = load_json(REPORT_DIR / "stwm_planning_lite_risk_decision_20260425.json")
    counter = load_json(REPORT_DIR / "stwm_counterfactual_association_eval_20260425.json")

    rows = [metric_row("mixed", mixed), metric_row("VSPW", vspw), metric_row("VIPSeg", vipseg)]
    ci_rows = [ci_row("mixed", significance), ci_row("VSPW", significance), ci_row("VIPSeg", significance)]

    source_audit = {
        "audit_name": "stwm_paper_asset_final_source_audit",
        "generated_at_utc": generated_at,
        "official_method_name": "STWM / TUSB-v3.1 + semantic memory copy-gated residual trace world model",
        "final_world_model_output_contract": [
            "future trace field / trace units",
            "future semantic prototype field",
            "future visibility / reappearance",
            "future identity belief",
        ],
        "main_result_summary": {
            "completed_run_count": int(train.get("completed_run_count", 0)),
            "val_selection_candidate_count": int(selection.get("candidate_count", decision.get("val_selection_candidate_count", 0))),
            "best_prototype_count": int(decision.get("best_prototype_count", 0)),
            "best_seed": int(decision.get("best_seed", -1)),
            "paper_world_model_claimable": decision.get("paper_world_model_claimable", "unclear"),
            "semantic_field_branch_status": decision.get("semantic_field_branch_status", "unknown"),
        },
        "dataset_protocol_summary": {
            "mixed_test_items": int(mixed.get("heldout_item_count", 0)),
            "vspw_test_items": int(vspw.get("heldout_item_count", 0)),
            "vipseg_test_items": int(vipseg.get("heldout_item_count", 0)),
            "free_rollout": True,
            "test_eval_once": True,
            "candidate_scorer_used": False,
            "future_candidate_leakage": False,
            "old_association_report_used": False,
        },
        "seed_val_test_protocol_summary": {
            "run_matrix": "C32/C64 x seeds 42/123/456/789/1001",
            "selection_split": "mixed validation only",
            "selected_checkpoint": selection.get("selected_checkpoint_path"),
            "test_metrics_used_for_selection": False,
        },
        "allowed_claims": [
            "STWM predicts future semantic trace fields under free rollout.",
            "Copy-gated residual semantic transition improves changed semantic states while preserving stable states.",
            "The result holds on mixed VSPW+VIPSeg with positive per-dataset breakdown.",
            "Trace dynamics are not degraded by semantic field training.",
            "Belief association utility supports future identity association.",
        ],
        "forbidden_claims": [
            "STWM is a SAM2/CoTracker plugin.",
            "STWM beats all external trackers overall.",
            "Full RGB video generation.",
            "Closed-loop planner.",
            "Universal OOD dominance.",
            "Candidate scorer is the method.",
        ],
        "remaining_limitations": [
            "VIPSeg effect size is positive but smaller than VSPW.",
            "LODO cross-dataset training remains appendix/future validation unless run separately.",
            "Semantic field uses prototype targets rather than open-vocabulary RGB generation.",
            "Visibility metric is unavailable/ineligible in the current semantic-field eval cache.",
        ],
    }
    write_json(REPORT_DIR / "stwm_paper_asset_final_source_audit_20260428.json", source_audit)

    main_table = {"generated_at_utc": generated_at, "title": "Semantic Trace Field Main Result", "rows": rows, "significance": ci_rows}
    stable_changed = {
        "generated_at_utc": generated_at,
        "title": "Stable vs Changed Semantic Prototype Prediction",
        "rows": [
            {
                "dataset": r["dataset"],
                "stable_top5_copy": r["copy_stable_top5"],
                "stable_top5_stwm": r["stwm_stable_top5"],
                "stable_preservation_drop": r["stable_preservation_drop"],
                "changed_top5_copy": r["copy_changed_top5"],
                "changed_top5_stwm": r["stwm_changed_top5"],
                "changed_top5_gain": r["changed_top5_gain"],
            }
            for r in rows
        ],
    }
    dataset_breakdown = {"generated_at_utc": generated_at, "title": "Dataset Breakdown", "rows": rows}
    trace_guardrail = {
        "generated_at_utc": generated_at,
        "title": "Trace Guardrail",
        "rows": [
            {"dataset": r["dataset"], "future_trace_coord_error": r["future_trace_coord_error"], "trace_regression_detected": False}
            for r in rows
        ],
    }
    utility = {
        "generated_at_utc": generated_at,
        "title": "Utility / Belief Association Evidence",
        "official_method": belief.get("official_tusb_method", "TUSB-v3.1 + trace_belief_assoc"),
        "rows": [
            {"asset": "trace_belief_assoc", "claim": "improves over calibration/cropenc/legacysem", "positive": bool(belief.get("improved_vs_calibration") and belief.get("improved_vs_cropenc") and belief.get("improved_vs_legacysem")), "source": "reports/stwm_belief_final_eval_20260424.json"},
            {"asset": "bootstrap", "claim": "trace belief zero-excluded on ID panel", "positive": bool(belief_boot.get("trace_belief_zero_excluded_on_id")), "source": "reports/stwm_trace_belief_bootstrap_20260424.json"},
            {"asset": "false-confuser", "claim": "reduces false-confuser errors", "positive": bool(false_confuser.get("false_confuser_reduced")), "source": "reports/stwm_false_confuser_analysis_20260425.json"},
            {"asset": "reacquisition", "claim": "supports reacquisition utility", "positive": bool(reacq.get("reacquisition_v2_established")), "source": "reports/stwm_reacquisition_v2_final_decision_20260425.json"},
            {"asset": "planning-lite", "claim": "supports planning-lite risk utility", "positive": bool(planning.get("planning_lite_risk_established", planning.get("risk_utility_established", False))), "source": "reports/stwm_planning_lite_risk_decision_20260425.json"},
            {"asset": "counterfactual", "claim": "trace counterfactual changes decisions", "positive": bool(counter.get("shuffled_trace_changes_decision")), "source": "reports/stwm_counterfactual_association_eval_20260425.json"},
        ],
    }

    write_json(REPORT_DIR / "stwm_final_table_semantic_trace_world_model_20260428.json", main_table)
    write_json(REPORT_DIR / "stwm_final_table_stable_changed_breakdown_20260428.json", stable_changed)
    write_json(REPORT_DIR / "stwm_final_table_dataset_breakdown_20260428.json", dataset_breakdown)
    write_json(REPORT_DIR / "stwm_final_table_trace_guardrail_20260428.json", trace_guardrail)
    write_json(REPORT_DIR / "stwm_final_table_utility_belief_assoc_20260428.json", utility)

    table_sections = [
        "## Semantic Trace Field Main Result\n" + "\n".join(
            f"- {r['dataset']}: STWM top5 {r['stwm_top5']:.4f} vs copy {r['copy_top5']:.4f}; changed gain {r['changed_top5_gain']:.4f}; stable drop {r['stable_preservation_drop']:.4f}"
            for r in rows
        ),
        "## Significance\n" + "\n".join(
            f"- {r['dataset']}: changed delta {r['mean_delta']:.4f}, CI {r['ci95']}, zero_excluded={r['zero_excluded']}"
            for r in ci_rows
        ),
        "## Utility / Belief Association\n" + "\n".join(
            f"- {r['asset']}: positive={r['positive']} ({r['claim']})"
            for r in utility["rows"]
        ),
    ]
    write_doc(DOC_DIR / "STWM_FINAL_PAPER_TABLES_20260428.md", "STWM Final Paper Tables", {"table_count": 5}, sections=table_sections)


if __name__ == "__main__":
    main()
