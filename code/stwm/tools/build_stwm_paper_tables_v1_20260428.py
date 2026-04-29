#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def metric_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "label",
        "method",
        "scoring_mode",
        "overall_top1",
        "hard_subset_top1",
        "ambiguity_top1",
        "appearance_change_top1",
        "occlusion_reappearance_top1",
        "long_gap_persistence_top1",
        "MRR",
        "top5_hit",
        "source",
        "source_report",
        "panel",
    ]
    return {k: row.get(k) for k in keys if k in row}


def external_row(name: str, overall: dict[str, Any], subsets: dict[str, Any]) -> dict[str, Any]:
    sub = subsets.get(name, {}) if isinstance(subsets, dict) else {}
    return {
        "method": name,
        "count": overall.get("count"),
        "overall_top1": overall.get("top1"),
        "MRR": overall.get("MRR"),
        "false_confuser_rate": overall.get("false_confuser_rate"),
        "occlusion_top1": sub.get("occlusion_reappearance_top1"),
        "long_gap_top1": sub.get("long_gap_persistence_top1"),
        "crossing_ambiguity_top1": sub.get("crossing_ambiguity_top1"),
        "OOD_hard_top1": sub.get("OOD_hard_top1"),
        "source": overall.get("source", "external/item-aligned summary"),
    }


def build(repo_root: Path) -> dict[str, Any]:
    reports = repo_root / "reports"
    docs = repo_root / "docs"
    paper_ready = load_json(reports / "stwm_paper_ready_tables_20260425.json", {})
    final_tables = load_json(reports / "stwm_final_paper_tables_20260425.json", {})
    false_confuser = load_json(reports / "stwm_false_confuser_analysis_20260425.json", {})
    reacq = load_json(reports / "stwm_reacquisition_utility_eval_20260425.json", {})
    reacq_decision = load_json(reports / "stwm_reacquisition_utility_decision_20260425.json", {})
    risk = load_json(reports / "stwm_planning_lite_risk_eval_20260425.json", {})
    risk_decision = load_json(reports / "stwm_planning_lite_risk_decision_20260425.json", {})
    cfa = load_json(reports / "stwm_counterfactual_association_eval_20260425.json", {})
    cfr = load_json(reports / "stwm_counterfactual_reacquisition_utility_20260425.json", {})
    cfp = load_json(reports / "stwm_counterfactual_planning_lite_risk_20260425.json", {})
    external = load_json(reports / "stwm_external_baseline_full_eval_summary_20260426.json", {})
    external_decision = load_json(reports / "stwm_external_baseline_full_eval_decision_20260426.json", {})

    main_rows = [metric_row(r) for r in paper_ready.get("table_1_main_comparison", [])]
    readout_rows = [metric_row(r) for r in paper_ready.get("table_2_readout_ablation", [])]

    utility_rows: list[dict[str, Any]] = []
    fc_overall = false_confuser.get("overall", {}) if isinstance(false_confuser, dict) else {}
    fc_id = false_confuser.get("id_summary", {}) if isinstance(false_confuser, dict) else {}
    utility_rows.append(
        {
            "utility": "false_confuser_analysis",
            "source_report": "reports/stwm_false_confuser_analysis_20260425.json",
            "overall_belief_top1": fc_overall.get("belief_top1"),
            "overall_teacher_top1": fc_overall.get("teacher_top1"),
            "overall_false_confuser_delta_belief_minus_teacher": fc_overall.get("false_confuser_delta_belief_minus_teacher"),
            "id_false_confuser_delta_belief_minus_teacher": fc_id.get("false_confuser_delta_belief_minus_teacher"),
            "claim_ready": bool(false_confuser.get("false_confuser_reduced")),
        }
    )
    for name, row in (reacq.get("methods") or {}).items():
        if name in {"STWM trace_belief_assoc", "legacysem", "frozen_external_teacher_only", "calibration-only", "cropenc"}:
            utility_rows.append(
                {
                    "utility": "reacquisition",
                    "method": name,
                    "source_report": "reports/stwm_reacquisition_utility_eval_20260425.json",
                    "count": row.get("count"),
                    "top1": row.get("top1"),
                    "MRR": row.get("MRR"),
                    "false_reacquisition_rate": row.get("false_reacquisition_rate"),
                    "claim_ready": bool(reacq_decision.get("supports_main_paper_utility_claim")),
                }
            )
    for name, row in (risk.get("methods") or {}).items():
        if name in {"STWM trace_belief_assoc risk", "legacysem risk", "frozen_external_teacher_only risk", "calibration-only risk", "cropenc risk"}:
            utility_rows.append(
                {
                    "utility": "planning_lite_risk",
                    "method": name,
                    "source_report": "reports/stwm_planning_lite_risk_eval_20260425.json",
                    "count": row.get("count"),
                    "risk_AUC": row.get("risk_AUC"),
                    "false_safe_rate": row.get("false_safe_rate"),
                    "false_alarm_rate": row.get("false_alarm_rate"),
                    "top1_safe_path_selection_accuracy": row.get("top1_safe_path_selection_accuracy"),
                    "claim_ready": bool(risk_decision.get("can_enter_main_paper")),
                    "claim_boundary": risk_decision.get("allowed_claim_wording"),
                }
            )
    utility_rows.append(
        {
            "utility": "counterfactual_association",
            "source_report": "reports/stwm_counterfactual_association_eval_20260425.json",
            "original_top1": ((cfa.get("original") or {}).get("full_trace_belief") or {}).get("top1"),
            "shuffled_trace_top1": ((cfa.get("counterfactual") or {}).get("shuffled_trace") or {}).get("top1"),
            "shuffled_trace_changes_decision": cfa.get("shuffled_trace_changes_decision"),
        }
    )
    utility_rows.append(
        {
            "utility": "counterfactual_reacquisition",
            "source_report": "reports/stwm_counterfactual_reacquisition_utility_20260425.json",
            "original_top1": ((cfr.get("original") or {}).get("full_trace_belief") or {}).get("top1"),
            "shuffled_trace_top1": ((cfr.get("counterfactual") or {}).get("shuffled_trace") or {}).get("top1"),
            "trace_prior_load_bearing_counterfactual": cfr.get("trace_prior_load_bearing_counterfactual"),
        }
    )
    utility_rows.append(
        {
            "utility": "counterfactual_planning_lite_risk",
            "source_report": "reports/stwm_counterfactual_planning_lite_risk_20260425.json",
            "original_risk_AUC": ((cfp.get("original_risk_score") or {})).get("risk_AUC"),
            "object_removed_risk_AUC": ((cfp.get("object_removed_risk") or {})).get("risk_AUC"),
            "object_shifted_risk_AUC": ((cfp.get("object_shifted_risk") or {})).get("risk_AUC"),
            "no_closed_loop_claim": cfp.get("no_closed_loop_claim"),
        }
    )

    overall = external.get("per_method_overall", {}) if isinstance(external, dict) else {}
    subsets = external.get("per_method_per_subset", {}) if isinstance(external, dict) else {}
    external_rows = []
    for name in ["stwm_trace_belief_assoc", "sam2", "cotracker", "cutie", "legacysem", "calibration-only", "cropenc"]:
        if name in overall:
            external_rows.append(external_row(name, overall[name], subsets))
    external_table = {
        "protocol": "389-item hard-case diagnostic item-aligned protocol; not a full-video benchmark",
        "rows": external_rows,
        "strongest_external_baseline": external_decision.get("strongest_external_baseline"),
        "stwm_overall_external_sota": bool(external_decision.get("stwm_overall_external_sota")),
        "stwm_significantly_improved_vs_cotracker_on_continuity_subsets": external_decision.get(
            "stwm_significantly_improved_vs_cotracker_on_continuity_subsets"
        ),
        "stwm_significantly_improved_vs_sam2_on_continuity_subsets": external_decision.get(
            "stwm_significantly_improved_vs_sam2_on_continuity_subsets"
        ),
        "boundary_note": "Use as external boundary evidence. Do not claim STWM beats SAM2 overall or is external overall SOTA.",
    }

    payloads = {
        "main": {
            "generated_at_utc": now_iso(),
            "title": "Main comparison",
            "source": "reports/stwm_paper_ready_tables_20260425.json",
            "rows": main_rows,
        },
        "readout": {
            "generated_at_utc": now_iso(),
            "title": "Readout ablation",
            "source": "reports/stwm_paper_ready_tables_20260425.json",
            "rows": readout_rows,
        },
        "utility": {
            "generated_at_utc": now_iso(),
            "title": "Mechanism and utility evidence",
            "sources": [
                "reports/stwm_false_confuser_analysis_20260425.json",
                "reports/stwm_reacquisition_utility_eval_20260425.json",
                "reports/stwm_planning_lite_risk_eval_20260425.json",
                "reports/stwm_counterfactual_association_eval_20260425.json",
                "reports/stwm_counterfactual_reacquisition_utility_20260425.json",
                "reports/stwm_counterfactual_planning_lite_risk_20260425.json",
            ],
            "mechanism_summary": final_tables.get("table_3_mechanism_evidence") or paper_ready.get("table_3_mechanism_summary"),
            "rows": utility_rows,
        },
        "external": {
            "generated_at_utc": now_iso(),
            "title": "External boundary diagnostic",
            "source": "reports/stwm_external_baseline_full_eval_summary_20260426.json",
            **external_table,
        },
    }
    write_json(reports / "stwm_paper_table_main_comparison_v1_20260428.json", payloads["main"])
    write_json(reports / "stwm_paper_table_readout_ablation_v1_20260428.json", payloads["readout"])
    write_json(reports / "stwm_paper_table_mechanism_utility_v1_20260428.json", payloads["utility"])
    write_json(reports / "stwm_paper_table_external_boundary_v1_20260428.json", payloads["external"])

    doc_lines = [
        "# STWM Paper Tables V1",
        "",
        "## Main Comparison",
        "",
        "| Method | Top1 | MRR | Occ. | Long-gap | Ambiguity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        doc_lines.append(
            f"| {row.get('label')} | {row.get('overall_top1')} | {row.get('MRR')} | "
            f"{row.get('occlusion_reappearance_top1')} | {row.get('long_gap_persistence_top1')} | {row.get('ambiguity_top1')} |"
        )
    doc_lines += [
        "",
        "## Readout Ablation",
        "",
        "| Readout | Top1 | MRR | Occ. | Long-gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in readout_rows:
        doc_lines.append(
            f"| {row.get('label')} | {row.get('overall_top1')} | {row.get('MRR')} | "
            f"{row.get('occlusion_reappearance_top1')} | {row.get('long_gap_persistence_top1')} |"
        )
    doc_lines += [
        "",
        "## External Boundary",
        "",
        f"- strongest_external_baseline: `{external_table['strongest_external_baseline']}`",
        f"- stwm_overall_external_sota: `{external_table['stwm_overall_external_sota']}`",
        f"- protocol: {external_table['protocol']}",
        "",
        "| Method | Top1 | MRR | false-confuser | Occ. | Long-gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in external_rows:
        doc_lines.append(
            f"| {row.get('method')} | {row.get('overall_top1')} | {row.get('MRR')} | {row.get('false_confuser_rate')} | "
            f"{row.get('occlusion_top1')} | {row.get('long_gap_top1')} |"
        )
    (docs / "STWM_PAPER_TABLES_V1_20260428.md").write_text("\n".join(doc_lines).rstrip() + "\n")
    return payloads


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--repo-root", default=os.environ.get("STWM_ROOT", "."))
    args = parser.parse_args()
    build(Path(args.repo_root).expanduser().resolve())


if __name__ == "__main__":
    main()
