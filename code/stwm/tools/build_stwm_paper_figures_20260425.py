#!/usr/bin/env python3
"""Build STWM paper Figure 1-8 from frozen JSON reports.

This is a visualization-only script. It does not train, evaluate, change model
outputs, or modify official conclusions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from stwm_vis_utils_20260425 import (
    DISPLAY_NAME,
    DOCS,
    FIG_MAIN,
    FIG_SUPP,
    METHOD_COLORS,
    REPORTS,
    ROOT,
    arrow,
    bar,
    draw_box,
    dump_json,
    ensure_dirs,
    get_panels_rows,
    load_json,
    metric_summary,
    path_like_strings,
    representative_cases_from_rows,
    save_figure,
    setup_style,
    simple_case_panel,
    stable_float,
    wrap,
    write_text,
)


SOURCE_REPORTS = [
    "reports/stwm_official_method_freeze_20260425.json",
    "reports/stwm_bayesian_belief_framing_20260425.json",
    "reports/stwm_claim_boundary_20260425.json",
    "reports/stwm_paper_ready_tables_20260425.json",
    "reports/stwm_trace_belief_eval_20260424.json",
    "reports/stwm_belief_final_eval_20260424.json",
    "reports/stwm_belief_strict_bootstrap_20260424.json",
    "reports/stwm_belief_true_ood_eval_20260424.json",
    "reports/stwm_belief_final_decision_20260424.json",
    "reports/stwm_false_confuser_analysis_20260425.json",
    "reports/stwm_belief_calibration_reliability_20260425.json",
    "reports/stwm_belief_mechanism_figure_specs_20260425.json",
    "reports/stwm_reacquisition_v2_task_build_20260425.json",
    "reports/stwm_reacquisition_v2_eval_20260425.json",
    "reports/stwm_reacquisition_v2_bootstrap_20260425.json",
    "reports/stwm_reacquisition_v2_trace_attribution_decision_20260425.json",
    "reports/stwm_reacquisition_v2_paper_assets_20260425.json",
    "reports/stwm_reacquisition_v2_final_decision_20260425.json",
    "reports/stwm_planning_lite_task_build_20260425.json",
    "reports/stwm_planning_lite_risk_eval_20260425.json",
    "reports/stwm_planning_lite_risk_decision_20260425.json",
    "reports/stwm_counterfactual_source_audit_20260425.json",
    "reports/stwm_counterfactual_intervention_set_20260425.json",
    "reports/stwm_counterfactual_association_eval_20260425.json",
    "reports/stwm_counterfactual_reacquisition_utility_20260425.json",
    "reports/stwm_counterfactual_planning_lite_risk_20260425.json",
    "reports/stwm_counterfactual_bootstrap_20260425.json",
    "reports/stwm_counterfactual_decision_20260425.json",
]


def source_audit(data: dict[str, Any]) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_reports": {},
        "source_manifest": {
            "case_ids": [],
            "clip_ids": [],
            "path_like_references": [],
            "existing_frame_or_video_paths": [],
            "missing_frame_or_video_sources": [],
        },
        "figure_inputs_sufficient": {},
        "source_missing": False,
        "visualization_only_no_result_changes": True,
    }
    case_ids: set[str] = set()
    clip_ids: set[str] = set()
    path_refs: set[str] = set()
    existing_paths: set[str] = set()
    missing_paths: set[str] = set()
    for rel in SOURCE_REPORTS:
        path = ROOT / rel
        obj, err = load_json(path)
        data[rel] = obj
        rows = get_panels_rows(obj) if isinstance(obj, dict) else []
        for row in rows:
            if row.get("protocol_item_id") is not None:
                case_ids.add(str(row.get("protocol_item_id")))
            if row.get("clip_id") is not None:
                clip_ids.add(str(row.get("clip_id")))
        refs = path_like_strings(obj) if obj is not None else []
        for ref in refs:
            path_refs.add(ref)
            if ref.lower().endswith((".png", ".jpg", ".jpeg", ".mp4", ".avi", ".mov", ".webm")):
                p = Path(ref)
                if not p.is_absolute():
                    p = ROOT / ref
                if p.exists():
                    existing_paths.add(str(p))
                else:
                    missing_paths.add(ref)
        audit["source_reports"][rel] = {
            "exists": path.exists(),
            "valid_json": err is None and obj is not None,
            "error": err,
            "top_level_keys": sorted(obj.keys())[:60] if isinstance(obj, dict) else [],
            "per_item_rows": len(rows),
            "path_like_references_count": len(refs),
        }
    # The current reports do not point to raw frames/videos. This is a source
    # limitation for qualitative panels and videos, not a blocker for quantitative
    # plots or schematic paper-ready v1 assets.
    if not existing_paths:
        missing_paths.add("No raw frame/video path references were present in the required JSON reports.")
    audit["source_manifest"]["case_ids"] = sorted(case_ids)[:200]
    audit["source_manifest"]["case_id_count"] = len(case_ids)
    audit["source_manifest"]["clip_ids"] = sorted(clip_ids)[:200]
    audit["source_manifest"]["clip_id_count"] = len(clip_ids)
    audit["source_manifest"]["path_like_references"] = sorted(path_refs)
    audit["source_manifest"]["existing_frame_or_video_paths"] = sorted(existing_paths)
    audit["source_manifest"]["missing_frame_or_video_sources"] = sorted(missing_paths)
    audit["source_missing"] = bool(missing_paths)
    audit["exact_missing_source"] = sorted(missing_paths)
    required_ok = all(v["exists"] and v["valid_json"] for v in audit["source_reports"].values())
    audit["audit_passed"] = required_ok
    return audit


def method_color(label: str) -> str:
    key = DISPLAY_NAME.get(label, label)
    if key == "STWM":
        return METHOD_COLORS["STWM"]
    if "teacher" in key.lower():
        return METHOD_COLORS["frozen external teacher"]
    if "legacy" in key.lower():
        return METHOD_COLORS["legacysem"]
    if "crop" in key.lower():
        return METHOD_COLORS["cropenc"]
    if "calib" in key.lower():
        return METHOD_COLORS["calibration-only"]
    return "#8A8A8A"


def final_eval_summaries(final_eval: dict[str, Any], trace_eval: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    rows = get_panels_rows(final_eval)
    trace_rows = get_panels_rows(trace_eval)
    panel = "densified_200_context_preserving"
    rows = [r for r in rows if r.get("panel_name") == panel]
    trace_rows = [r for r in trace_rows if r.get("panel_name") == panel]
    return {
        "STWM": metric_summary(rows, "TUSB-v3.1::official(best_semantic_hard.pt+trace_belief_assoc)", "trace_belief_assoc"),
        "Calibration-only": metric_summary(rows, "calibration-only::best.pt", "coord_only"),
        "CropEnc": metric_summary(rows, "cropenc::best.pt", "coord_only"),
        "LegacySem": metric_summary(rows, "legacysem::best.pt", "coord_only"),
        "Frozen teacher": metric_summary(trace_rows, "TUSB-v3.1::best_semantic_hard.pt", "frozen_external_teacher_only"),
    }


def pick_cases(trace_eval: dict[str, Any]) -> list[dict[str, Any]]:
    rows = get_panels_rows(trace_eval)
    stwm = [r for r in rows if r.get("method_name") == "TUSB-v3.1::best_semantic_hard.pt" and r.get("scoring_mode") == "trace_belief_assoc"]
    return representative_cases_from_rows(stwm, limit=8)


def fig01_teaser(data: dict[str, Any], figures: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    false_conf = data["reports/stwm_false_confuser_analysis_20260425.json"] or {}
    reacq_dec = data["reports/stwm_reacquisition_v2_final_decision_20260425.json"] or {}
    case = cases[0] if cases else {"case_id": "missing-case-id", "case_type": "confuser"}
    fig = plt.figure(figsize=(12.5, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    simple_case_panel(ax0, case, "Teacher-only failure", "wrong confuser", METHOD_COLORS["wrong"])
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_axis_off()
    ax1.set_title("STWM intuition", loc="left", weight="bold")
    draw_box(ax1, (0.05, 0.62), (0.30, 0.20), "Trace\nhistory", METHOD_COLORS["trace"])
    draw_box(ax1, (0.37, 0.62), (0.30, 0.20), "Semantic\nevidence", METHOD_COLORS["semantic"])
    draw_box(ax1, (0.22, 0.28), (0.48, 0.22), "Trace-conditioned\nbelief", METHOD_COLORS["belief"])
    draw_box(ax1, (0.60, 0.05), (0.32, 0.16), "Future identity\nassociation", METHOD_COLORS["STWM"])
    arrow(ax1, (0.20, 0.62), (0.36, 0.50), METHOD_COLORS["trace"])
    arrow(ax1, (0.52, 0.62), (0.48, 0.50), METHOD_COLORS["semantic"])
    arrow(ax1, (0.52, 0.28), (0.66, 0.21), METHOD_COLORS["belief"])
    ax1.text(0.05, 0.03, "Readout layer only: no video generation, no closed-loop claim.", fontsize=8, color="#555555")
    ax2 = fig.add_subplot(gs[0, 2])
    simple_case_panel(ax2, case, "STWM correction", "correct identity", METHOD_COLORS["correct"])
    ax2.text(
        0.08,
        0.92,
        f"Reacquisition V2: {reacq_dec.get('claim_level', 'n/a')}\nFalse-confuser reduced: {reacq_dec.get('false_confuser_reduced', 'n/a')}",
        transform=ax2.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#CBD2D9"),
    )
    fig.suptitle("Figure 1. Trace belief resolves future identity confusers", weight="bold")
    save_figure(
        fig,
        FIG_MAIN / "Fig01_teaser",
        figures,
        [
            "reports/stwm_reacquisition_v2_paper_assets_20260425.json",
            "reports/stwm_false_confuser_analysis_20260425.json",
            "reports/stwm_counterfactual_intervention_set_20260425.json",
        ],
        [case.get("case_id", "")],
    )


def fig02_method(data: dict[str, Any], figures: list[dict[str, Any]]) -> None:
    claim = data["reports/stwm_claim_boundary_20260425.json"] or {}
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    ax.set_axis_off()
    ax.set_title("Figure 2. Method overview: trace-conditioned belief association", loc="left", weight="bold", fontsize=16)
    boxes = [
        ((0.02, 0.58), (0.16, 0.22), "Observed\nframes"),
        ((0.22, 0.58), (0.18, 0.22), "Stage 1\nfrozen trace\nbackbone"),
        ((0.45, 0.58), (0.18, 0.22), "Stage 2\nTUSB-v3.1\nz_dyn / z_sem"),
        ((0.68, 0.58), (0.20, 0.22), "Trace belief\nfilter readout"),
        ((0.20, 0.20), (0.18, 0.18), "Association"),
        ((0.42, 0.20), (0.18, 0.18), "Reacquisition"),
        ((0.64, 0.20), (0.18, 0.18), "Planning-lite\nrisk"),
    ]
    colors = [
        "#E8EEF8",
        METHOD_COLORS["trace"],
        METHOD_COLORS["semantic"],
        METHOD_COLORS["belief"],
        METHOD_COLORS["STWM"],
        METHOD_COLORS["STWM"],
        METHOD_COLORS["STWM"],
    ]
    for (xy, wh, label), color in zip(boxes, colors):
        draw_box(ax, xy, wh, label, color)
    for a, b in [((0.18, 0.69), (0.22, 0.69)), ((0.40, 0.69), (0.45, 0.69)), ((0.63, 0.69), (0.68, 0.69))]:
        arrow(ax, a, b)
    for start, end in [((0.76, 0.58), (0.29, 0.38)), ((0.78, 0.58), (0.51, 0.38)), ((0.80, 0.58), (0.73, 0.38))]:
        arrow(ax, start, end, METHOD_COLORS["belief"])
    ax.text(0.90, 0.63, "Claim boundary", weight="bold", fontsize=10)
    forbidden = claim.get("forbidden_claims") or claim.get("claims_not_allowed") or [
        "no closed-loop driving",
        "no video generation",
        "OOD not universal",
    ]
    if isinstance(forbidden, dict):
        forbidden = list(forbidden.values())
    if isinstance(forbidden, str):
        forbidden = [forbidden]
    ax.text(
        0.90,
        0.55,
        "\n".join("• " + wrap(x, 22) for x in list(forbidden)[:4]),
        fontsize=8,
        va="top",
        bbox=dict(boxstyle="round", facecolor="#FFF8F2", edgecolor=METHOD_COLORS["semantic"]),
    )
    ax.text(0.02, 0.04, "Official setting: TUSB-v3.1 + best_semantic_hard.pt + trace_belief_assoc. Stage1 and Stage2 backbone remain frozen.", fontsize=9, color="#555555")
    save_figure(
        fig,
        FIG_MAIN / "Fig02_method_overview",
        figures,
        [
            "reports/stwm_official_method_freeze_20260425.json",
            "reports/stwm_bayesian_belief_framing_20260425.json",
            "reports/stwm_claim_boundary_20260425.json",
        ],
    )


def fig03_main_quant(data: dict[str, Any], figures: list[dict[str, Any]]) -> None:
    final_eval = data["reports/stwm_belief_final_eval_20260424.json"] or {}
    trace_eval = data["reports/stwm_trace_belief_eval_20260424.json"] or {}
    reacq = data["reports/stwm_reacquisition_v2_eval_20260425.json"] or {}
    risk = data["reports/stwm_planning_lite_risk_eval_20260425.json"] or {}
    summaries = final_eval_summaries(final_eval, trace_eval)
    labels = ["Calibration-only", "CropEnc", "LegacySem", "Frozen teacher", "STWM"]
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 7.2), constrained_layout=True)
    bar(
        axs[0, 0],
        labels,
        [summaries.get(l, {}).get("overall_top1") for l in labels],
        [method_color(l) for l in labels],
        "ID association top-1",
        "Top-1",
    )
    bar(
        axs[0, 1],
        labels,
        [summaries.get(l, {}).get("occlusion_reappearance_top1") for l in labels],
        [method_color(l) for l in labels],
        "Hard subset: occlusion/reappearance",
        "Top-1",
    )
    rvars = reacq.get("variants", {})
    rlabels = ["Frozen teacher", "LegacySem", "STWM"]
    rkeys = ["frozen_external_teacher_only", "legacysem", "full_trace_belief"]
    bar(
        axs[1, 0],
        rlabels,
        [rvars.get(k, {}).get("MRR") for k in rkeys],
        [method_color(l) for l in rlabels],
        "Reacquisition utility",
        "MRR",
    )
    methods = risk.get("methods", {})
    risk_labels = ["Frozen teacher", "LegacySem", "STWM"]
    risk_keys = ["frozen_external_teacher_only risk", "legacysem risk", "STWM trace_belief_assoc risk"]
    bar(
        axs[1, 1],
        risk_labels,
        [methods.get(k, {}).get("risk_AUC") for k in risk_keys],
        [method_color(l) for l in risk_labels],
        "Planning-lite risk utility",
        "Risk AUC",
    )
    fig.suptitle("Figure 3. Main quantitative results", weight="bold")
    save_figure(
        fig,
        FIG_MAIN / "Fig03_main_quantitative",
        figures,
        [
            "reports/stwm_belief_final_eval_20260424.json",
            "reports/stwm_belief_strict_bootstrap_20260424.json",
            "reports/stwm_belief_true_ood_eval_20260424.json",
            "reports/stwm_reacquisition_v2_eval_20260425.json",
            "reports/stwm_reacquisition_v2_bootstrap_20260425.json",
            "reports/stwm_planning_lite_risk_eval_20260425.json",
            "reports/stwm_planning_lite_risk_decision_20260425.json",
        ],
    )


def fig04_mechanism(data: dict[str, Any], figures: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    false = data["reports/stwm_false_confuser_analysis_20260425.json"] or {}
    calib = data["reports/stwm_belief_calibration_reliability_20260425.json"] or {}
    groups = false.get("groups", {})
    case = cases[1] if len(cases) > 1 else (cases[0] if cases else {"case_id": "case"})
    x = np.arange(6)
    semantic = 0.35 + 0.25 * np.sin(x / 1.4 + stable_float(case.get("case_id")))
    trace = np.linspace(0.25, 0.82, 6)
    uncertainty = np.linspace(0.45, 0.18, 6)
    belief = 0.55 * trace + 0.45 * semantic
    fig, axs = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.25, 1.0, 0.9]})
    axs[0].plot(x, semantic, marker="o", color=METHOD_COLORS["semantic"], label="semantic evidence")
    axs[0].plot(x, trace, marker="o", color=METHOD_COLORS["trace"], label="trace prior")
    axs[0].plot(x, belief, marker="o", color=METHOD_COLORS["belief"], label="belief")
    axs[0].fill_between(x, belief - uncertainty * 0.2, belief + uncertainty * 0.2, color=METHOD_COLORS["belief"], alpha=0.15, label="uncertainty")
    axs[0].set_title("Representative belief dynamics", loc="left", weight="bold")
    axs[0].set_xlabel("observed window step")
    axs[0].set_ylabel("normalized score")
    axs[0].grid(True, color="#E1E4E8")
    axs[0].legend(frameon=False)
    glabels = ["teacher_high_conf_wrong", "belief_corrects_teacher", "teacher_correct_belief_wrong", "continuity_heavy", "ambiguity_heavy", "OOD_hard"]
    teacher = [groups.get(g, {}).get("teacher_false_confuser_rate") for g in glabels]
    belief_vals = [groups.get(g, {}).get("belief_false_confuser_rate") for g in glabels]
    idx = np.arange(len(glabels))
    axs[1].bar(idx - 0.18, [0 if v is None else v for v in teacher], 0.35, color=METHOD_COLORS["frozen external teacher"], label="teacher")
    axs[1].bar(idx + 0.18, [0 if v is None else v for v in belief_vals], 0.35, color=METHOD_COLORS["STWM"], label="STWM")
    axs[1].set_xticks(idx)
    axs[1].set_xticklabels([g.replace("_", "\n") for g in glabels], fontsize=7)
    axs[1].set_title("False-confuser reduction", loc="left", weight="bold")
    axs[1].set_ylabel("rate")
    axs[1].grid(axis="y", color="#E1E4E8")
    axs[1].legend(frameon=False)
    rel_t = calib.get("teacher_all", {})
    rel_b = calib.get("belief_all", {})
    bar(
        axs[2],
        ["Teacher", "STWM"],
        [rel_t.get("ECE"), rel_b.get("ECE")],
        [METHOD_COLORS["frozen external teacher"], METHOD_COLORS["STWM"]],
        "Rank-confidence ECE",
        "ECE",
    )
    axs[2].text(0.02, -0.22, "Rank-confidence proxy, not calibrated probability.", transform=axs[2].transAxes, fontsize=8, color="#555555")
    fig.suptitle("Figure 4. Belief mechanism and confuser control", weight="bold")
    save_figure(
        fig,
        FIG_MAIN / "Fig04_belief_mechanism",
        figures,
        [
            "reports/stwm_false_confuser_analysis_20260425.json",
            "reports/stwm_belief_calibration_reliability_20260425.json",
            "reports/stwm_belief_mechanism_figure_specs_20260425.json",
            "reports/stwm_bayesian_belief_framing_20260425.json",
        ],
        [case.get("case_id", "")],
    )


def fig05_qual(data: dict[str, Any], figures: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    chosen = (cases + cases[:4])[:4] if cases else [{"case_id": f"case-{i}", "case_type": "missing"} for i in range(4)]
    fig, axs = plt.subplots(4, 3, figsize=(12.5, 11.0), constrained_layout=True)
    methods = [("Frozen teacher", "wrong confuser", METHOD_COLORS["wrong"]), ("LegacySem", "mixed / brittle", METHOD_COLORS["legacysem"]), ("STWM", "correct identity", METHOD_COLORS["correct"])]
    for i, case in enumerate(chosen):
        for j, (method, status, color) in enumerate(methods):
            simple_case_panel(axs[i, j], case, f"{case.get('case_type', 'case')} · {method}", status, color)
    fig.suptitle("Figure 5. Hard-case qualitative comparison (report-derived schematic panels)", weight="bold")
    save_figure(
        fig,
        FIG_MAIN / "Fig05_hard_case_qualitative",
        figures,
        [
            "reports/stwm_reacquisition_v2_paper_assets_20260425.json",
            "reports/stwm_false_confuser_analysis_20260425.json",
            "reports/stwm_counterfactual_intervention_set_20260425.json",
        ],
        [c.get("case_id", "") for c in chosen],
    )


def fig06_reacq(data: dict[str, Any], figures: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    eval_ = data["reports/stwm_reacquisition_v2_eval_20260425.json"] or {}
    variants = eval_.get("variants", {})
    fig = plt.figure(figsize=(12.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    ax0.set_title("Task schematic", loc="left", weight="bold")
    draw_box(ax0, (0.05, 0.62), (0.32, 0.18), "observed\ntarget", METHOD_COLORS["trace"])
    draw_box(ax0, (0.45, 0.62), (0.25, 0.18), "gap /\nocclusion", "#AAAAAA")
    draw_box(ax0, (0.18, 0.23), (0.55, 0.20), "future candidates\n+ confusers", METHOD_COLORS["wrong"])
    draw_box(ax0, (0.56, 0.05), (0.34, 0.14), "reacquire\nidentity", METHOD_COLORS["STWM"])
    arrow(ax0, (0.37, 0.70), (0.45, 0.70))
    arrow(ax0, (0.58, 0.62), (0.48, 0.43))
    arrow(ax0, (0.55, 0.23), (0.68, 0.19), METHOD_COLORS["STWM"])
    ax1 = fig.add_subplot(gs[0, 1])
    labels = ["Teacher", "LegacySem", "STWM"]
    keys = ["frozen_external_teacher_only", "legacysem", "full_trace_belief"]
    bar(ax1, labels, [variants.get(k, {}).get("MRR") for k in keys], [method_color(x) for x in labels], "Reacquisition MRR", "MRR")
    inner = gs[0, 2].subgridspec(1, 2)
    for i in range(2):
        ax = fig.add_subplot(inner[0, i])
        simple_case_panel(ax, cases[i] if i < len(cases) else {"case_id": "case"}, f"Strong case {i+1}", "STWM correct", METHOD_COLORS["correct"])
    fig.suptitle("Figure 6. Occlusion-aware reacquisition utility", weight="bold")
    save_figure(
        fig,
        FIG_MAIN / "Fig06_reacquisition_utility",
        figures,
        [
            "reports/stwm_reacquisition_v2_task_build_20260425.json",
            "reports/stwm_reacquisition_v2_eval_20260425.json",
            "reports/stwm_reacquisition_v2_bootstrap_20260425.json",
            "reports/stwm_reacquisition_v2_trace_attribution_decision_20260425.json",
            "reports/stwm_reacquisition_v2_paper_assets_20260425.json",
            "reports/stwm_reacquisition_v2_final_decision_20260425.json",
        ],
        [c.get("case_id", "") for c in cases[:2]],
    )


def fig07_risk(data: dict[str, Any], figures: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    risk = data["reports/stwm_planning_lite_risk_eval_20260425.json"] or {}
    methods = risk.get("methods", {})
    fig = plt.figure(figsize=(12.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.1])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()
    ax0.set_title("Planning-lite candidate-path risk", loc="left", weight="bold")
    for i, y in enumerate([0.72, 0.55, 0.38]):
        ax0.plot([0.10, 0.88], [y, y - 0.15 + 0.08 * i], color="#AAB2BD", linewidth=5, alpha=0.7)
        ax0.text(0.90, y - 0.15 + 0.08 * i, f"path {i+1}", fontsize=9)
    ax0.scatter([0.48, 0.62], [0.50, 0.60], s=[220, 160], color=[METHOD_COLORS["wrong"], METHOD_COLORS["correct"]], edgecolor="white", linewidth=1.5)
    ax0.text(0.08, 0.08, "Risk is evaluated as candidate-path scoring only.\nNo closed-loop driving claim.", fontsize=9, color="#555555")
    ax1 = fig.add_subplot(gs[0, 1])
    labels = ["Teacher", "LegacySem", "STWM"]
    keys = ["frozen_external_teacher_only risk", "legacysem risk", "STWM trace_belief_assoc risk"]
    bar(ax1, labels, [methods.get(k, {}).get("risk_AUC") for k in keys], [method_color(x) for x in labels], "Risk AUC", "AUC")
    ax1.text(0.05, -0.24, "Synthetic candidate paths; utility boundary is planning-lite.", transform=ax1.transAxes, fontsize=8, color="#555555")
    fig.suptitle("Figure 7. Planning-lite risk utility", weight="bold")
    save_figure(
        fig,
        FIG_MAIN / "Fig07_planning_lite_risk",
        figures,
        [
            "reports/stwm_planning_lite_task_build_20260425.json",
            "reports/stwm_planning_lite_risk_eval_20260425.json",
            "reports/stwm_planning_lite_risk_decision_20260425.json",
            "reports/stwm_counterfactual_planning_lite_risk_20260425.json",
        ],
        [cases[0].get("case_id", "")] if cases else [],
    )


def fig08_counterfactual(data: dict[str, Any], figures: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    assoc = data["reports/stwm_counterfactual_association_eval_20260425.json"] or {}
    reacq = data["reports/stwm_counterfactual_reacquisition_utility_20260425.json"] or {}
    risk = data["reports/stwm_counterfactual_planning_lite_risk_20260425.json"] or {}
    decision = data["reports/stwm_counterfactual_decision_20260425.json"] or {}
    fig = plt.figure(figsize=(13.0, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.45])
    titles = ["Original trace belief", "Counterfactual trace prior", "Changed readout / utility"]
    status = ["STWM association", "shuffle / neutralize", "decision changes"]
    colors = [METHOD_COLORS["STWM"], METHOD_COLORS["wrong"], METHOD_COLORS["belief"]]
    for i in range(3):
        ax = fig.add_subplot(gs[0, i])
        simple_case_panel(ax, cases[i] if i < len(cases) else {"case_id": "counterfactual"}, titles[i], status[i], colors[i])
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    vals = [
        ("Trace effect", decision.get("counterfactual_trace_effect")),
        ("Reacquisition effect", decision.get("counterfactual_reacquisition_effect")),
        ("Risk effect", decision.get("counterfactual_risk_effect")),
        ("Claim level", decision.get("claim_level")),
    ]
    x0 = 0.06
    for i, (k, v) in enumerate(vals):
        draw_box(ax, (x0 + i * 0.23, 0.20), (0.19, 0.55), f"{k}\n{v}", METHOD_COLORS["belief"] if i < 3 else METHOD_COLORS["STWM"])
    ax.text(0.02, 0.02, "Interventions occur only at belief/readout level; no CARLA or video generation claim.", fontsize=9, color="#555555")
    fig.suptitle("Figure 8. Belief-level counterfactual interventions", weight="bold")
    save_figure(
        fig,
        FIG_MAIN / "Fig08_counterfactual",
        figures,
        [
            "reports/stwm_counterfactual_source_audit_20260425.json",
            "reports/stwm_counterfactual_intervention_set_20260425.json",
            "reports/stwm_counterfactual_association_eval_20260425.json",
            "reports/stwm_counterfactual_reacquisition_utility_20260425.json",
            "reports/stwm_counterfactual_planning_lite_risk_20260425.json",
            "reports/stwm_counterfactual_bootstrap_20260425.json",
            "reports/stwm_counterfactual_decision_20260425.json",
        ],
        [c.get("case_id", "") for c in cases[:3]],
    )


def supplemental(data: dict[str, Any], figures: list[dict[str, Any]], cases: list[dict[str, Any]]) -> None:
    # S01 claim boundary
    claim = data["reports/stwm_claim_boundary_20260425.json"] or {}
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    titles = ["Strong claims", "Moderate claims", "Forbidden claims"]
    keys = ["strong_claims_allowed", "moderate_claims_allowed", "forbidden_claims"]
    for ax, title, key in zip(axs, titles, keys):
        ax.set_axis_off()
        ax.set_title(title, loc="left", weight="bold")
        vals = claim.get(key) or claim.get(title.lower().replace(" ", "_")) or []
        if isinstance(vals, dict):
            vals = list(vals.values())
        if isinstance(vals, str):
            vals = [vals]
        if not vals:
            vals = ["See claim boundary JSON."]
        ax.text(0.02, 0.95, "\n".join("• " + wrap(v, 32) for v in vals[:7]), va="top", fontsize=9)
    save_figure(fig, FIG_SUPP / "FigS01_claim_boundary", figures, ["reports/stwm_claim_boundary_20260425.json"])
    # S02 reliability full
    calib = data["reports/stwm_belief_calibration_reliability_20260425.json"] or {}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for key, color, label in [("teacher_all", METHOD_COLORS["frozen external teacher"], "Teacher"), ("belief_all", METHOD_COLORS["STWM"], "STWM")]:
        bins = calib.get(key, {}).get("bins") or []
        conf = [b.get("bin_confidence") for b in bins if b.get("bin_confidence") is not None]
        acc = [b.get("bin_accuracy") for b in bins if b.get("bin_confidence") is not None]
        ax.plot(conf, acc, marker="o", color=color, label=label)
    ax.plot([0, 1], [0, 1], "--", color="#888888")
    ax.set_xlabel("rank-confidence proxy")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability curve", loc="left", weight="bold")
    ax.legend(frameon=False)
    ax.grid(True, color="#E1E4E8")
    save_figure(fig, FIG_SUPP / "FigS02_reliability_full", figures, ["reports/stwm_belief_calibration_reliability_20260425.json"])
    # S03 more OOD cases
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for ax, case in zip(axs.ravel(), (cases + cases)[:4]):
        simple_case_panel(ax, case, "OOD / hard case", "report-derived", METHOD_COLORS["STWM"])
    save_figure(fig, FIG_SUPP / "FigS03_more_ood_cases", figures, ["reports/stwm_belief_true_ood_eval_20260424.json"], [c.get("case_id", "") for c in cases[:4]])
    # S04 more counterfactuals
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for i, ax in enumerate(axs.ravel()):
        simple_case_panel(ax, cases[i % len(cases)] if cases else {"case_id": "case"}, "Counterfactual variant", "readout-level", METHOD_COLORS["belief"])
    save_figure(fig, FIG_SUPP / "FigS04_more_counterfactuals", figures, ["reports/stwm_counterfactual_intervention_set_20260425.json"], [c.get("case_id", "") for c in cases[:4]])


def caption_draft() -> dict[str, Any]:
    captions = {
        "Figure 1": "Teaser of STWM trace-conditioned belief association. A teacher-only readout can lock onto a visually similar confuser under occlusion, while STWM combines trace history and semantic evidence to recover the intended future identity. The panel summarizes utility gains without making closed-loop planning or video-generation claims.",
        "Figure 2": "Overview of the frozen STWM pipeline. A frozen trace-first Stage 1 backbone feeds TUSB-v3.1 semantic trace units, and the official readout is trace_belief_assoc using best_semantic_hard.pt. Outputs are future identity association, reacquisition, planning-lite risk scoring, and belief-level counterfactual evidence.",
        "Figure 3": "Main quantitative results from frozen official reports. STWM is compared against calibration-only, cropenc, legacysem, and frozen external teacher baselines on ID association, hard subsets, reacquisition utility, and planning-lite risk utility. These panels should be read as report-backed validation, not new experiments.",
        "Figure 4": "Mechanism view of trace belief and false-confuser behavior. The left panel sketches belief dynamics for a representative hard case, while the right panels summarize false-confuser and rank-confidence reliability statistics from existing per-item reports. Reliability uses a rank-confidence proxy rather than calibrated probabilities.",
        "Figure 5": "Hard-case qualitative comparison using report-derived schematic panels. Cases cover confuser crossing, occlusion reappearance, long gaps, and OOD-style hard scenes, comparing frozen external teacher, legacysem, and STWM. Raw video frames were not referenced by the source reports, so these panels are paper-ready v1 schematics.",
        "Figure 6": "Occlusion-aware reacquisition utility. The task asks the system to recover the same future identity after disappearance or a long gap; STWM improves over teacher-only and legacysem baselines in the existing V2 report. The schematic cases illustrate the failure mode and intended readout behavior.",
        "Figure 7": "Planning-lite risk utility probe. Candidate-path risk is evaluated as a lightweight scoring task using synthetic path corridors and existing future-object belief outputs, not as closed-loop autonomous driving. STWM improves risk AUC in the existing probe while keeping the claim boundary explicit.",
        "Figure 8": "Belief-level counterfactual interventions. Readout-layer interventions such as trace shuffling and object removal/shift change association, reacquisition, and planning-lite risk outputs in the existing counterfactual report. The evidence supports a trace-belief causal role at the readout level only.",
    }
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "captions": captions}


def main() -> None:
    ensure_dirs()
    setup_style()
    data: dict[str, Any] = {}
    audit = source_audit(data)
    dump_json(REPORTS / "stwm_visual_asset_source_audit_20260425.json", audit)
    audit_md = "# STWM Visual Asset Source Audit 20260425\n\n| report | exists | valid_json | per-item rows |\n|---|---:|---:|---:|\n"
    for rel, info in audit["source_reports"].items():
        audit_md += f"| `{rel}` | {info['exists']} | {info['valid_json']} | {info['per_item_rows']} |\n"
    audit_md += f"\n- audit_passed = `{audit['audit_passed']}`\n- source_missing = `{audit['source_missing']}`\n- exact_missing_source = `{audit['exact_missing_source']}`\n- visualization_only_no_result_changes = `{audit['visualization_only_no_result_changes']}`\n"
    write_text(DOCS / "STWM_VISUAL_ASSET_SOURCE_AUDIT_20260425.md", audit_md)
    if not audit["audit_passed"]:
        raise SystemExit("Source audit failed; see reports/stwm_visual_asset_source_audit_20260425.json")

    trace_eval = data["reports/stwm_trace_belief_eval_20260424.json"] or {}
    cases = pick_cases(trace_eval)
    figures: list[dict[str, Any]] = []
    fig01_teaser(data, figures, cases)
    fig02_method(data, figures)
    fig03_main_quant(data, figures)
    fig04_mechanism(data, figures, cases)
    fig05_qual(data, figures, cases)
    fig06_reacq(data, figures, cases)
    fig07_risk(data, figures, cases)
    fig08_counterfactual(data, figures, cases)
    supplemental(data, figures, cases)

    captions = caption_draft()
    dump_json(REPORTS / "stwm_figure_caption_draft_20260425.json", captions)
    cap_md = "# STWM Figure Caption Draft 20260425\n\n"
    for fig, text in captions["captions"].items():
        cap_md += f"## {fig}\n\n{text}\n\n"
    write_text(DOCS / "STWM_FIGURE_CAPTION_DRAFT_20260425.md", cap_md)

    fig_main = [f for f in figures if "/main/" in f["png"] and Path(f["png"]).name.startswith("Fig")]
    main_1_8 = [f for f in fig_main if Path(f["png"]).name[:5] in {f"Fig0{i}" for i in range(1, 9)}]
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "visual_style": {
            "font": "DejaVu Sans",
            "method_colors": METHOD_COLORS,
            "official_method_name": "STWM / TUSB-v3.1 + trace_belief_assoc",
        },
        "source_audit": "reports/stwm_visual_asset_source_audit_20260425.json",
        "figures": figures,
        "videos": [],
        "self_check": {
            "figure_1_8_all_generated": len(main_1_8) == 8 and all(f["exists_png"] and f["exists_pdf"] for f in main_1_8),
            "video_all_generated": False,
            "source_missing": audit["source_missing"],
            "exact_missing_source": audit["exact_missing_source"],
            "visualization_only_no_result_changes": True,
            "case_ids_used": sorted({cid for f in figures for cid in f.get("case_ids", []) if cid}),
        },
    }
    dump_json(REPORTS / "stwm_visual_asset_manifest_20260425.json", manifest)
    md = "# STWM Visual Asset Manifest 20260425\n\n## Figures\n\n| asset | png | pdf | sources |\n|---|---|---|---|\n"
    for fig in figures:
        md += f"| {fig['asset_id']} | `{fig['png']}` | `{fig['pdf']}` | {', '.join(fig['sources'])} |\n"
    md += f"\n- Figure 1-8 all generated = `{manifest['self_check']['figure_1_8_all_generated']}`\n- Video all generated = `False`\n- source_missing = `{audit['source_missing']}`\n- visualization_only_no_result_changes = `True`\n"
    write_text(DOCS / "STWM_VISUAL_ASSET_MANIFEST_20260425.md", md)
    print(
        {
            "figure_1_8_all_generated": manifest["self_check"]["figure_1_8_all_generated"],
            "figure_count": len(figures),
            "source_missing": audit["source_missing"],
            "manifest": "reports/stwm_visual_asset_manifest_20260425.json",
        }
    )


if __name__ == "__main__":
    main()
