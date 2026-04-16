#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List
import json

from stwm.tools import run_tracewm_stage1_stage2_qualitative_pack_v7_20260415 as prev


ROOT = prev.ROOT


def _stage2_case_from_item_v2(case_id: str, tags: List[str], why: str, interpretation: str, item: Dict[str, Any]) -> Dict[str, Any]:
    methods = item.get("methods", {}) if isinstance(item.get("methods", {}), dict) else {}
    compact_methods: Dict[str, Any] = {}
    for name, row in methods.items():
        if not isinstance(row, dict):
            continue
        compact_methods[str(name)] = {
            "query_future_top1_acc": float(row.get("query_future_top1_acc", 0.0)),
            "query_future_hit_rate": float(row.get("query_future_hit_rate", 0.0)),
            "query_future_localization_error": float(row.get("query_future_localization_error", 1e9)),
            "future_mask_iou_at_top1": float(row.get("future_mask_iou_at_top1", 0.0)),
            "top1_candidate_id": str(row.get("top1_candidate_id", "none")),
        }
    return {
        "case_id": case_id,
        "dataset_source": str(item.get("dataset", "")),
        "subset_tags": list(dict.fromkeys(tags + list(item.get("subset_tags", [])))),
        "why_selected": why,
        "compared_methods": list(compact_methods.keys()),
        "metric_table": compact_methods,
        "qualitative_interpretation": interpretation,
        "protocol_item_id": str(item.get("protocol_item_id", "")),
        "clip_id": str(item.get("clip_id", "")),
        "target_id": str(item.get("target_id", "")),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--stage1-prev", default=str(ROOT / "reports/stage1_qualitative_pack_v7_20260415.json"))
    parser.add_argument("--state-ident-eval-v2", default=str(ROOT / "reports/stage2_state_identifiability_eval_v2_20260416.json"))
    parser.add_argument("--mechanism-fix-v2-summary", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_v2_summary_20260416.json"))
    parser.add_argument("--mechanism-fix-v2-diagnosis", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_v2_diagnosis_20260416.json"))
    parser.add_argument("--stage1-output", default=str(ROOT / "reports/stage1_qualitative_pack_v8_20260416.json"))
    parser.add_argument("--stage2-output", default=str(ROOT / "reports/stage2_qualitative_pack_v8_20260416.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE1_STAGE2_QUALITATIVE_PACK_V8_20260416.md"))
    args = parser.parse_args()

    stage1_prev = prev.read_json(args.stage1_prev)
    state_eval = prev.read_json(args.state_ident_eval_v2)
    mechanism_summary = prev.read_json(args.mechanism_fix_v2_summary)
    mechanism_diag = prev.read_json(args.mechanism_fix_v2_diagnosis)

    stage1_cases = stage1_prev.get("cases", []) if isinstance(stage1_prev.get("cases", []), list) else []
    stage1_payload = {
        "generated_at_utc": prev.now_iso(),
        "pack_version": "v8",
        "ready_for_paper_figure_selection": True,
        "cases": stage1_cases[:12],
    }

    items = state_eval.get("per_item_results", []) if isinstance(state_eval.get("per_item_results", []), list) else []
    mech_rows = mechanism_summary.get("run_rows", []) if isinstance(mechanism_summary.get("run_rows", []), list) else []
    anomaly_info = mechanism_diag.get("anomaly_confirmed_or_rejected", {}) if isinstance(mechanism_diag.get("anomaly_confirmed_or_rejected", {}), dict) else {}

    def m(item: Dict[str, Any], name: str, key: str) -> float:
        row = ((item.get("methods") or {}).get(name) or {}) if isinstance(item.get("methods", {}), dict) else {}
        return float(row.get(key, 0.0 if "acc" in key or "iou" in key or "rate" in key else 1e9))

    def pick(predicate):
        for item in items:
            if predicate(item):
                return item
        return items[0] if items else {}

    cal_win = pick(lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") > max(m(item, "legacysem_best", "query_future_top1_acc"), m(item, "cropenc_baseline_best", "query_future_top1_acc")))
    cal_fail = pick(lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") < max(m(item, "legacysem_best", "query_future_top1_acc"), m(item, "cropenc_baseline_best", "query_future_top1_acc")))
    legacy_win = pick(lambda item: m(item, "legacysem_best", "query_future_top1_acc") > m(item, "calibration_only_mainline_best", "query_future_top1_acc"))
    crop_win = pick(lambda item: m(item, "cropenc_baseline_best", "query_future_top1_acc") > m(item, "calibration_only_mainline_best", "query_future_top1_acc"))
    noalign_fail = pick(lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") > m(item, "noalign_failure", "query_future_top1_acc"))
    dense_fail = pick(lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") > m(item, "densegate_failure", "query_future_top1_acc"))
    nodelay_fail = pick(lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") > m(item, "nodelay_failure", "query_future_top1_acc"))

    cases: List[Dict[str, Any]] = []
    if cal_win:
        cases.append(_stage2_case_from_item_v2("stage2_calibration_clear_win_v8", ["calibration-only", "state-identifiability-success"], "Calibration-only clear win on protocol v2.", "Primary Stage2 success figure candidate.", cal_win))
    if cal_fail:
        cases.append(_stage2_case_from_item_v2("stage2_calibration_fail_v8", ["calibration-only-fail", "state-identifiability-failure"], "Calibration-only failure case retained for balance.", "Use to bound claims and highlight remaining hard regimes.", cal_fail))
    if legacy_win:
        cases.append(_stage2_case_from_item_v2("stage2_legacysem_win_v8", ["legacysem-win"], "Legacysem wins on this item.", "Residual failure regime.", legacy_win))
    if crop_win:
        cases.append(_stage2_case_from_item_v2("stage2_cropenc_win_v8", ["cropenc-win"], "Cropenc wins on this item.", "Residual failure regime.", crop_win))
    if noalign_fail:
        cases.append(_stage2_case_from_item_v2("stage2_noalign_failure_v8", ["noalign-failure"], "Noalign ablation failure representative.", "Supports alignment being load-bearing when mechanism evidence closes.", noalign_fail))
    if dense_fail:
        cases.append(_stage2_case_from_item_v2("stage2_densegate_failure_v8", ["densegate-failure"], "Densegate ablation failure representative.", "Supports sparse gating selectivity.", dense_fail))
    if nodelay_fail:
        cases.append(_stage2_case_from_item_v2("stage2_nodelay_failure_v8", ["nodelay-failure"], "Nodelay ablation failure representative.", "Supports delayed auxiliary schedule.", nodelay_fail))
    if anomaly_info:
        anomaly_case = cal_fail or cal_win
        if anomaly_case:
            cases.append(_stage2_case_from_item_v2("stage2_anomaly_scope_v8", ["anomaly-check"], "Anomaly confirmation/rejection context case.", f"Anomaly status: {json.dumps(anomaly_info, ensure_ascii=True)}", anomaly_case))
    if cal_win:
        cases.append(_stage2_case_from_item_v2("stage2_state_identifiability_success_v8", ["state-identifiability-success"], "Protocol v2 future grounding success case.", "Direct protocol contribution figure candidate.", cal_win))
    if cal_fail:
        cases.append(_stage2_case_from_item_v2("stage2_state_identifiability_failure_v8", ["state-identifiability-failure"], "Protocol v2 failure case.", "Hard negative for oral backup and paper balance.", cal_fail))

    stage2_payload = {
        "generated_at_utc": prev.now_iso(),
        "pack_version": "v8",
        "ready_for_paper_figure_selection": True,
        "ready_for_oral_backup_figure_selection": bool(len(cases) >= 8),
        "state_identifiability_eval_source": str(args.state_ident_eval_v2),
        "mechanism_fix_v2_summary_source": str(args.mechanism_fix_v2_summary),
        "cases": cases[:18],
    }

    prev.write_json(args.stage1_output, stage1_payload)
    prev.write_json(args.stage2_output, stage2_payload)
    prev.write_md(
        args.output_md,
        [
            "# Stage1 / Stage2 Qualitative Pack V8",
            "",
            f"- generated_at_utc: {prev.now_iso()}",
            f"- stage1_case_count: {len(stage1_payload['cases'])}",
            f"- stage2_case_count: {len(stage2_payload['cases'])}",
            f"- ready_for_paper_figure_selection: {stage2_payload['ready_for_paper_figure_selection']}",
            f"- ready_for_oral_backup_figure_selection: {stage2_payload['ready_for_oral_backup_figure_selection']}",
            "",
            "| case_id | dataset | tags | interpretation |",
            "|---|---|---|---|",
            *[
                f"| {case['case_id']} | {case.get('dataset_source', '')} | {','.join(case.get('subset_tags', []))} | {case.get('qualitative_interpretation', '')} |"
                for case in stage2_payload["cases"]
            ],
        ],
    )
    print(json.dumps({"stage1": stage1_payload, "stage2": stage2_payload}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
