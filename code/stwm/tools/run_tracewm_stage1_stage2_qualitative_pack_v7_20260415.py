#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import json


def _repo_root() -> Path:
    for candidate in [
        Path("/raid/chen034/workspace/stwm"),
        Path("/home/chen034/workspace/stwm"),
    ]:
        if candidate.exists():
            return candidate
    raise RuntimeError("unable to resolve STWM repo root")


ROOT = _repo_root()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path_like: Any) -> Dict[str, Any]:
    path = Path(path_like)
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path_like: Any, payload: Dict[str, Any]) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_md(path_like: Any, lines: List[str]) -> None:
    path = Path(path_like)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _metric_triplet(row: Dict[str, Any]) -> Tuple[float, float, float]:
    metrics = ((row.get("best_checkpoint_metric") or {}).get("metrics") or {}) if isinstance(row, dict) else {}
    return (
        float(metrics.get("free_rollout_endpoint_l2", 1e9)),
        float(metrics.get("free_rollout_coord_mean_l2", 1e9)),
        float(metrics.get("teacher_forced_coord_loss", 1e9)),
    )


def _compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    side = row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}
    return {
        "run_name": str(row.get("run_name", "none")),
        "endpoint_l2": _metric_triplet(row)[0],
        "coord_mean_l2": _metric_triplet(row)[1],
        "teacher_forced_coord_loss": _metric_triplet(row)[2],
        "semantic_hard_sidecar_score": float(side.get("semantic_hard_sidecar_score", 1e9)),
    }


def _case(case_id: str, dataset: str, tags: List[str], why: str, methods: Dict[str, Any], interpretation: str) -> Dict[str, Any]:
    return {
        "case_id": case_id,
        "dataset_source": dataset,
        "subset_tags": tags,
        "why_selected": why,
        "compared_methods": list(methods.keys()),
        "metric_table": methods,
        "qualitative_interpretation": interpretation,
    }


def _stage2_case_from_item(case_id: str, tags: List[str], why: str, interpretation: str, item: Dict[str, Any]) -> Dict[str, Any]:
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


def _pick_item(items: List[Dict[str, Any]], predicate) -> Dict[str, Any]:
    for item in items:
        if predicate(item):
            return item
    return items[0] if items else {}


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--final-pack-summary", default=str(ROOT / "reports/stage2_calibration_only_final_pack_summary_20260414.json"))
    parser.add_argument("--final-pack-diagnosis", default=str(ROOT / "reports/stage2_calibration_only_final_pack_diagnosis_20260414.json"))
    parser.add_argument("--state-ident-eval", default=str(ROOT / "reports/stage2_state_identifiability_eval_20260415.json"))
    parser.add_argument("--mechanism-fix-summary", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_summary_20260415.json"))
    parser.add_argument("--stage1-prev", default=str(ROOT / "reports/stage1_qualitative_pack_v6_20260414.json"))
    parser.add_argument("--stage2-prev", default=str(ROOT / "reports/stage2_qualitative_pack_v6_20260414.json"))
    parser.add_argument("--stage1-output", default=str(ROOT / "reports/stage1_qualitative_pack_v7_20260415.json"))
    parser.add_argument("--stage2-output", default=str(ROOT / "reports/stage2_qualitative_pack_v7_20260415.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE1_STAGE2_QUALITATIVE_PACK_V7_20260415.md"))
    args = parser.parse_args()

    final_pack = read_json(args.final_pack_summary)
    state_eval = read_json(args.state_ident_eval)
    mechanism = read_json(args.mechanism_fix_summary)
    stage1_prev = read_json(args.stage1_prev)
    stage2_prev = read_json(args.stage2_prev)

    final_rows = final_pack.get("run_rows", []) if isinstance(final_pack.get("run_rows", []), list) else []
    mech_rows = mechanism.get("run_rows", []) if isinstance(mechanism.get("run_rows", []), list) else []
    state_items = state_eval.get("per_item_results", []) if isinstance(state_eval.get("per_item_results", []), list) else []

    by_seed_mainline = {
        int(r.get("seed", -1)): r
        for r in final_rows
        if isinstance(r, dict) and str(r.get("family", "")) == "topk1" and str(r.get("status", "")).lower() == "completed"
    }
    main_best = min(by_seed_mainline.values(), key=_metric_triplet) if by_seed_mainline else {}
    noalign_row = next((r for r in mech_rows if str(r.get("ablation_name", "")) == "noalign" and int(r.get("seed", -1)) == 654), {}) or next((r for r in mech_rows if str(r.get("ablation_name", "")) == "noalign"), {})
    dense_row = next((r for r in mech_rows if str(r.get("ablation_name", "")) == "densegate" and int(r.get("seed", -1)) == 654), {}) or next((r for r in mech_rows if str(r.get("ablation_name", "")) == "densegate"), {})
    nodelay_row = next((r for r in mech_rows if str(r.get("ablation_name", "")) == "nodelay" and int(r.get("seed", -1)) == 654), {}) or next((r for r in mech_rows if str(r.get("ablation_name", "")) == "nodelay"), {})

    stage1_cases = stage1_prev.get("cases", []) if isinstance(stage1_prev.get("cases", []), list) else []
    if not stage1_cases:
        stage1_cases = [
            _case("stage1_easy_v7", "VSPW/VIPSeg", ["easy"], "Stable future trace/state anchor.", {"stage1_frozen_baseline": {"role": "trace backbone"}}, "Easy case for Stage1 trace forecasting."),
            _case("stage1_dynamic_change_v7", "VSPW/VIPSeg", ["dynamic-change"], "Stage1 dynamic-change example.", {"stage1_frozen_baseline": {"role": "trace backbone"}}, "Dynamic-change case for Stage1."),
            _case("stage1_failure_boundary_v7", "VSPW/VIPSeg", ["failure-boundary"], "Boundary case motivating selective Stage2 calibration.", {"stage1_frozen_baseline": {"role": "trace backbone"}}, "Failure-boundary case retained for figure selection."),
        ]
    stage1_payload = {
        "generated_at_utc": now_iso(),
        "pack_version": "v7",
        "ready_for_paper_figure_selection": True,
        "cases": stage1_cases[:12],
    }

    def m(item: Dict[str, Any], name: str, key: str) -> float:
        row = ((item.get("methods") or {}).get(name) or {}) if isinstance(item.get("methods", {}), dict) else {}
        return float(row.get(key, 0.0 if "acc" in key or "iou" in key or "rate" in key else 1e9))

    cal_win = _pick_item(
        state_items,
        lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") > max(
            m(item, "stage1_frozen_baseline", "query_future_top1_acc"),
            m(item, "legacysem_best", "query_future_top1_acc"),
            m(item, "cropenc_baseline_best", "query_future_top1_acc"),
        ),
    )
    cal_fail = _pick_item(
        state_items,
        lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") < max(
            m(item, "stage1_frozen_baseline", "query_future_top1_acc"),
            m(item, "legacysem_best", "query_future_top1_acc"),
            m(item, "cropenc_baseline_best", "query_future_top1_acc"),
        ),
    )
    legacy_win = _pick_item(state_items, lambda item: m(item, "legacysem_best", "query_future_top1_acc") > m(item, "calibration_only_mainline_best", "query_future_top1_acc"))
    crop_win = _pick_item(state_items, lambda item: m(item, "cropenc_baseline_best", "query_future_top1_acc") > m(item, "calibration_only_mainline_best", "query_future_top1_acc"))
    noalign_fail = _pick_item(state_items, lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") > m(item, "noalign_failure", "query_future_top1_acc"))
    dense_fail = _pick_item(state_items, lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") > m(item, "densegate_failure", "query_future_top1_acc"))
    nodelay_fail = _pick_item(state_items, lambda item: m(item, "calibration_only_mainline_best", "query_future_top1_acc") > m(item, "nodelay_failure", "query_future_top1_acc"))

    stage2_cases: List[Dict[str, Any]] = []
    if cal_win:
        stage2_cases.append(_stage2_case_from_item("stage2_calibration_clear_win_v7", ["calibration-only", "state-identifiability-success"], "Calibration-only beats Stage1/legacysem/cropenc on a real identifiability item.", "Figure candidate for the main Stage2 success story.", cal_win))
    if cal_fail:
        stage2_cases.append(_stage2_case_from_item("stage2_calibration_fail_v7", ["calibration-only-fail", "state-identifiability-failure"], "Calibration-only does not win everywhere; include a hard negative example.", "Failure case kept to prevent one-sided qualitative selection.", cal_fail))
    if legacy_win:
        stage2_cases.append(_stage2_case_from_item("stage2_legacysem_win_v7", ["legacysem-win"], "Legacy semantic baseline wins on this protocol item.", "Use this when discussing residual failure regimes.", legacy_win))
    if crop_win:
        stage2_cases.append(_stage2_case_from_item("stage2_cropenc_win_v7", ["cropenc-win"], "Crop visual baseline wins on this protocol item.", "Shows calibration-only is not universally dominant.", crop_win))
    if noalign_fail:
        stage2_cases.append(_stage2_case_from_item("stage2_noalign_failure_v7", ["noalign-failure"], "No-alignment ablation degrades relative to the calibration mainline.", "Mechanism figure candidate for semantic alignment being load-bearing.", noalign_fail))
    if dense_fail:
        stage2_cases.append(_stage2_case_from_item("stage2_densegate_failure_v7", ["densegate-failure"], "Dense-gate ablation removes useful selectivity.", "Mechanism figure candidate for sparse gating.", dense_fail))
    if nodelay_fail:
        stage2_cases.append(_stage2_case_from_item("stage2_nodelay_failure_v7", ["nodelay-failure"], "No-delay ablation disrupts the delayed auxiliary schedule.", "Mechanism figure candidate for delayed schedule.", nodelay_fail))
    if cal_win:
        stage2_cases.append(_stage2_case_from_item("stage2_state_identifiability_success_v7", ["state-identifiability-success"], "Real future grounding success case under the new protocol.", "Use directly in protocol contribution figures.", cal_win))
    if cal_fail:
        stage2_cases.append(_stage2_case_from_item("stage2_state_identifiability_failure_v7", ["state-identifiability-failure"], "Real future grounding failure case under the new protocol.", "Use this to bound claims and show protocol difficulty.", cal_fail))

    summary_case = _case(
        "stage2_mechanism_summary_v7",
        "VSPW+VIPSeg",
        ["mechanism-summary"],
        "Compact summary of current best mainline and three ablation controls.",
        {
            "calibration_mainline_best": _compact_row(main_best),
            "noalign_reference_failure": _compact_row(noalign_row),
            "densegate_reference_failure": _compact_row(dense_row),
            "nodelay_reference_failure": _compact_row(nodelay_row),
        },
        "Compact method table for paper figure selection and caption drafting.",
    )
    stage2_cases.insert(0, summary_case)

    previous_cases = stage2_prev.get("cases", []) if isinstance(stage2_prev.get("cases", []), list) else []
    for row in previous_cases:
        if isinstance(row, dict) and len(stage2_cases) < 16:
            stage2_cases.append(row)

    stage2_payload = {
        "generated_at_utc": now_iso(),
        "pack_version": "v7",
        "ready_for_paper_figure_selection": True,
        "state_identifiability_eval_source": str(args.state_ident_eval),
        "mechanism_fix_summary_source": str(args.mechanism_fix_summary),
        "cases": stage2_cases[:16],
    }

    write_json(args.stage1_output, stage1_payload)
    write_json(args.stage2_output, stage2_payload)
    write_md(
        args.output_md,
        [
            "# Stage1 / Stage2 Qualitative Pack V7",
            "",
            f"- generated_at_utc: {now_iso()}",
            f"- stage1_case_count: {len(stage1_payload['cases'])}",
            f"- stage2_case_count: {len(stage2_payload['cases'])}",
            "- ready_for_paper_figure_selection: True",
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
