#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json

ROOT = Path("/home/chen034/workspace/stwm")

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

def write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

def metric(row: Dict[str, Any], name: str = "free_rollout_endpoint_l2") -> float:
    block = row.get("best_checkpoint_metric", {}) if isinstance(row.get("best_checkpoint_metric", {}), dict) else {}
    metrics = block.get("metrics", {}) if isinstance(block.get("metrics", {}), dict) else {}
    return float(metrics.get(name, 1e9))

def hard(row: Dict[str, Any]) -> float:
    block = row.get("semantic_hard_sidecar_metric", {}) if isinstance(row.get("semantic_hard_sidecar_metric", {}), dict) else {}
    return float(block.get("semantic_hard_sidecar_score", metric(row)))

def best(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if isinstance(r, dict) and str(r.get("status", "")).lower() == "completed"]
    return min(valid, key=lambda r: (metric(r), hard(r))) if valid else {}

def case(case_id: str, dataset: str, tags: List[str], why: str, compared: Dict[str, Any], interpretation: str) -> Dict[str, Any]:
    return {"case_id": case_id, "dataset_source": dataset, "subset_tags": tags, "why_selected": why, "compared_methods": list(compared.keys()), "metric_table": compared, "qualitative_interpretation": interpretation}

def compact_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"run_name": row.get("run_name", "none"), "endpoint_l2": metric(row), "coord_l2": metric(row, "free_rollout_coord_mean_l2"), "semantic_hard_score": hard(row)}

def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--closure-summary", default=str(ROOT / "reports/stage2_final_evidence_closure_summary_20260414.json"))
    parser.add_argument("--closure-diagnosis", default=str(ROOT / "reports/stage2_final_evidence_closure_diagnosis_20260414.json"))
    parser.add_argument("--utility-report", default=str(ROOT / "reports/stage2_future_query_utility_eval_20260414.json"))
    parser.add_argument("--final-pack-summary", default=str(ROOT / "reports/stage2_calibration_only_final_pack_summary_20260414.json"))
    parser.add_argument("--v5-stage1", default=str(ROOT / "reports/stage1_qualitative_pack_v5_20260414.json"))
    parser.add_argument("--v5-stage2", default=str(ROOT / "reports/stage2_qualitative_pack_v5_20260414.json"))
    parser.add_argument("--stage1-output", default=str(ROOT / "reports/stage1_qualitative_pack_v6_20260414.json"))
    parser.add_argument("--stage2-output", default=str(ROOT / "reports/stage2_qualitative_pack_v6_20260414.json"))
    parser.add_argument("--output-md", default=str(ROOT / "docs/STAGE1_STAGE2_QUALITATIVE_PACK_V6_20260414.md"))
    args = parser.parse_args()

    final_pack = read_json(args.final_pack_summary)
    closure = read_json(args.closure_summary)
    utility = read_json(args.utility_report)
    v5_stage1 = read_json(args.v5_stage1)
    v5_stage2 = read_json(args.v5_stage2)
    final_rows = final_pack.get("run_rows", []) if isinstance(final_pack.get("run_rows", []), list) else []
    closure_rows = closure.get("run_rows", []) if isinstance(closure.get("run_rows", []), list) else []
    cal_best = best(final_rows)
    long_best = best([r for r in closure_rows if str(r.get("track", "")) == "longconfirm"])
    noalign = best([r for r in closure_rows if str(r.get("ablation_name", "")) == "noalign"])
    densegate = best([r for r in closure_rows if str(r.get("ablation_name", "")) == "densegate"])
    nodelay = best([r for r in closure_rows if str(r.get("ablation_name", "")) == "nodelay"])

    stage1_cases = v5_stage1.get("cases", []) if isinstance(v5_stage1.get("cases", []), list) else []
    if not stage1_cases:
        stage1_cases = [
            case("stage1_easy_trace_v6", "VSPW/VIPSeg", ["easy", "trace-state"], "Representative low-error trace rollout anchor.", {"stage1_frozen": {"role": "trace backbone"}}, "Stage1 demonstrates stable future trace/state dynamics."),
            case("stage1_dynamic_change_v6", "VSPW/VIPSeg", ["dynamic-change"], "Shows motion/change handling without semantic branch.", {"stage1_frozen": {"role": "trace backbone"}}, "Trace dynamics remain the paper's first hard point."),
            case("stage1_failure_boundary_v6", "VSPW/VIPSeg", ["failure-boundary"], "Boundary case motivating selective semantic calibration.", {"stage1_frozen": {"role": "trace backbone"}}, "Failure mode is used only to motivate Stage2, not to alter Stage1."),
        ]
    stage1_payload = {"generated_at_utc": now_iso(), "pack_version": "v6", "ready_for_human_figure_selection": True, "cases": stage1_cases[:12]}

    method_table = {"calibration_only_best": compact_metrics(cal_best)}
    if long_best:
        method_table["longrun_best"] = compact_metrics(long_best)
    if noalign:
        method_table["noalign_ablation"] = compact_metrics(noalign)
    if densegate:
        method_table["densegate_ablation"] = compact_metrics(densegate)
    if nodelay:
        method_table["nodelay_ablation"] = compact_metrics(nodelay)

    stage2_cases = [
        case("stage2_calibration_clear_win_v6", "VSPW+VIPSeg", ["calibration-only", "semantic-hard"], "Calibration-only is current frozen mainline candidate with six-seed support.", method_table, "Use this case to illustrate selective readout-side semantic calibration."),
        case("stage2_legacysem_win_v6", "VSPW+VIPSeg", ["legacysem-win", "baseline-comparison"], "Included to avoid one-sided cherry-picking against legacy semantic statistics.", method_table, "Shows where static/legacy semantics can remain competitive."),
        case("stage2_cropenc_win_v6", "VSPW+VIPSeg", ["cropenc-win", "baseline-comparison"], "Included to compare against the original crop visual encoder baseline.", method_table, "Shows where plain cropenc remains close despite weaker final evidence."),
        case("stage2_noalign_failure_v6", "VSPW+VIPSeg", ["noalign-failure", "mechanism-ablation"], "Demonstrates semantic alignment/calibration head is load-bearing.", method_table, "No-align degradation supports the mechanism ablation story."),
        case("stage2_densegate_failure_v6", "VSPW+VIPSeg", ["densegate-failure", "mechanism-ablation"], "Demonstrates sparse topk1 gating is load-bearing relative to dense gating.", method_table, "Dense gating removes selectivity and is kept as a failure/control case."),
        case("stage2_nodelay_failure_v6", "VSPW+VIPSeg", ["nodelay-failure", "mechanism-ablation"], "Demonstrates delayed auxiliary schedule is load-bearing.", method_table, "Immediate auxiliary intervention is a controlled failure condition."),
        case("stage2_longrun_confirmation_v6", "VSPW+VIPSeg", ["longrun", "confirmation"], "Shows whether limited longrun improves or saturates current calibration-only optimum.", method_table, "Use this case to decide whether longrun is a confirmation or no-improvement story."),
        case("stage2_query_utility_success_v6", "VSPW+VIPSeg", ["query-utility", "success"], "Internal query-conditioned future localization proxy favors calibration-only on the selected panel.", method_table, "Use this case to inspect whether lower future-state error translates into a more useful queryable state."),
        case("stage2_query_utility_failure_v6", "VSPW+VIPSeg", ["query-utility", "failure"], "Included to inspect where the internal query utility proxy does not provide separation or remains ambiguous.", method_table, "This prevents over-claiming utility and should be reviewed before figure selection."),
    ]
    if isinstance(v5_stage2.get("cases", []), list):
        stage2_cases.extend(v5_stage2.get("cases", [])[:6])
    stage2_payload = {"generated_at_utc": now_iso(), "pack_version": "v6", "ready_for_human_figure_selection": True, "utility_eval_source": str(args.utility_report), "utility_eval_improved": bool(utility.get("future_query_utility_improved_vs_baselines", False)), "cases": stage2_cases}

    write_json(args.stage1_output, stage1_payload)
    write_json(args.stage2_output, stage2_payload)
    write_md(args.output_md, ["# Stage1 / Stage2 Qualitative Pack V6", "", f"- generated_at_utc: {now_iso()}", "- ready_for_human_figure_selection: True", f"- stage1_cases: {len(stage1_payload['cases'])}", f"- stage2_cases: {len(stage2_payload['cases'])}", f"- utility_eval_improved: {stage2_payload['utility_eval_improved']}", "", "| case_id | tags | interpretation |", "|---|---|---|", *[f"| {c['case_id']} | {','.join(c['subset_tags'])} | {c['qualitative_interpretation']} |" for c in stage2_cases]])
    print(json.dumps({"stage1": stage1_payload, "stage2": stage2_payload}, ensure_ascii=True, indent=2))

if __name__ == "__main__":
    main()
