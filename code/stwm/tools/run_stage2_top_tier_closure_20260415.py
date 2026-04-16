#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
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


def summarize(args: Any) -> Dict[str, Any]:
    final_pack_diag = read_json(args.final_pack_diagnosis)
    state_eval = read_json(args.state_ident_eval)
    mechanism_diag = read_json(args.mechanism_fix_diagnosis)
    stage2_qual = read_json(args.stage2_qual)
    aux_probe = read_json(args.aux_probe)

    payload = {
        "generated_at_utc": now_iso(),
        "sources": {
            "final_pack_diagnosis": str(args.final_pack_diagnosis),
            "state_identifiability_eval": str(args.state_ident_eval),
            "mechanism_fix_diagnosis": str(args.mechanism_fix_diagnosis),
            "stage2_qualitative_pack_v7": str(args.stage2_qual),
            "aux_probe_v2": str(args.aux_probe),
        },
        "mainline_still_calibration_only": bool(final_pack_diag.get("calibration_only_is_final_stage2_mainline", False)),
        "state_identifiability_protocol_success": bool(state_eval.get("state_identifiability_protocol_success", False)),
        "future_grounding_usefulness_improved_vs_baselines": bool(state_eval.get("future_grounding_usefulness_improved_vs_baselines", False)),
        "alignment_load_bearing_cross_seed": bool(mechanism_diag.get("alignment_load_bearing_cross_seed", False)),
        "sparse_gating_load_bearing_cross_seed": bool(mechanism_diag.get("sparse_gating_load_bearing_cross_seed", False)),
        "delayed_schedule_load_bearing_cross_seed": bool(mechanism_diag.get("delayed_schedule_load_bearing_cross_seed", False)),
        "qualitative_pack_ready_for_paper_figure_selection": bool(stage2_qual.get("ready_for_paper_figure_selection", False)),
        "aux_probe_is_only_auxiliary": bool(aux_probe.get("adapter_probe_only", False) and aux_probe.get("paper_official_benchmark", True) is False),
    }
    write_json(args.summary_report, payload)
    return payload


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    state_ok = bool(summary.get("state_identifiability_protocol_success", False))
    utility_ok = bool(summary.get("future_grounding_usefulness_improved_vs_baselines", False))
    align_ok = bool(summary.get("alignment_load_bearing_cross_seed", False))
    sparse_ok = bool(summary.get("sparse_gating_load_bearing_cross_seed", False))
    delay_ok = bool(summary.get("delayed_schedule_load_bearing_cross_seed", False))
    qual_ok = bool(summary.get("qualitative_pack_ready_for_paper_figure_selection", False))
    current_stage2_ready_to_freeze = bool(
        summary.get("mainline_still_calibration_only", False)
        and state_ok
        and utility_ok
        and align_ok
        and sparse_ok
        and delay_ok
        and qual_ok
    )
    if not state_ok or not utility_ok:
        next_step_choice = "run_one_more_targeted_identifiability_fix"
    elif not (align_ok and sparse_ok and delay_ok):
        next_step_choice = "run_one_more_targeted_ablation_fix"
    elif current_stage2_ready_to_freeze:
        next_step_choice = "freeze_stage2_calibration_only_mainline"
    else:
        next_step_choice = "reconsider_stage2_only_if_identifiability_fails"
    payload = {
        **summary,
        "current_stage2_ready_to_freeze": bool(current_stage2_ready_to_freeze),
        "next_step_choice": next_step_choice,
    }
    write_json(args.diagnosis_report, payload)
    write_md(
        args.results_md,
        [
            "# Stage2 Top-tier Closure 20260415",
            "",
            f"- mainline_still_calibration_only: {payload['mainline_still_calibration_only']}",
            f"- state_identifiability_protocol_success: {payload['state_identifiability_protocol_success']}",
            f"- future_grounding_usefulness_improved_vs_baselines: {payload['future_grounding_usefulness_improved_vs_baselines']}",
            f"- alignment_load_bearing_cross_seed: {payload['alignment_load_bearing_cross_seed']}",
            f"- sparse_gating_load_bearing_cross_seed: {payload['sparse_gating_load_bearing_cross_seed']}",
            f"- delayed_schedule_load_bearing_cross_seed: {payload['delayed_schedule_load_bearing_cross_seed']}",
            f"- qualitative_pack_ready_for_paper_figure_selection: {payload['qualitative_pack_ready_for_paper_figure_selection']}",
            f"- current_stage2_ready_to_freeze: {payload['current_stage2_ready_to_freeze']}",
            f"- next_step_choice: {payload['next_step_choice']}",
        ],
    )
    return payload


def parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument("--final-pack-diagnosis", default=str(ROOT / "reports/stage2_calibration_only_final_pack_diagnosis_20260414.json"))
    parser.add_argument("--state-ident-eval", default=str(ROOT / "reports/stage2_state_identifiability_eval_20260415.json"))
    parser.add_argument("--mechanism-fix-diagnosis", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_diagnosis_20260415.json"))
    parser.add_argument("--stage2-qual", default=str(ROOT / "reports/stage2_qualitative_pack_v7_20260415.json"))
    parser.add_argument("--aux-probe", default=str(ROOT / "reports/stage2_aux_external_probe_batch_v2_20260414.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_top_tier_closure_summary_20260415.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_top_tier_closure_diagnosis_20260415.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_TOP_TIER_CLOSURE_20260415.md"))
    parser.add_argument("--mode", default="all", choices=["all", "summarize", "diagnose"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "all":
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))
    elif args.mode == "summarize":
        print(json.dumps(summarize(args), ensure_ascii=True, indent=2))
    elif args.mode == "diagnose":
        print(json.dumps(diagnose(args), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
