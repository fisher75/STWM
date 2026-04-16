#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List
import json


def _repo_root() -> Path:
    for candidate in [Path("/raid/chen034/workspace/stwm"), Path("/home/chen034/workspace/stwm")]:
        if candidate.exists():
            return candidate
    raise RuntimeError("unable to resolve STWM repo root")


ROOT = _repo_root()


def now_iso() -> str:
    from datetime import datetime, timezone
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
    state_eval = read_json(args.state_ident_eval_v3)
    mechanism_diag = read_json(args.mechanism_fix_v3_diagnosis)
    runtime_bench = read_json(args.runtime_benchmark)
    local_temporal_diag = read_json(args.local_temporal_diagnosis)
    stage2_qual = read_json(args.stage2_qual_v9)
    aux_probe = read_json(args.aux_probe_v2)

    payload = {
        "generated_at_utc": now_iso(),
        "sources": {
            "final_pack_diagnosis": str(args.final_pack_diagnosis),
            "state_identifiability_eval_v3": str(args.state_ident_eval_v3),
            "mechanism_fix_v3_diagnosis": str(args.mechanism_fix_v3_diagnosis),
            "runtime_benchmark": str(args.runtime_benchmark),
            "local_temporal_diagnosis": str(args.local_temporal_diagnosis),
            "stage2_qualitative_pack_v9": str(args.stage2_qual_v9),
            "aux_probe_v2": str(args.aux_probe_v2),
        },
        "mainline_still_calibration_only": bool(final_pack_diag.get("calibration_only_is_final_stage2_mainline", False)),
        "state_identifiability_protocol_v3_success": bool(state_eval.get("state_identifiability_protocol_v3_success", False)),
        "protocol_v3_discriminative_enough_for_top_tier": bool(state_eval.get("protocol_v3_discriminative_enough_for_top_tier", False)),
        "future_grounding_usefulness_improved_vs_baselines": bool(state_eval.get("future_grounding_usefulness_improved_vs_baselines", False)),
        "future_grounding_usefulness_improved_on_hard_subsets": bool(state_eval.get("future_grounding_usefulness_improved_on_hard_subsets", False)),
        "alignment_load_bearing_cross_seed": bool(mechanism_diag.get("alignment_load_bearing_cross_seed", False)),
        "sparse_gating_load_bearing_cross_seed": bool(mechanism_diag.get("sparse_gating_load_bearing_cross_seed", False)),
        "delayed_schedule_load_bearing_cross_seed": bool(mechanism_diag.get("delayed_schedule_load_bearing_cross_seed", False)),
        "runtime_bottleneck_relieved": bool(runtime_bench.get("runtime_bottleneck_relieved", False)),
        "local_temporal_branch_promising": bool(local_temporal_diag.get("local_temporal_branch_promising", False)),
        "qualitative_pack_ready_for_paper_figure_selection": bool(stage2_qual.get("ready_for_paper_figure_selection", False)),
        "qualitative_pack_ready_for_oral_backup_figure_selection": bool(stage2_qual.get("ready_for_oral_backup_figure_selection", False)),
        "aux_probe_is_only_auxiliary": bool(aux_probe.get("adapter_probe_only", False) and aux_probe.get("paper_official_benchmark", True) is False),
    }
    write_json(args.summary_report, payload)
    return payload


def diagnose(args: Any) -> Dict[str, Any]:
    summary = summarize(args)
    mech_all = bool(summary["alignment_load_bearing_cross_seed"] and summary["sparse_gating_load_bearing_cross_seed"] and summary["delayed_schedule_load_bearing_cross_seed"])
    current_stage2_ready_to_freeze = bool(
        summary["mainline_still_calibration_only"]
        and summary["state_identifiability_protocol_v3_success"]
        and summary["protocol_v3_discriminative_enough_for_top_tier"]
        and summary["future_grounding_usefulness_improved_vs_baselines"]
        and summary["future_grounding_usefulness_improved_on_hard_subsets"]
        and mech_all
        and summary["runtime_bottleneck_relieved"]
        and summary["qualitative_pack_ready_for_paper_figure_selection"]
    )
    if current_stage2_ready_to_freeze:
        next_step_choice = "freeze_stage2_calibration_only_mainline"
        thickness = "oral_ready"
    elif summary["local_temporal_branch_promising"] and not mech_all:
        next_step_choice = "stage2_local_temporal_branch_wave1"
        thickness = "oral_borderline"
    elif not summary["state_identifiability_protocol_v3_success"] or not summary["future_grounding_usefulness_improved_vs_baselines"]:
        next_step_choice = "reconsider_stage2_only_if_identifiability_fails"
        thickness = "cvpr_borderline"
    elif not summary["protocol_v3_discriminative_enough_for_top_tier"]:
        next_step_choice = "run_one_more_protocol_v4_scaleup"
        thickness = "cvpr_borderline"
    elif not mech_all:
        next_step_choice = "run_one_more_targeted_ablation_fix"
        thickness = "oral_borderline" if summary["qualitative_pack_ready_for_paper_figure_selection"] else "cvpr_ready"
    elif not summary["runtime_bottleneck_relieved"]:
        next_step_choice = "run_one_more_targeted_ablation_fix"
        thickness = "oral_borderline"
    else:
        next_step_choice = "freeze_stage2_calibration_only_mainline"
        thickness = "oral_ready"
    payload = {
        **summary,
        "current_stage2_ready_to_freeze": bool(current_stage2_ready_to_freeze),
        "current_paper_thickness_level": thickness,
        "next_step_choice": next_step_choice,
    }
    write_json(args.diagnosis_report, payload)
    write_md(
        args.results_md,
        [
            "# Stage2 Oral Hardening 20260416",
            "",
            f"- mainline_still_calibration_only: {payload['mainline_still_calibration_only']}",
            f"- state_identifiability_protocol_v3_success: {payload['state_identifiability_protocol_v3_success']}",
            f"- protocol_v3_discriminative_enough_for_top_tier: {payload['protocol_v3_discriminative_enough_for_top_tier']}",
            f"- future_grounding_usefulness_improved_vs_baselines: {payload['future_grounding_usefulness_improved_vs_baselines']}",
            f"- future_grounding_usefulness_improved_on_hard_subsets: {payload['future_grounding_usefulness_improved_on_hard_subsets']}",
            f"- alignment_load_bearing_cross_seed: {payload['alignment_load_bearing_cross_seed']}",
            f"- sparse_gating_load_bearing_cross_seed: {payload['sparse_gating_load_bearing_cross_seed']}",
            f"- delayed_schedule_load_bearing_cross_seed: {payload['delayed_schedule_load_bearing_cross_seed']}",
            f"- runtime_bottleneck_relieved: {payload['runtime_bottleneck_relieved']}",
            f"- local_temporal_branch_promising: {payload['local_temporal_branch_promising']}",
            f"- qualitative_pack_ready_for_paper_figure_selection: {payload['qualitative_pack_ready_for_paper_figure_selection']}",
            f"- current_stage2_ready_to_freeze: {payload['current_stage2_ready_to_freeze']}",
            f"- current_paper_thickness_level: {payload['current_paper_thickness_level']}",
            f"- next_step_choice: {payload['next_step_choice']}",
        ],
    )
    return payload


def parse_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument("--final-pack-diagnosis", default=str(ROOT / "reports/stage2_calibration_only_final_pack_diagnosis_20260414.json"))
    parser.add_argument("--state-ident-eval-v3", default=str(ROOT / "reports/stage2_state_identifiability_eval_v3_20260416.json"))
    parser.add_argument("--mechanism-fix-v3-diagnosis", default=str(ROOT / "reports/stage2_mechanism_ablation_fix_v3_diagnosis_20260416.json"))
    parser.add_argument("--runtime-benchmark", default=str(ROOT / "reports/stage2_runtime_pipeline_benchmark_20260416.json"))
    parser.add_argument("--local-temporal-diagnosis", default=str(ROOT / "reports/stage2_local_temporal_semantic_branch_diagnosis_20260416.json"))
    parser.add_argument("--stage2-qual-v9", default=str(ROOT / "reports/stage2_qualitative_pack_v9_20260416.json"))
    parser.add_argument("--aux-probe-v2", default=str(ROOT / "reports/stage2_aux_external_probe_batch_v2_20260414.json"))
    parser.add_argument("--summary-report", default=str(ROOT / "reports/stage2_oral_hardening_summary_20260416.json"))
    parser.add_argument("--diagnosis-report", default=str(ROOT / "reports/stage2_oral_hardening_diagnosis_20260416.json"))
    parser.add_argument("--results-md", default=str(ROOT / "docs/STAGE2_ORAL_HARDENING_20260416.md"))
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
