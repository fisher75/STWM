#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
import json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"json not found: {p}")
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json payload must be dict: {p}")
    return payload


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _write_md(path: str | Path, lines: List[str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metric(payload: Dict[str, Any], key: str, default: float = 1e9) -> float:
    try:
        return float(payload.get(key, default))
    except Exception:
        return float(default)


def parse_args() -> Any:
    p = ArgumentParser(description="Stage2 external-eval claim-boundary fix round")
    p.add_argument(
        "--completion-json",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_20260408.json",
    )
    p.add_argument(
        "--completion-md",
        default="/home/chen034/workspace/stwm/docs/STAGE2_EXTERNAL_EVAL_COMPLETION_RESULTS_20260408.md",
    )
    p.add_argument(
        "--fidelity-json",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_fidelity_audit_20260409.json",
    )
    p.add_argument(
        "--fidelity-md",
        default="/home/chen034/workspace/stwm/docs/STAGE2_EXTERNAL_EVAL_FIDELITY_AUDIT_20260409.md",
    )
    p.add_argument(
        "--readiness-json",
        default="/home/chen034/workspace/stwm/reports/tracewm_project_readiness_20260409.json",
    )
    p.add_argument(
        "--readiness-md",
        default="/home/chen034/workspace/stwm/docs/TRACEWM_PROJECT_READINESS_20260409.md",
    )
    p.add_argument(
        "--claim-boundary-md",
        default="/home/chen034/workspace/stwm/docs/STAGE2_EXTERNAL_EVAL_CLAIM_BOUNDARY_20260409.md",
    )
    p.add_argument(
        "--guardrails-json",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_result_interpretation_guardrails_20260409.json",
    )
    p.add_argument(
        "--guardrails-md",
        default="/home/chen034/workspace/stwm/docs/STAGE2_EXTERNAL_EVAL_RESULT_INTERPRETATION_GUARDRAILS_20260409.md",
    )
    p.add_argument(
        "--consistency-json",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_status_consistency_20260409.json",
    )
    return p.parse_args()


def _status_rule() -> Dict[str, str]:
    return {
        "fully_implemented_and_run": "Requires official_evaluator_invoked=true and official_task_faithfully_instantiated=true, meaning the official evaluator ran on a benchmark-faithful task instantiation.",
        "partially_bridged": "Allowed only when the official TAP-Vid evaluator successfully ran on an adapter-converted payload and returned metrics, but official_task_faithfully_instantiated=false remains true.",
        "proxy_only": "Required when only proxy bridge outputs exist, or when no successful official TAP-Vid evaluator invocation/result is available on adapter payload.",
        "not_yet_implemented": "Use when neither proxy bridge nor official-evaluator-side adapter path is operational enough to produce a meaningful probe.",
    }


def _derive_tap_style_claims(completion: Dict[str, Any]) -> Dict[str, Any]:
    primary = completion.get("primary_checkpoint_eval", {}) if isinstance(completion.get("primary_checkpoint_eval", {}), dict) else {}
    tap = primary.get("tap_style_eval", {}) if isinstance(primary.get("tap_style_eval", {}), dict) else {}
    official_eval = tap.get("official_eval", {}) if isinstance(tap.get("official_eval", {}), dict) else {}
    official_payload = tap.get("official_payload_export", {}) if isinstance(tap.get("official_payload_export", {}), dict) else {}
    rule = _status_rule()

    proxy_bridge_connected = bool(tap.get("proxy_bridge_connected", False))
    official_evaluator_invoked = bool(tap.get("official_evaluator_invoked", False))
    official_tapvid_evaluator_connected = bool(tap.get("official_tapvid_evaluator_connected", False))
    official_task_faithfully_instantiated = bool(tap.get("official_task_faithfully_instantiated", False))
    dataset_family_match = bool(tap.get("dataset_binding_is_official_tap_dataset_family", False))
    query_protocol_match = bool(tap.get("query_time_matches_official_task", False))
    visibility_protocol_match = bool(tap.get("pred_visibility_from_model_output", False))
    official_payload_exists = Path(str(official_payload.get("output_tap_payload_npz", ""))).exists()
    official_metric_means = official_eval.get("metric_means", {}) if isinstance(official_eval.get("metric_means", {}), dict) else {}
    official_metric_result_present = len(official_metric_means) > 0

    if official_evaluator_invoked and official_task_faithfully_instantiated:
        status = "fully_implemented_and_run"
    elif official_evaluator_invoked and official_tapvid_evaluator_connected and official_payload_exists and official_metric_result_present:
        status = "partially_bridged"
    elif proxy_bridge_connected:
        status = "proxy_only"
    else:
        status = "not_yet_implemented"

    why_not_proxy_only = ""
    if status == "partially_bridged":
        why_not_proxy_only = (
            "Kept as partially_bridged because the official TAP-Vid evaluator was actually invoked on an "
            "adapter-converted payload, returned metric tensors, and therefore exceeds a pure proxy-only bridge. "
            "It is still not an official benchmark result because task faithfulness remains false."
        )

    current_metric_scope = (
        "adapter-based TAP-style probe: official TAP-Vid evaluator run on an adapter-converted, "
        "non-benchmark-faithful payload exported from the frozen Stage2 core-only VSPW+VIPSeg binding"
    )
    safest_sentence = (
        "We report an adapter-based TAP-style probe in which the official TAP-Vid evaluator is run on a converted "
        "payload from the frozen VSPW+VIPSeg Stage2 rollout; this is not an official TAP-Vid benchmark result."
    )

    return {
        "tap_style_eval_status": status,
        "status_boundary_rule": rule,
        "why_partially_bridged_not_proxy_only": why_not_proxy_only,
        "current_metric_scope": current_metric_scope,
        "official_benchmark_equivalent": bool(status == "fully_implemented_and_run"),
        "dataset_family_match": dataset_family_match,
        "query_protocol_match": query_protocol_match,
        "visibility_protocol_match": visibility_protocol_match,
        "official_evaluator_invoked": official_evaluator_invoked,
        "official_tapvid_evaluator_connected": official_tapvid_evaluator_connected,
        "official_task_faithfully_instantiated": official_task_faithfully_instantiated,
        "adapter_probe_only": bool(status != "fully_implemented_and_run"),
        "paper_official_benchmark": bool(status == "fully_implemented_and_run"),
        "allowed_paper_usage": [
            "Can be reported as an adapter-based TAP-style probe or proxy-style external tracking probe.",
            "Can be used as supporting evidence that the frozen Stage2 rollout can be exported into an official evaluator interface.",
            "Can be discussed in text or appendix as boundary-checked external evidence, with explicit non-official labeling.",
        ],
        "forbidden_paper_usage": [
            "Do not call these numbers official TAP-Vid benchmark results.",
            "Do not place these numbers into a main table labeled TAP-Vid benchmark or official external benchmark.",
            "Do not compare these numbers directly against papers evaluated on the official TAP-Vid dataset family and protocol as if they were commensurate.",
            "Do not claim TAPVid-3D results are obtained.",
        ],
        "safest_one_sentence_description_for_paper": safest_sentence,
    }


def _apply_claim_labels_to_checkpoint_eval(checkpoint_eval: Dict[str, Any], claims: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(checkpoint_eval)
    tap = out.get("tap_style_eval", {}) if isinstance(out.get("tap_style_eval", {}), dict) else {}
    tap = dict(tap)
    tap["status"] = str(claims["tap_style_eval_status"])
    tap["claim_scope_name"] = "adapter_based_tap_style_probe"
    tap["adapter_probe_only"] = bool(claims["adapter_probe_only"])
    tap["paper_official_benchmark"] = bool(claims["paper_official_benchmark"])
    tap["current_metric_scope"] = str(claims["current_metric_scope"])
    tap["status_boundary_rule"] = claims["status_boundary_rule"]
    tap["why_partially_bridged_not_proxy_only"] = str(claims["why_partially_bridged_not_proxy_only"])
    tap["allowed_paper_usage"] = claims["allowed_paper_usage"]
    tap["forbidden_paper_usage"] = claims["forbidden_paper_usage"]
    official_eval = tap.get("official_eval", {}) if isinstance(tap.get("official_eval", {}), dict) else {}
    official_eval = dict(official_eval)
    official_eval["adapter_probe_only"] = bool(claims["adapter_probe_only"])
    official_eval["paper_official_benchmark"] = bool(claims["paper_official_benchmark"])
    tap["official_eval"] = official_eval
    out["tap_style_eval"] = tap
    return out


def _build_completion_payload(completion: Dict[str, Any], claims: Dict[str, Any], next_step_choice: str) -> Dict[str, Any]:
    updated = dict(completion)
    updated["generated_at_utc"] = now_iso()
    updated["tap_style_eval_status"] = str(claims["tap_style_eval_status"])
    updated["official_evaluator_invoked"] = bool(claims["official_evaluator_invoked"])
    updated["official_task_faithfully_instantiated"] = bool(claims["official_task_faithfully_instantiated"])
    updated["paper_official_benchmark"] = bool(claims["paper_official_benchmark"])
    updated["adapter_probe_only"] = bool(claims["adapter_probe_only"])
    updated["current_metric_scope"] = str(claims["current_metric_scope"])
    updated["status_boundary_rule"] = claims["status_boundary_rule"]
    updated["why_partially_bridged_not_proxy_only"] = str(claims["why_partially_bridged_not_proxy_only"])
    updated["allowed_paper_usage"] = claims["allowed_paper_usage"]
    updated["forbidden_paper_usage"] = claims["forbidden_paper_usage"]
    updated["safest_one_sentence_description_for_paper"] = str(claims["safest_one_sentence_description_for_paper"])
    updated["external_eval_readiness"] = "training_ready_but_eval_gap_remains"
    updated["next_step_choice"] = next_step_choice
    updated["primary_checkpoint_eval"] = _apply_claim_labels_to_checkpoint_eval(
        completion.get("primary_checkpoint_eval", {}) if isinstance(completion.get("primary_checkpoint_eval", {}), dict) else {},
        claims,
    )
    secondary = completion.get("secondary_checkpoint_eval", {})
    if isinstance(secondary, dict) and secondary:
        updated["secondary_checkpoint_eval"] = _apply_claim_labels_to_checkpoint_eval(secondary, claims)
    compare = updated.get("best_vs_latest_reference", {}) if isinstance(updated.get("best_vs_latest_reference", {}), dict) else {}
    if compare:
        compare = dict(compare)
        compare["comparison_scope"] = "adapter_based_tap_style_probe_only"
        compare["adapter_probe_only"] = bool(claims["adapter_probe_only"])
        compare["paper_official_benchmark"] = bool(claims["paper_official_benchmark"])
        updated["best_vs_latest_reference"] = compare
    return updated


def _build_fidelity_payload(fidelity: Dict[str, Any], claims: Dict[str, Any], completion_payload: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(fidelity)
    updated["generated_at_utc"] = now_iso()
    updated["tap_style_eval_status"] = str(claims["tap_style_eval_status"])
    updated["official_evaluator_invoked"] = bool(claims["official_evaluator_invoked"])
    updated["official_task_faithfully_instantiated"] = bool(claims["official_task_faithfully_instantiated"])
    updated["paper_official_benchmark"] = bool(claims["paper_official_benchmark"])
    updated["adapter_probe_only"] = bool(claims["adapter_probe_only"])
    updated["current_metric_scope"] = str(claims["current_metric_scope"])
    updated["official_benchmark_equivalent"] = bool(claims["official_benchmark_equivalent"])
    updated["dataset_family_match"] = bool(claims["dataset_family_match"])
    updated["query_protocol_match"] = bool(claims["query_protocol_match"])
    updated["visibility_protocol_match"] = bool(claims["visibility_protocol_match"])
    updated["status_boundary_rule"] = claims["status_boundary_rule"]
    updated["why_partially_bridged_not_proxy_only"] = str(claims["why_partially_bridged_not_proxy_only"])
    updated["allowed_paper_usage"] = claims["allowed_paper_usage"]
    updated["forbidden_paper_usage"] = claims["forbidden_paper_usage"]
    updated["safest_one_sentence_description_for_paper"] = str(claims["safest_one_sentence_description_for_paper"])
    updated["claim_boundary_hardened"] = True
    updated["exact_blocking_reasons"] = completion_payload.get("exact_blocking_reasons", [])
    return updated


def _build_guardrails_payload(claims: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "generated_at_utc": now_iso(),
        "current_metric_scope": str(claims["current_metric_scope"]),
        "official_benchmark_equivalent": bool(claims["official_benchmark_equivalent"]),
        "dataset_family_match": bool(claims["dataset_family_match"]),
        "query_protocol_match": bool(claims["query_protocol_match"]),
        "visibility_protocol_match": bool(claims["visibility_protocol_match"]),
        "adapter_probe_only": bool(claims["adapter_probe_only"]),
        "paper_official_benchmark": bool(claims["paper_official_benchmark"]),
        "allowed_paper_usage": claims["allowed_paper_usage"],
        "forbidden_paper_usage": claims["forbidden_paper_usage"],
        "safest_one_sentence_description_for_paper": str(claims["safest_one_sentence_description_for_paper"]),
        "why_partially_bridged_not_proxy_only": str(claims["why_partially_bridged_not_proxy_only"]),
    }


def _build_readiness_payload(readiness: Dict[str, Any], completion_payload: Dict[str, Any], claims: Dict[str, Any], next_step_choice: str) -> Dict[str, Any]:
    updated = dict(readiness)
    updated["generated_at_utc"] = now_iso()
    updated["current_stage2_mainline_still_valid"] = bool(readiness.get("current_stage2_mainline_still_valid", False))
    updated["dataset_evidence_bundle_ready"] = bool(readiness.get("dataset_evidence_bundle_ready", False))
    updated["official_evaluator_invoked"] = bool(claims["official_evaluator_invoked"])
    updated["official_task_faithfully_instantiated"] = bool(claims["official_task_faithfully_instantiated"])
    updated["tap_style_eval_status"] = str(claims["tap_style_eval_status"])
    updated["tap3d_style_eval_status"] = str(completion_payload.get("tap3d_style_eval_status", "not_yet_implemented"))
    updated["paper_official_benchmark"] = bool(claims["paper_official_benchmark"])
    updated["claim_boundary_hardened"] = True
    updated["external_eval_paper_grade"] = bool(claims["paper_official_benchmark"]) and str(completion_payload.get("tap3d_style_eval_status", "")) == "fully_implemented_and_run"
    updated["project_readiness"] = "training_ready_but_eval_gap_remains"
    updated["next_step_choice"] = next_step_choice
    return updated


def _completion_md_lines(payload: Dict[str, Any]) -> List[str]:
    primary = payload.get("primary_checkpoint_eval", {}) if isinstance(payload.get("primary_checkpoint_eval", {}), dict) else {}
    secondary = payload.get("secondary_checkpoint_eval", {}) if isinstance(payload.get("secondary_checkpoint_eval", {}), dict) else {}
    primary_tap = primary.get("tap_style_eval", {}) if isinstance(primary.get("tap_style_eval", {}), dict) else {}
    primary_official = primary_tap.get("official_eval", {}) if isinstance(primary_tap.get("official_eval", {}), dict) else {}
    primary_metric_means = primary_official.get("metric_means", {}) if isinstance(primary_official.get("metric_means", {}), dict) else {}
    compare = payload.get("best_vs_latest_reference", {}) if isinstance(payload.get("best_vs_latest_reference", {}), dict) else {}
    lines = [
        "# Stage2 External Eval Completion Results",
        "",
        "> This document is claim-boundary-hardened completion status only. The frozen Stage2 mainline was not retrained in this round.",
        "",
        "## Locked Facts",
        f"- current_stage2_mainline_checkpoint: {payload.get('current_stage2_mainline_checkpoint', '')}",
        f"- secondary_checkpoint_reference: {payload.get('secondary_checkpoint_reference', '')}",
        f"- datasets_bound_for_eval: {payload.get('datasets_bound_for_eval', [])}",
        f"- current_mainline_semantic_source: {payload.get('current_mainline_semantic_source', '')}",
        f"- frozen_boundary_kept_correct: {bool(payload.get('frozen_boundary_kept_correct', False))}",
        "",
        "## Claim Boundary",
        f"- tap_style_eval_status: {payload.get('tap_style_eval_status', '')}",
        f"- official_evaluator_invoked: {bool(payload.get('official_evaluator_invoked', False))}",
        f"- official_task_faithfully_instantiated: {bool(payload.get('official_task_faithfully_instantiated', False))}",
        f"- paper_official_benchmark: {bool(payload.get('paper_official_benchmark', False))}",
        f"- current_metric_scope: {payload.get('current_metric_scope', '')}",
        f"- why_partially_bridged_not_proxy_only: {payload.get('why_partially_bridged_not_proxy_only', '')}",
        "",
        "## Adapter-Based TAP-Style Probe",
        f"- claim_scope_name: {primary_tap.get('claim_scope_name', '')}",
        f"- adapter_probe_only: {bool(primary_tap.get('adapter_probe_only', False))}",
        f"- paper_official_benchmark: {bool(primary_tap.get('paper_official_benchmark', False))}",
        f"- primary_checkpoint_path: {primary.get('checkpoint_path', '')}",
        f"- average_jaccard: {float(primary_metric_means.get('average_jaccard', 1e9)):.6f}",
        f"- average_pts_within_thresh: {float(primary_metric_means.get('average_pts_within_thresh', 1e9)):.6f}",
        f"- occlusion_accuracy: {float(primary_metric_means.get('occlusion_accuracy', 1e9)):.6f}",
        "",
        "## TAP-Style Remaining Gaps",
    ]
    for item in primary_tap.get("exact_blocking_reasons", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Best vs Latest Reference",
            f"- comparison_scope: {compare.get('comparison_scope', '')}",
            f"- adapter_probe_only: {bool(compare.get('adapter_probe_only', False))}",
            f"- paper_official_benchmark: {bool(compare.get('paper_official_benchmark', False))}",
            f"- free_rollout_coord_mean_l2_best: {float(compare.get('free_rollout_coord_mean_l2_best', 1e9)):.6f}",
            f"- free_rollout_coord_mean_l2_latest: {float(compare.get('free_rollout_coord_mean_l2_latest', 1e9)):.6f}",
            f"- tapvid_average_jaccard_best: {float(compare.get('tapvid_average_jaccard_best', 1e9)):.6f}",
            f"- tapvid_average_jaccard_latest: {float(compare.get('tapvid_average_jaccard_latest', 1e9)):.6f}",
            "",
            "## Secondary Reference Label",
            f"- secondary_exists: {bool(isinstance(secondary, dict) and secondary)}",
            "- secondary numbers remain adapter-probe-only and must not be treated as official benchmark values.",
            "",
            "## Mandatory Answers",
            f"1. current mainline checkpoint is still `best.pt`: {str(payload.get('current_stage2_mainline_checkpoint', '')).endswith('/best.pt')}",
            f"2. TAP-style is currently: `{payload.get('tap_style_eval_status', '')}`",
            f"3. official TAP evaluator connected: {bool(payload.get('official_tapvid_evaluator_connected', False))}",
            f"4. TAP3D-style progressed to: `{payload.get('tap3d_style_eval_status', '')}`",
            f"5. project readiness is: `{payload.get('external_eval_readiness', '')}`",
        ]
    )
    return lines


def _fidelity_md_lines(payload: Dict[str, Any]) -> List[str]:
    lines = [
        "# Stage2 External Eval Fidelity Audit",
        "",
        f"- current_stage2_mainline_checkpoint: {payload.get('current_stage2_mainline_checkpoint', '')}",
        f"- tap_style_eval_status: {payload.get('tap_style_eval_status', '')}",
        f"- tap3d_style_eval_status: {payload.get('tap3d_style_eval_status', '')}",
        f"- official_evaluator_invoked: {bool(payload.get('official_evaluator_invoked', False))}",
        f"- official_task_faithfully_instantiated: {bool(payload.get('official_task_faithfully_instantiated', False))}",
        f"- paper_official_benchmark: {bool(payload.get('paper_official_benchmark', False))}",
        f"- official_benchmark_equivalent: {bool(payload.get('official_benchmark_equivalent', False))}",
        f"- why_partially_bridged_not_proxy_only: {payload.get('why_partially_bridged_not_proxy_only', '')}",
        "",
        "## TAP-Style Checks",
        f"- dataset_family_match: {bool(payload.get('dataset_family_match', False))}",
        f"- query_protocol_match: {bool(payload.get('query_protocol_match', False))}",
        f"- visibility_protocol_match: {bool(payload.get('visibility_protocol_match', False))}",
        "",
        "## Status Boundary Rule",
    ]
    for k, v in (payload.get("status_boundary_rule", {}) or {}).items():
        lines.append(f"- {k}: {v}")
    lines.extend(["", "## Blocking Reasons"])
    for reason in payload.get("exact_blocking_reasons", []):
        lines.append(f"- {reason}")
    lines.extend(["", "## Packaging Note"])
    log_row = payload.get("completion_log", {}) if isinstance(payload.get("completion_log", {}), dict) else {}
    lines.append(f"- completion_log_exists: {bool(log_row.get('exists', False))}")
    lines.append(f"- packaged_in_repo_snapshot: {bool(log_row.get('packaged_in_repo_snapshot', False))}")
    lines.append(f"- note: {log_row.get('note', '')}")
    return lines


def _readiness_md_lines(payload: Dict[str, Any]) -> List[str]:
    return [
        "# TRACEWM Project Readiness",
        "",
        f"- current_stage2_mainline_still_valid: {bool(payload.get('current_stage2_mainline_still_valid', False))}",
        f"- dataset_evidence_bundle_ready: {bool(payload.get('dataset_evidence_bundle_ready', False))}",
        f"- official_evaluator_invoked: {bool(payload.get('official_evaluator_invoked', False))}",
        f"- official_task_faithfully_instantiated: {bool(payload.get('official_task_faithfully_instantiated', False))}",
        f"- tap_style_eval_status: {payload.get('tap_style_eval_status', '')}",
        f"- tap3d_style_eval_status: {payload.get('tap3d_style_eval_status', '')}",
        f"- paper_official_benchmark: {bool(payload.get('paper_official_benchmark', False))}",
        f"- claim_boundary_hardened: {bool(payload.get('claim_boundary_hardened', False))}",
        f"- project_readiness: {payload.get('project_readiness', '')}",
        f"- next_step_choice: {payload.get('next_step_choice', '')}",
    ]


def _guardrails_md_lines(payload: Dict[str, Any]) -> List[str]:
    lines = [
        "# Stage2 External Eval Result Interpretation Guardrails",
        "",
        f"- current_metric_scope: {payload.get('current_metric_scope', '')}",
        f"- official_benchmark_equivalent: {bool(payload.get('official_benchmark_equivalent', False))}",
        f"- dataset_family_match: {bool(payload.get('dataset_family_match', False))}",
        f"- query_protocol_match: {bool(payload.get('query_protocol_match', False))}",
        f"- visibility_protocol_match: {bool(payload.get('visibility_protocol_match', False))}",
        f"- adapter_probe_only: {bool(payload.get('adapter_probe_only', False))}",
        f"- paper_official_benchmark: {bool(payload.get('paper_official_benchmark', False))}",
        "",
        "## Allowed Paper Usage",
    ]
    for item in payload.get("allowed_paper_usage", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Forbidden Paper Usage"])
    for item in payload.get("forbidden_paper_usage", []):
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Safest One-Sentence Description",
            f"- {payload.get('safest_one_sentence_description_for_paper', '')}",
        ]
    )
    return lines


def _claim_boundary_md_lines(completion_payload: Dict[str, Any], guardrails_payload: Dict[str, Any]) -> List[str]:
    return [
        "# Stage2 External Eval Claim Boundary",
        "",
        "- Current TAP-style numbers must be called `adapter-based TAP-style probe` or `proxy-style external tracking probe`.",
        "- They must not be called `official TAP-Vid benchmark result` because official_task_faithfully_instantiated remains false.",
        "- Current TAP-style numbers may be reported only as adapter-probe evidence that the frozen Stage2 rollout can be exported into the official evaluator interface.",
        "- Current TAP-style numbers must not be reported as commensurate official benchmark scores, must not be placed into a TAP-Vid main benchmark table, and must not be compared against benchmark-native TAP-Vid papers as if the protocol matched.",
        "- TAP3D remains `not_yet_implemented` because aligned 3D GT for the frozen VSPW+VIPSeg binding is absent, camera geometry / lifting path is absent, and a verified exporter to `tracks_XYZ + visibility` is absent.",
        "- The safest current paper wording for external eval is to say that we performed an adapter-based TAP-style probe by running the official TAP-Vid evaluator on a converted non-native payload from the frozen Stage2 rollout, and that this does not constitute an official TAP-Vid benchmark evaluation.",
        "",
        "## Forbidden Terms",
        "- `official benchmark completed`",
        "- `faithfully evaluated on TAP-Vid`",
        "- `official TAP-Vid benchmark result`",
        "- `TAPVid-3D result obtained`",
        "",
        "## Safe Replacement Terms",
        "- `adapter-based TAP-style probe`",
        "- `proxy-style external tracking probe`",
        "- `official evaluator invoked on a non-benchmark-faithful adapter payload`",
        "",
        "## Current Guardrail Summary",
        f"- tap_style_eval_status: {completion_payload.get('tap_style_eval_status', '')}",
        f"- official_evaluator_invoked: {bool(completion_payload.get('official_evaluator_invoked', False))}",
        f"- official_task_faithfully_instantiated: {bool(completion_payload.get('official_task_faithfully_instantiated', False))}",
        f"- paper_official_benchmark: {bool(completion_payload.get('paper_official_benchmark', False))}",
        f"- safest_one_sentence_description_for_paper: {guardrails_payload.get('safest_one_sentence_description_for_paper', '')}",
    ]


def main() -> None:
    args = parse_args()
    completion = _read_json(args.completion_json)
    fidelity = _read_json(args.fidelity_json)
    readiness = _read_json(args.readiness_json)

    claims = _derive_tap_style_claims(completion)
    next_step_choice = "start_paper_framing_prep"

    completion_payload = _build_completion_payload(completion, claims, next_step_choice)
    fidelity_payload = _build_fidelity_payload(fidelity, claims, completion_payload)
    guardrails_payload = _build_guardrails_payload(claims)
    readiness_payload = _build_readiness_payload(readiness, completion_payload, claims, next_step_choice)

    _write_json(args.completion_json, completion_payload)
    _write_md(args.completion_md, _completion_md_lines(completion_payload))
    _write_json(args.fidelity_json, fidelity_payload)
    _write_md(args.fidelity_md, _fidelity_md_lines(fidelity_payload))
    _write_json(args.guardrails_json, guardrails_payload)
    _write_md(args.guardrails_md, _guardrails_md_lines(guardrails_payload))
    _write_json(args.readiness_json, readiness_payload)
    _write_md(args.readiness_md, _readiness_md_lines(readiness_payload))
    _write_md(args.claim_boundary_md, _claim_boundary_md_lines(completion_payload, guardrails_payload))

    print(
        json.dumps(
            {
                "completion_json": str(args.completion_json),
                "fidelity_json": str(args.fidelity_json),
                "guardrails_json": str(args.guardrails_json),
                "readiness_json": str(args.readiness_json),
                "claim_boundary_md": str(args.claim_boundary_md),
                "tap_style_eval_status": completion_payload.get("tap_style_eval_status", ""),
                "tap3d_style_eval_status": completion_payload.get("tap3d_style_eval_status", ""),
                "paper_official_benchmark": completion_payload.get("paper_official_benchmark", False),
                "project_readiness": readiness_payload.get("project_readiness", ""),
                "next_step_choice": readiness_payload.get("next_step_choice", ""),
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
