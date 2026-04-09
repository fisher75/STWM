#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List
import json


TAP_STYLE_ALLOWED = {
    "fully_implemented_and_run",
    "partially_bridged",
    "proxy_only",
    "not_yet_implemented",
}
TAP3D_ALLOWED = {
    "fully_implemented_and_run",
    "partially_bridged",
    "not_yet_implemented",
}
READINESS_ALLOWED = {
    "paper_eval_ready",
    "training_ready_but_eval_gap_remains",
    "eval_not_ready",
}


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


def parse_args() -> Any:
    p = ArgumentParser(description="Check consistency of Stage2 external-eval claim-boundary outputs")
    p.add_argument(
        "--completion-json",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_completion_20260408.json",
    )
    p.add_argument(
        "--fidelity-json",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_fidelity_audit_20260409.json",
    )
    p.add_argument(
        "--readiness-json",
        default="/home/chen034/workspace/stwm/reports/tracewm_project_readiness_20260409.json",
    )
    p.add_argument(
        "--guardrails-json",
        default="/home/chen034/workspace/stwm/reports/stage2_external_eval_result_interpretation_guardrails_20260409.json",
    )
    p.add_argument(
        "--output-json",
        default="",
    )
    return p.parse_args()


def _bool(x: Any) -> bool:
    return bool(x)


def _get_tap_style(completion: Dict[str, Any]) -> Dict[str, Any]:
    primary = completion.get("primary_checkpoint_eval", {}) if isinstance(completion.get("primary_checkpoint_eval", {}), dict) else {}
    return primary.get("tap_style_eval", {}) if isinstance(primary.get("tap_style_eval", {}), dict) else {}


def build_consistency_report(
    completion: Dict[str, Any],
    fidelity: Dict[str, Any],
    readiness: Dict[str, Any],
    guardrails: Dict[str, Any],
) -> Dict[str, Any]:
    tap = _get_tap_style(completion)
    status = str(completion.get("tap_style_eval_status", tap.get("status", "not_yet_implemented")))
    tap3d_status = str(completion.get("tap3d_style_eval_status", "not_yet_implemented"))
    official_invoked = _bool(completion.get("official_evaluator_invoked", tap.get("official_evaluator_invoked", False)))
    official_connected = _bool(completion.get("official_tapvid_evaluator_connected", tap.get("official_tapvid_evaluator_connected", False)))
    faithful = _bool(completion.get("official_task_faithfully_instantiated", tap.get("official_task_faithfully_instantiated", False)))
    paper_official_benchmark = _bool(readiness.get("paper_official_benchmark", completion.get("paper_official_benchmark", False)))
    dataset_family_match = _bool(guardrails.get("dataset_family_match", False))
    query_protocol_match = _bool(guardrails.get("query_protocol_match", False))
    visibility_protocol_match = _bool(guardrails.get("visibility_protocol_match", False))
    official_benchmark_equivalent = _bool(guardrails.get("official_benchmark_equivalent", False))

    errors: List[str] = []
    warnings: List[str] = []

    if status not in TAP_STYLE_ALLOWED:
        errors.append(f"invalid tap_style_eval_status={status}")
    if tap3d_status not in TAP3D_ALLOWED:
        errors.append(f"invalid tap3d_style_eval_status={tap3d_status}")
    readiness_status = str(readiness.get("project_readiness", ""))
    if readiness_status not in READINESS_ALLOWED:
        errors.append(f"invalid project_readiness={readiness_status}")

    if status == "fully_implemented_and_run" and not faithful:
        errors.append("tap_style_eval_status cannot be fully_implemented_and_run when official_task_faithfully_instantiated=false")
    if status == "fully_implemented_and_run" and not official_invoked:
        errors.append("tap_style_eval_status cannot be fully_implemented_and_run when official_evaluator_invoked=false")
    if status == "partially_bridged" and not (official_invoked and official_connected):
        errors.append("tap_style_eval_status=partially_bridged requires official_evaluator_invoked=true and official_tapvid_evaluator_connected=true")
    if status == "proxy_only" and official_invoked:
        errors.append("tap_style_eval_status=proxy_only is inconsistent with official_evaluator_invoked=true")
    if paper_official_benchmark and not faithful:
        errors.append("paper_official_benchmark cannot be true when official_task_faithfully_instantiated=false")
    if paper_official_benchmark and not official_benchmark_equivalent:
        errors.append("paper_official_benchmark cannot be true when official_benchmark_equivalent=false")
    if paper_official_benchmark and not (dataset_family_match and query_protocol_match and visibility_protocol_match):
        errors.append("paper_official_benchmark requires dataset/query/visibility protocol match")
    if readiness_status == "paper_eval_ready" and not paper_official_benchmark:
        errors.append("project_readiness=paper_eval_ready requires paper_official_benchmark=true")
    if tap3d_status == "fully_implemented_and_run":
        tap3d_checks = fidelity.get("tap3d_task_checks", {}) if isinstance(fidelity.get("tap3d_task_checks", {}), dict) else {}
        if not all(
            _bool(tap3d_checks.get(k, False))
            for k in [
                "aligned_3d_gt_for_current_binding",
                "camera_geometry_projection_or_lifting_path_available",
                "verified_exporter_to_tracks_xyz_visibility",
            ]
        ):
            errors.append("tap3d_style_eval_status=fully_implemented_and_run requires all strict TAP3D checks")

    if not faithful and status == "partially_bridged":
        warnings.append("partially_bridged is being used in the adapter-only sense; keep official benchmark language disabled")
    if official_invoked and not official_benchmark_equivalent:
        warnings.append("official evaluator invocation does not imply official benchmark equivalence")

    return {
        "ok": len(errors) == 0,
        "tap_style_eval_status": status,
        "tap3d_style_eval_status": tap3d_status,
        "official_evaluator_invoked": official_invoked,
        "official_task_faithfully_instantiated": faithful,
        "paper_official_benchmark": paper_official_benchmark,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    completion = _read_json(args.completion_json)
    fidelity = _read_json(args.fidelity_json)
    readiness = _read_json(args.readiness_json)
    guardrails = _read_json(args.guardrails_json)

    report = build_consistency_report(completion, fidelity, readiness, guardrails)
    if args.output_json:
        _write_json(args.output_json, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if not bool(report.get("ok", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
