#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_claim_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF Final Claim Boundary V13", ""]
    for title, key in [
        ("Allowed Strong Claims", "allowed_strong_claims"),
        ("Allowed Moderate Claims", "allowed_moderate_claims"),
        ("Forbidden Claims", "forbidden_claims"),
    ]:
        lines.append(f"## {title}")
        for item in payload.get(key, []):
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## Key Flags")
    for key in [
        "dense_trace_field_claim_allowed",
        "long_horizon_claim_allowed",
        "model_size_scaling_claim_allowed",
        "raw_video_end_to_end_training_claim_allowed",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_ready_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF V13 CVPR Readiness Gate", ""]
    for key in [
        "next_step_choice",
        "corrected_prototype_scaling_positive",
        "corrected_horizon_scaling_positive",
        "corrected_trace_density_scaling_positive",
        "corrected_model_size_scaling_positive",
        "dense_trace_field_claim_allowed",
        "long_horizon_claim_allowed",
        "model_size_scaling_claim_allowed",
        "raw_visualization_pack_ready",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    lines.append("## Remaining Risks")
    for item in payload.get("remaining_risks", []):
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    verification = _load(Path("reports/stwm_fstf_scaling_claim_verification_v13_20260502.json"))
    horizon_audit = _load(Path("reports/stwm_fstf_trace_conditioning_horizon_v13_20260502.json"))
    density_audit = _load(Path("reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json"))
    visual_audit = _load(Path("reports/stwm_fstf_visualization_artifact_audit_v13_20260502.json"))
    dense_allowed = bool(density_audit.get("dense_trace_field_claim_allowed") and verification.get("dense_trace_field_claim_allowed"))
    long_horizon_allowed = bool(
        verification.get("corrected_horizon_scaling_positive")
        and horizon_audit.get("long_horizon_trace_condition_claim_allowed")
    )
    model_size_allowed = bool(verification.get("model_size_scaling_claim_allowed"))
    raw_pack_ready = bool(visual_audit.get("paper_ready_visualization_pack_ready"))

    claim = {
        "audit_name": "stwm_fstf_final_claim_boundary_v13",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_reports": {
            "scaling_claim_verification": "reports/stwm_fstf_scaling_claim_verification_v13_20260502.json",
            "horizon_trace_conditioning": "reports/stwm_fstf_trace_conditioning_horizon_v13_20260502.json",
            "trace_density_valid_units": "reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json",
            "visualization_artifact_audit": "reports/stwm_fstf_visualization_artifact_audit_v13_20260502.json",
        },
        "allowed_strong_claims": [
            "STWM predicts future semantic trace-unit fields over frozen video-derived trace/semantic states.",
            "STWM improves changed semantic prototype prediction over copy and strong copy-aware baselines while preserving stable semantic memory.",
            "Future rollout hidden is load-bearing at H8 and remains load-bearing at H16/H24 under V13 hidden-shuffle/random intervention audits.",
            "C32 is selected as the best prototype vocabulary tradeoff; C128 fails the stability/granularity tradeoff.",
        ],
        "allowed_moderate_claims": [
            "H16/H24 retain positive changed-subset gains under the frozen-cache FSTF protocol.",
            "K16/K32 are evaluated as trace-unit density stress tests, but current valid-unit coverage only supports semantic trace-unit field wording.",
            "Raw-frame rollout visualizations are available as system demonstrations, while training/evaluation uses frozen video-derived trace/semantic caches.",
        ],
        "forbidden_claims": [
            "Raw-video end-to-end training.",
            "Full RGB video generation world model.",
            "Dense semantic trace field, because K16/K32 valid-unit coverage is weak/inconclusive.",
            "Model-size scaling is positive, because base/large do not beat small under strict grouped rules.",
            "Future trace coordinate or temporal order is load-bearing.",
            "Universal OOD dominance or universal cross-dataset generalization.",
            "STWM beats SAM2/CoTracker overall external SOTA or treats SAM2/CoTracker as same-output FSTF baselines.",
        ],
        "dense_trace_field_claim_allowed": dense_allowed,
        "long_horizon_claim_allowed": long_horizon_allowed,
        "model_size_scaling_claim_allowed": model_size_allowed,
        "raw_video_end_to_end_training_claim_allowed": False,
        "required_wording_for_paper": "video-derived trace/semantic state cache for training and evaluation; video-to-FSTF system visualization for qualitative demos.",
    }

    remaining_risks = []
    if not dense_allowed:
        remaining_risks.append("K16/K32 density experiments have weak valid-unit coverage, so dense field wording remains forbidden.")
    if not model_size_allowed:
        remaining_risks.append("Model-size scaling is not positive under strict grouped comparison; do not present a scaling-law claim for capacity.")
    if not raw_pack_ready:
        remaining_risks.append("Visualization artifact pack is missing or incomplete.")
    next_step = "revise_claim_boundary_and_start_overleaf"
    if not raw_pack_ready:
        next_step = "package_missing_visual_assets"
    if not horizon_audit:
        next_step = "run_horizon_trace_conditioning_audit"
    if verification.get("corrected_trace_density_scaling_positive") not in [True, "weak_or_inconclusive"]:
        next_step = "fix_trace_density_materialization"

    readiness = {
        "audit_name": "stwm_fstf_v13_cvpr_readiness_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scaling_claim_verification_path": "reports/stwm_fstf_scaling_claim_verification_v13_20260502.json",
        "horizon_trace_conditioning_audit_path": "reports/stwm_fstf_trace_conditioning_horizon_v13_20260502.json" if horizon_audit else "",
        "trace_density_valid_units_audit_path": "reports/stwm_fstf_trace_density_valid_units_audit_v13_20260502.json",
        "visualization_artifact_audit_path": "reports/stwm_fstf_visualization_artifact_audit_v13_20260502.json",
        "artifact_pack_path": visual_audit.get("artifact_pack_path", ""),
        "corrected_prototype_scaling_positive": bool(verification.get("corrected_prototype_scaling_positive")),
        "corrected_horizon_scaling_positive": bool(verification.get("corrected_horizon_scaling_positive")),
        "corrected_trace_density_scaling_positive": density_audit.get("trace_density_scaling_positive", verification.get("corrected_trace_density_scaling_positive")),
        "corrected_model_size_scaling_positive": bool(verification.get("corrected_model_size_scaling_positive")),
        "dense_trace_field_claim_allowed": dense_allowed,
        "long_horizon_claim_allowed": long_horizon_allowed,
        "model_size_scaling_claim_allowed": model_size_allowed,
        "raw_visualization_pack_ready": raw_pack_ready,
        "remaining_risks": remaining_risks,
        "next_step_choice": next_step,
    }

    _dump(Path("reports/stwm_fstf_final_claim_boundary_v13_20260502.json"), claim)
    _dump(Path("reports/stwm_fstf_v13_cvpr_readiness_gate_20260502.json"), readiness)
    _write_claim_doc(Path("docs/STWM_FSTF_FINAL_CLAIM_BOUNDARY_V13_20260502.md"), claim)
    _write_ready_doc(Path("docs/STWM_FSTF_V13_CVPR_READINESS_GATE_20260502.md"), readiness)
    print("reports/stwm_fstf_v13_cvpr_readiness_gate_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
