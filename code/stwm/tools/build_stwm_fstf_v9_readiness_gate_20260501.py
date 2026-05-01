#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    v8 = _load("reports/stwm_fstf_strong_copyaware_baseline_suite_v8_20260501.json")
    v8_audit = _load("reports/stwm_fstf_v8_live_artifact_audit_20260501.json")
    mech = _load("reports/stwm_fstf_mechanism_ablation_v9_20260501.json")
    pipe = _load("reports/stwm_fstf_video_input_pipeline_audit_v9_20260501.json")
    scaling = _load("reports/stwm_fstf_scaling_v9_20260501.json")
    judgments = mech.get("judgments", {})
    mechanism_completed = bool(mech.get("results"))
    pipeline_passed = bool(
        pipe.get("video_input_claim_allowed")
        and pipe.get("frozen_frontend_pipeline")
        and pipe.get("cache_training_disclosed")
        and not pipe.get("future_leakage_detected", True)
    )
    scaling_completed = bool(scaling.get("scaling_completed", False))
    full_beats_ablations = judgments.get("full_STWM_beats_all_internal_ablations")
    if not mechanism_completed:
        next_step = "run_missing_mechanism_ablation"
    elif full_beats_ablations is False:
        next_step = "revise_claim_boundary"
    elif not scaling_completed:
        next_step = "run_missing_scaling"
    elif not pipeline_passed:
        next_step = "revise_claim_boundary"
    else:
        next_step = "build_paper_figures_and_videos"
    report = {
        "audit_name": "stwm_fstf_v9_cvpr_readiness_gate",
        "v8_reproducibility_fixed": bool(v8.get("baseline_suite_completed") and v8_audit.get("live_artifacts_exist")),
        "mechanism_ablation_completed": mechanism_completed,
        "full_STWM_beats_all_internal_ablations": full_beats_ablations,
        "video_input_pipeline_audit_passed": pipeline_passed,
        "scaling_completed": scaling_completed,
        "dense_trace_field_claim_allowed": bool(scaling.get("whether_dense_trace_field_claim_allowed", False)),
        "long_horizon_claim_allowed": bool(scaling.get("whether_long_horizon_world_model_claim_allowed", False)),
        "current_allowed_claims": [
            "STWM-FSTF uses frozen video-derived trace/semantic states and observed semantic memory to predict future semantic trace-unit fields under free rollout.",
            "V8 supports that STWM outperforms strong controlled copy-aware same-output baselines on mixed test.",
            "Visibility/reappearance are auxiliary states but remain metric_invalid_or_untrained for positive claims.",
        ],
        "current_forbidden_claims": [
            "Do not claim dense trace field until K scaling succeeds.",
            "Do not claim long-horizon world model until H16/H24 scaling succeeds.",
            "Do not claim end-to-end raw-video training; disclose frozen materialized cache training.",
            "Do not use candidate scorer, SAM2/CoTracker plugin, future candidate leakage, or test-set selection.",
        ],
        "next_step_choice": next_step,
        "source_reports": {
            "v8_suite": "reports/stwm_fstf_strong_copyaware_baseline_suite_v8_20260501.json",
            "v8_artifact_audit": "reports/stwm_fstf_v8_live_artifact_audit_20260501.json",
            "mechanism_ablation": "reports/stwm_fstf_mechanism_ablation_v9_20260501.json",
            "pipeline_audit": "reports/stwm_fstf_video_input_pipeline_audit_v9_20260501.json",
            "scaling": "reports/stwm_fstf_scaling_v9_20260501.json",
        },
    }
    out = Path("reports/stwm_fstf_v9_cvpr_readiness_gate_20260501.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    doc = Path("docs/STWM_FSTF_V9_CVPR_READINESS_GATE_20260501.md")
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(
        "\n".join(
            [
                "# STWM FSTF V9 CVPR Readiness Gate",
                "",
                f"- v8_reproducibility_fixed: `{report['v8_reproducibility_fixed']}`",
                f"- mechanism_ablation_completed: `{mechanism_completed}`",
                f"- full_STWM_beats_all_internal_ablations: `{report['full_STWM_beats_all_internal_ablations']}`",
                f"- video_input_pipeline_audit_passed: `{pipeline_passed}`",
                f"- scaling_completed: `{scaling_completed}`",
                f"- dense_trace_field_claim_allowed: `{report['dense_trace_field_claim_allowed']}`",
                f"- long_horizon_claim_allowed: `{report['long_horizon_claim_allowed']}`",
                f"- next_step_choice: `{next_step}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[readiness-v9] report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
