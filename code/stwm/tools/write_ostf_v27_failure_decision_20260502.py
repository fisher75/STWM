#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc


TARGET_PATH = ROOT / "reports/stwm_traceanything_target_semantics_audit_v27_20260502.json"
BASELINE_PATH = ROOT / "reports/stwm_ostf_prior_baseline_hierarchy_v27_20260502.json"
SUBSETS_PATH = ROOT / "reports/stwm_ostf_hard_subsets_v27_20260502.json"
V26_EVAL_PATH = ROOT / "reports/stwm_ostf_v26_eval_summary_20260502.json"
V26_DECISION_PATH = ROOT / "reports/stwm_ostf_v26_decision_20260502.json"
V26_BOOT_PATH = ROOT / "reports/stwm_ostf_v26_bootstrap_20260502.json"
FAILURE_PATH = ROOT / "reports/stwm_ostf_v26_failure_attribution_v27_20260502.json"
FAILURE_DOC = ROOT / "docs/STWM_OSTF_V26_FAILURE_ATTRIBUTION_V27_20260502.md"
DECISION_PATH = ROOT / "reports/stwm_ostf_v27_prior_decision_20260502.json"
DECISION_DOC = ROOT / "docs/STWM_OSTF_V27_PRIOR_DECISION_20260502.md"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    target = _load(TARGET_PATH)
    baseline = _load(BASELINE_PATH)
    subsets = _load(SUBSETS_PATH)
    v26_eval = _load(V26_EVAL_PATH)
    v26_decision = _load(V26_DECISION_PATH)
    v26_boot = _load(V26_BOOT_PATH)

    m128 = v26_eval.get("experiments", {}).get("v26_traceanything_m128_h32_seed42", {}).get("test_metrics", {})
    cv = v26_eval.get("baselines", {}).get("constant_velocity_copy_M128_H32", {}).get("test_metrics", {})
    last = v26_eval.get("baselines", {}).get("last_observed_copy_M128_H32", {}).get("test_metrics", {})
    v26_vs_last_gap = None
    if m128.get("minFDE_K_px") is not None and last.get("minFDE_K_px") is not None:
        v26_vs_last_gap = float(m128["minFDE_K_px"]) - float(last["minFDE_K_px"])

    top1_cv_like = bool((m128.get("top1_mode_non_cv_rate") or 0.0) == 0.0)
    topk_has_useful_noncv = bool((m128.get("best_mode_non_cv_rate") or 0.0) > 0.0)
    topk_has_last_like = bool(
        m128.get("minFDE_K_px") is not None
        and last.get("minFDE_K_px") is not None
        and float(m128["minFDE_K_px"]) <= float(last["minFDE_K_px"]) * 1.15
    )
    failure = {
        "audit_name": "stwm_ostf_v26_failure_attribution_v27",
        "generated_at_utc": datetime.now().astimezone().isoformat(),
        "V26_beats_CV_but_loses_to_last_observed": bool(
            m128.get("minFDE_K_px") is not None
            and cv.get("minFDE_K_px") is not None
            and last.get("minFDE_K_px") is not None
            and float(m128["minFDE_K_px"]) < float(cv["minFDE_K_px"])
            and float(m128["minFDE_K_px"]) > float(last["minFDE_K_px"])
        ),
        "V26_M128_H32_minFDE_K_px": m128.get("minFDE_K_px"),
        "CV_M128_H32_minFDE_K_px": cv.get("minFDE_K_px"),
        "last_observed_M128_H32_minFDE_K_px": last.get("minFDE_K_px"),
        "V26_minus_last_observed_minFDE_K_px": v26_vs_last_gap,
        "whether_V26_learned_CV_based_residual_instead_of_last_observed_prior": top1_cv_like,
        "why": (
            "V26 top-1 mode selection remains CV-like, while oracle minFDE uses non-CV modes on some items. "
            "The model did not include a last-observed/damped-prior mode strong enough to match the actual target hierarchy."
        ),
        "dense_points_non_load_bearing_reason": (
            "The strongest target prior is object-level persistence/damped motion; dense point input does not change the selected prior under the current objective."
        ),
        "semantic_memory_non_load_bearing_reason": (
            "Current TraceAnything trajectory targets are mostly geometric and semantic prototype labels are static/observed-memory dominated."
        ),
        "topK_modes_include_last_observed_like_mode": topk_has_last_like,
        "topK_modes_include_some_non_CV_alternatives": topk_has_useful_noncv,
        "mode_logits_select_CV_or_learned_modes_appropriately": False,
        "mode_logits_issue": "Top-1 mode non-CV rate is zero for M128_H32; usable mode selection remains unresolved.",
        "new_prior_should_be_last_observed_or_damped_velocity": True,
        "recommended_model_prior": baseline.get("recommended_model_prior", "last_observed"),
        "supporting_bootstrap": {
            "m128_vs_cv_all_minfde": v26_boot.get("m128_vs_cv_all_minfde"),
            "m128_vs_cv_hard_minfde": v26_boot.get("m128_vs_cv_hard_minfde"),
        },
    }
    dump_json(FAILURE_PATH, failure)
    write_doc(
        FAILURE_DOC,
        "STWM OSTF V26 Failure Attribution V27",
        failure,
        [
            "V26_beats_CV_but_loses_to_last_observed",
            "V26_M128_H32_minFDE_K_px",
            "CV_M128_H32_minFDE_K_px",
            "last_observed_M128_H32_minFDE_K_px",
            "whether_V26_learned_CV_based_residual_instead_of_last_observed_prior",
            "topK_modes_include_last_observed_like_mode",
            "mode_logits_select_CV_or_learned_modes_appropriately",
            "recommended_model_prior",
        ],
    )

    extraction_bug = bool(target.get("target_extraction_bug_detected", True))
    target_valid = bool(target.get("target_semantics_valid", False))
    revised_ready = bool(subsets.get("revised_hardbench_ready", False))
    recommended = baseline.get("recommended_model_prior", "last_observed")
    if extraction_bug:
        proceed = "fix_traceanything_extraction"
    elif not revised_ready:
        proceed = "rebuild_hardbench"
    elif target_valid:
        proceed = "train_v28_with_correct_prior"
    else:
        proceed = "add_PointOdyssey_TAPVid_GT"
    decision = {
        "audit_name": "stwm_ostf_v27_prior_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_semantics_valid": target_valid,
        "target_extraction_bug_detected": extraction_bug,
        "last_observed_strength_explained": target.get("last_observed_strength_expected_or_bug") == "expected_under_current_teacher_target_semantics_not_a_direct_extraction_bug",
        "strongest_nonlearned_prior": baseline.get("strongest_nonlearned_prior"),
        "whether_CV_was_weak_baseline": baseline.get("whether_CV_is_weak_baseline"),
        "affine_bug_fixed_or_blocked": baseline.get("affine_bug_fixed_or_blocked"),
        "revised_hardbench_ready": revised_ready,
        "recommended_model_prior": recommended,
        "proceed_to": proceed,
        "next_step_choice": proceed,
    }
    dump_json(DECISION_PATH, decision)
    write_doc(
        DECISION_DOC,
        "STWM OSTF V27 Prior Decision",
        decision,
        [
            "target_semantics_valid",
            "target_extraction_bug_detected",
            "last_observed_strength_explained",
            "strongest_nonlearned_prior",
            "whether_CV_was_weak_baseline",
            "affine_bug_fixed_or_blocked",
            "revised_hardbench_ready",
            "recommended_model_prior",
            "next_step_choice",
        ],
    )
    print(DECISION_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
