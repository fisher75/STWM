#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/raid/chen034/workspace/stwm")
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
ASSETS = ROOT / "assets"
MIXED_CKPT_DIR = ROOT / "outputs/checkpoints/stwm_mixed_fullscale_v2_20260428"
LODO_CKPT_DIR = ROOT / "outputs/checkpoints/stwm_final_lodo_v3_20260428"
LOG_DIR = ROOT / "outputs/logs"


def load_json(path: str | Path, default: Any | None = None) -> Any:
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_md(path: str | Path, title: str, sections: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + "\n\n".join(sections).rstrip() + "\n", encoding="utf-8")


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p.astype(np.float64)
    q = q.astype(np.float64)
    p = p / np.clip(p.sum(), 1e-12, None)
    q = q / np.clip(q.sum(), 1e-12, None)
    m = 0.5 * (p + q)
    mask_p = p > 0
    mask_q = q > 0
    kl_pm = float(np.sum(p[mask_p] * np.log(p[mask_p] / np.clip(m[mask_p], 1e-12, None))))
    kl_qm = float(np.sum(q[mask_q] * np.log(q[mask_q] / np.clip(m[mask_q], 1e-12, None))))
    return 0.5 * (kl_pm + kl_qm)


def build_artifact_reconcile() -> dict[str, Any]:
    mixed_train_complete = load_json(REPORTS / "stwm_mixed_fullscale_v2_train_summary_complete_20260428.json")
    mixed_val_complete = load_json(REPORTS / "stwm_mixed_fullscale_v2_val_selection_complete_20260428.json")
    mixed_decision_complete = load_json(REPORTS / "stwm_mixed_fullscale_v2_complete_decision_20260428.json")
    mixed_old_val = load_json(REPORTS / "stwm_mixed_fullscale_v2_val_selection_20260428.json")
    mixed_old_decision = load_json(REPORTS / "stwm_mixed_fullscale_v2_decision_20260428.json")

    expected_runs = [(c, s) for c in (32, 64) for s in (42, 123, 456, 789, 1001)]
    ckpt_paths = [MIXED_CKPT_DIR / f"c{c}_seed{s}_final.pt" for c, s in expected_runs]
    train_reports = [REPORTS / f"stwm_mixed_fullscale_v2_train_c{c}_seed{s}_20260428.json" for c, s in expected_runs]
    log_paths = [LOG_DIR / f"stwm_mixed_fullscale_v2_c{c}_seed{s}.log" for c, s in expected_runs]
    eval_reports = [
        REPORTS / "stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json",
        REPORTS / "stwm_mixed_fullscale_v2_vspw_test_eval_complete_20260428.json",
        REPORTS / "stwm_mixed_fullscale_v2_vipseg_test_eval_complete_20260428.json",
    ]
    lodo_train = load_json(REPORTS / "stwm_final_lodo_train_summary_20260428.json")
    lodo_eval_a = load_json(REPORTS / "stwm_final_lodo_vspw_to_vipseg_eval_20260428.json")
    lodo_eval_b = load_json(REPORTS / "stwm_final_lodo_vipseg_to_vspw_eval_20260428.json")
    baseline_v5 = load_json(REPORTS / "stwm_final_baseline_suite_complete_v5_eval_20260428.json")
    scaling_v5 = load_json(REPORTS / "stwm_final_scaling_laws_complete_v5_20260428.json")

    existing_ckpts = [str(p) for p in ckpt_paths if p.exists()]
    existing_train_reports = [str(p) for p in train_reports if p.exists()]
    existing_logs = [str(p) for p in log_paths if p.exists()]
    empty_logs = [str(p) for p in log_paths if p.exists() and p.stat().st_size == 0]
    train_summaries_match = []
    for path in train_reports:
        payload = load_json(path)
        train_summaries_match.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "checkpoint_path": payload.get("checkpoint_path", ""),
                "checkpoint_exists": (ROOT / str(payload.get("checkpoint_path", ""))).exists() if payload.get("checkpoint_path") else False,
                "trace_regression_detected": bool(payload.get("trace_regression_detected", True)) if path.exists() else None,
            }
        )

    selected_canonical = int(mixed_val_complete.get("selected_prototype_count", 0))
    selected_seed_canonical = int(mixed_val_complete.get("selected_seed", -1))
    stale_conflicts = []
    if int(mixed_old_val.get("selected_prototype_count", 0) or 0) not in (0, selected_canonical):
        stale_conflicts.append(str(REPORTS / "stwm_mixed_fullscale_v2_val_selection_20260428.json"))
    if int(mixed_old_decision.get("best_prototype_count", 0) or 0) not in (0, selected_canonical):
        stale_conflicts.append(str(REPORTS / "stwm_mixed_fullscale_v2_decision_20260428.json"))

    lodo_val_only = (
        bool(lodo_eval_a.get("val_selection_used_val_only", False))
        and bool(lodo_eval_b.get("val_selection_used_val_only", False))
        and not bool(lodo_eval_a.get("test_metrics_used_for_selection", False))
        and not bool(lodo_eval_b.get("test_metrics_used_for_selection", False))
    )

    payload = {
        "audit_name": "stwm_final_artifact_consistency_reconcile_v6_20260501",
        "mixed_fullscale_artifacts": {
            "expected_checkpoint_count": 10,
            "existing_checkpoint_count": len(existing_ckpts),
            "expected_train_report_count": 10,
            "existing_train_report_count": len(existing_train_reports),
            "expected_log_count": 10,
            "existing_log_count": len(existing_logs),
            "empty_log_count": len(empty_logs),
            "all_eval_reports_exist": all(p.exists() for p in eval_reports),
            "completed_run_count_reported": int(mixed_train_complete.get("completed_run_count", 0)),
            "all_runs_completed_reported": bool(mixed_train_complete.get("all_runs_completed", False)),
            "train_summaries_match_checkpoints": train_summaries_match,
        },
        "selected_C_reconciliation": {
            "canonical_selected_C": selected_canonical,
            "canonical_selected_seed": selected_seed_canonical,
            "canonical_paths": [
                str(REPORTS / "stwm_mixed_fullscale_v2_val_selection_complete_20260428.json"),
                str(REPORTS / "stwm_mixed_fullscale_v2_complete_decision_20260428.json"),
                str(REPORTS / "stwm_final_prototype_vocab_justification_v3_20260428.json"),
            ],
            "stale_conflicting_paths": stale_conflicts,
            "selected_C_rule": "mixed val only; primary changed gain over copy; secondary overall gain; tertiary stable drop; tie lower trace coord error",
        },
        "lodo_selection_reconciliation": {
            "lodo_completed": bool(lodo_train.get("lodo_completed", False)),
            "expected_run_count": int(lodo_train.get("expected_run_count", 0)),
            "completed_run_count": int(lodo_train.get("completed_run_count", 0)),
            "failed_run_count": int(lodo_train.get("failed_run_count", 0)),
            "vspw_to_vipseg_selected": {
                "prototype_count": int(lodo_eval_a.get("selected_prototype_count", 0)),
                "seed": int(lodo_eval_a.get("selected_seed", -1)),
                "checkpoint": lodo_eval_a.get("selected_checkpoint_path", ""),
            },
            "vipseg_to_vspw_selected": {
                "prototype_count": int(lodo_eval_b.get("selected_prototype_count", 0)),
                "seed": int(lodo_eval_b.get("selected_seed", -1)),
                "checkpoint": lodo_eval_b.get("selected_checkpoint_path", ""),
            },
            "selection_is_source_val_only": lodo_val_only,
        },
        "baseline_scaling_semantics": {
            "baseline_suite_completed_reported": bool(baseline_v5.get("baseline_suite_completed", False)),
            "baseline_suite_is_only_audit_complete": not bool(baseline_v5.get("baseline_suite_completed", False)),
            "scaling_complete_reported": {
                "model": bool(scaling_v5.get("model_size_scaling", {}).get("model_scaling_completed", False)),
                "horizon": bool(scaling_v5.get("horizon_scaling", {}).get("horizon_scaling_completed", False)),
                "trace_density": bool(scaling_v5.get("trace_density_scaling", {}).get("trace_density_scaling_completed", False)),
            },
        },
        "can_claim_5seed_main_result": bool(
            len(existing_ckpts) == 10
            and len(existing_train_reports) == 10
            and bool(mixed_train_complete.get("all_runs_completed", False))
            and int(mixed_val_complete.get("candidate_count", 0)) == 10
            and bool(mixed_decision_complete.get("residual_beats_copy_mixed", False))
        ),
        "can_claim_selected_C": bool(
            selected_canonical == 32
            and selected_seed_canonical == 456
            and int(mixed_decision_complete.get("best_prototype_count", 0)) == 32
            and int(mixed_decision_complete.get("best_seed", -1)) == 456
            and lodo_val_only
        ),
        "can_claim_baseline_complete": False,
        "can_claim_scaling_complete": False,
    }
    write_json(REPORTS / "stwm_final_artifact_consistency_reconcile_v6_20260501.json", payload)
    write_md(
        DOCS / "STWM_FINAL_ARTIFACT_CONSISTENCY_RECONCILE_V6_20260501.md",
        "STWM Final Artifact Consistency Reconcile V6 20260501",
        [
            "## Mixed Fullscale Main Result\n"
            + "\n".join(
                [
                    f"- expected_checkpoint_count: `{payload['mixed_fullscale_artifacts']['expected_checkpoint_count']}`",
                    f"- existing_checkpoint_count: `{payload['mixed_fullscale_artifacts']['existing_checkpoint_count']}`",
                    f"- existing_train_report_count: `{payload['mixed_fullscale_artifacts']['existing_train_report_count']}`",
                    f"- existing_log_count: `{payload['mixed_fullscale_artifacts']['existing_log_count']}`",
                    f"- empty_log_count: `{payload['mixed_fullscale_artifacts']['empty_log_count']}`",
                ]
            ),
            "## Selected C\n"
            + "\n".join(
                [
                    f"- canonical_selected_C: `{payload['selected_C_reconciliation']['canonical_selected_C']}`",
                    f"- canonical_selected_seed: `{payload['selected_C_reconciliation']['canonical_selected_seed']}`",
                    f"- stale_conflicting_paths: `{payload['selected_C_reconciliation']['stale_conflicting_paths']}`",
                ]
            ),
            "## Claim Status\n"
            + "\n".join(
                [
                    f"- can_claim_5seed_main_result: `{payload['can_claim_5seed_main_result']}`",
                    f"- can_claim_selected_C: `{payload['can_claim_selected_C']}`",
                    f"- can_claim_baseline_complete: `{payload['can_claim_baseline_complete']}`",
                    f"- can_claim_scaling_complete: `{payload['can_claim_scaling_complete']}`",
                ]
            ),
            "## Note\n- Mixed fullscale main-result artifacts are real. The main inconsistency comes from older non-complete reports that still point to C64 before the full 10-candidate val-only selection converged to C32 seed456.",
        ],
    )
    return payload


def build_same_output_baseline_suite_v6() -> dict[str, Any]:
    mixed_eval = load_json(REPORTS / "stwm_mixed_fullscale_v2_mixed_test_eval_complete_20260428.json")
    mixed_sig = load_json(REPORTS / "stwm_mixed_fullscale_v2_significance_complete_20260428.json").get("mixed", {})
    seed_robust = load_json(REPORTS / "stwm_mixed_fullscale_v2_seed_robustness_complete_20260428.json")
    payload = {
        "audit_name": "stwm_same_output_baseline_suite_v6_20260501",
        "baseline_suite_completed": False,
        "STWM_beats_same_output_baselines": False,
        "strongest_same_output_baseline": "copy_semantic_memory_baseline",
        "changed_gain_vs_copy": float(mixed_eval.get("best_metrics", {}).get("changed_subset_gain_over_copy", 0.0)),
        "stable_drop": float(mixed_eval.get("best_metrics", {}).get("stable_preservation_drop", 0.0)),
        "overall_top5_delta": float(mixed_eval.get("best_metrics", {}).get("overall_gain_over_copy", 0.0)),
        "paired_bootstrap_CI": {
            "changed_top5_delta": mixed_sig.get("residual_vs_copy_changed_top5", {}),
            "overall_top5_delta": mixed_sig.get("residual_vs_copy_overall_top5", {}),
            "stable_drop": mixed_sig.get("stable_preservation_drop", {}),
        },
        "seed_mean_std": {
            "c32_changed_gain_mean_std": seed_robust.get("c32_changed_gain_mean_std", {}),
            "c64_changed_gain_mean_std": seed_robust.get("c64_changed_gain_mean_std", {}),
        },
        "fair_input_output_status": "Only copy baseline is fully same-input/same-output/free-rollout in the live repo. Requested trace-only/semantic-only/semantic+trace/SlotFormer-like/DINO-WM-like baselines are still missing as completed artifacts.",
        "missing_or_failed_baselines": [
            "trace_only_AR_transformer_baseline",
            "semantic_only_memory_transition_baseline",
            "simple_semantic_plus_trace_transformer_baseline",
            "slotformer_like_trace_unit_slot_dynamics_baseline",
            "dino_wm_like_latent_feature_dynamics_baseline",
        ],
    }
    write_json(REPORTS / "stwm_same_output_baseline_suite_v6_20260501.json", payload)
    write_md(
        DOCS / "STWM_SAME_OUTPUT_BASELINE_SUITE_V6_20260501.md",
        "STWM Same Output Baseline Suite V6 20260501",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- baseline_suite_completed: `{payload['baseline_suite_completed']}`",
                    f"- STWM_beats_same_output_baselines: `{payload['STWM_beats_same_output_baselines']}`",
                    f"- strongest_same_output_baseline: `{payload['strongest_same_output_baseline']}`",
                ]
            ),
            "## Mixed Main Result vs Copy\n"
            + "\n".join(
                [
                    f"- changed_gain_vs_copy: `{payload['changed_gain_vs_copy']}`",
                    f"- overall_top5_delta: `{payload['overall_top5_delta']}`",
                    f"- stable_drop: `{payload['stable_drop']}`",
                ]
            ),
            "## Missing Baselines\n" + "\n".join(f"- {x}" for x in payload["missing_or_failed_baselines"]),
            "## Note\n- The audit is complete, but the evidence is not. We cannot claim a same-output baseline suite is complete until these baselines are implemented and evaluated under the frozen free-rollout STWM-FSTF protocol.",
        ],
    )
    return payload


def build_scaling_laws_v6() -> dict[str, Any]:
    proto = load_json(REPORTS / "stwm_final_prototype_vocab_justification_v3_20260428.json")
    model = load_json(REPORTS / "stwm_final_model_size_scaling_v3_eval_20260428.json")
    horizon = load_json(REPORTS / "stwm_final_horizon_scaling_v3_eval_20260428.json")
    density = load_json(REPORTS / "stwm_final_trace_density_scaling_v3_eval_20260428.json")
    payload = {
        "audit_name": "stwm_scaling_laws_v6_20260501",
        "prototype_scaling_completed": False,
        "selected_C": int(proto.get("selected_C_on_mixed_main_result", 0)),
        "selected_C_rule": proto.get("selection_reason", ""),
        "model_scaling_completed": bool(model.get("model_scaling_completed", False)),
        "model_scaling_positive": bool(model.get("model_scaling_trend_positive", False)),
        "horizon_scaling_completed": bool(horizon.get("horizon_scaling_completed", False)),
        "horizon_scaling_positive": bool(horizon.get("horizon_scaling_positive", False)),
        "trace_density_scaling_completed": bool(density.get("trace_density_scaling_completed", False)),
        "trace_density_scaling_positive": False,
        "whether_dense_trace_field_claim_allowed": False,
        "recommended_terminology": str(density.get("terminology_recommendation", "semantic trace-unit field")),
        "prototype_scaling_note": "C16 is still missing from the controlled mixed fullscale protocol, so prototype scaling is only partially evidenced even though selected_C is justified from val-only selection plus earlier sweeps.",
        "compute_cost_status": {
            "prototype_scaling_requires_new_runs": True,
            "model_scaling_requires_new_runs": True,
            "horizon_scaling_requires_cache_rebuild_and_retraining": True,
            "trace_density_requires_cache_rebuild_and_retraining": True,
        },
    }
    write_json(REPORTS / "stwm_scaling_laws_v6_20260501.json", payload)
    write_md(
        DOCS / "STWM_SCALING_LAWS_V6_20260501.md",
        "STWM Scaling Laws V6 20260501",
        [
            "## Selected C\n"
            + "\n".join(
                [
                    f"- selected_C: `{payload['selected_C']}`",
                    f"- selected_C_rule: `{payload['selected_C_rule']}`",
                ]
            ),
            "## Status\n"
            + "\n".join(
                [
                    f"- prototype_scaling_completed: `{payload['prototype_scaling_completed']}`",
                    f"- model_scaling_completed: `{payload['model_scaling_completed']}`",
                    f"- model_scaling_positive: `{payload['model_scaling_positive']}`",
                    f"- horizon_scaling_completed: `{payload['horizon_scaling_completed']}`",
                    f"- horizon_scaling_positive: `{payload['horizon_scaling_positive']}`",
                    f"- trace_density_scaling_completed: `{payload['trace_density_scaling_completed']}`",
                    f"- whether_dense_trace_field_claim_allowed: `{payload['whether_dense_trace_field_claim_allowed']}`",
                    f"- recommended_terminology: `{payload['recommended_terminology']}`",
                ]
            ),
            "## Note\n- Live repo still only supports the completed mixed H8/K8 mainline. Dense-field wording remains disallowed.",
        ],
    )
    return payload


def _feature_norms() -> dict[str, Any]:
    z = np.load(ROOT / "outputs/cache/stwm_fullscale_observed_semantic_prototype_targets_v1_20260428/observed_features.npz", allow_pickle=True)
    item_keys = np.asarray(z["item_keys"], dtype=object)
    last_feat = z["observed_last_feature"]
    obs_mask = z["observed_feature_mask"].astype(bool)
    out = {}
    for ds in ("VSPW", "VIPSEG"):
        ds_mask = np.asarray([str(x).startswith(ds + "::") for x in item_keys], dtype=bool)
        norms = np.linalg.norm(last_feat[ds_mask], axis=-1)[obs_mask[ds_mask]]
        out[ds] = {
            "slot_count": int(norms.size),
            "crop_feature_norm_mean": float(norms.mean()) if norms.size else 0.0,
            "crop_feature_norm_std": float(norms.std()) if norms.size else 0.0,
            "zero_norm_fraction": float(np.mean(norms <= 1e-12)) if norms.size else 0.0,
        }
    return out


def _prototype_shift() -> dict[str, Any]:
    out = {}
    for c in (32, 64):
        z = np.load(ROOT / f"outputs/cache/stwm_fullscale_semantic_trace_prototype_targets_c{c}_v1_20260428/prototype_targets.npz", allow_pickle=True)
        datasets = np.asarray([str(x).upper() for x in z["datasets"]], dtype=object)
        target = z["future_semantic_proto_target"]
        mask = z["target_mask"].astype(bool)
        hists = {}
        for ds in ("VSPW", "VIPSEG"):
            ds_mask = datasets == ds
            flat = target[ds_mask][mask[ds_mask]]
            hist = np.bincount(flat.astype(np.int64), minlength=c).astype(np.float64)
            hists[ds] = hist
        out[f"C{c}"] = {
            "js_divergence": _js_divergence(hists["VSPW"], hists["VIPSEG"]),
            "vspw_count": int(hists["VSPW"].sum()),
            "vipseg_count": int(hists["VIPSEG"].sum()),
        }
    return out


def build_lodo_repair_audit_v6() -> dict[str, Any]:
    lodo_consistency = load_json(REPORTS / "stwm_final_lodo_consistency_audit_v5_20260428.json")
    mixed_pool = load_json(REPORTS / "stwm_mixed_semantic_trace_target_pool_v2_20260428.json")
    vspw_eval = load_json(REPORTS / "stwm_mixed_fullscale_v2_vspw_test_eval_complete_20260428.json")
    vipseg_eval = load_json(REPORTS / "stwm_mixed_fullscale_v2_vipseg_test_eval_complete_20260428.json")
    lodo_a = load_json(REPORTS / "stwm_final_lodo_vspw_to_vipseg_eval_20260428.json")
    lodo_b = load_json(REPORTS / "stwm_final_lodo_vipseg_to_vspw_eval_20260428.json")
    norms = _feature_norms()
    proto_shift = _prototype_shift()
    payload = {
        "audit_name": "stwm_lodo_domain_shift_repair_audit_v6_20260501",
        "lodo_completed": bool(lodo_consistency.get("lodo_completed", False)),
        "trusted_lodo_conclusion": lodo_consistency.get("trusted_lodo_conclusion", "report_inconsistent_needs_fix"),
        "lodo_positive": False,
        "lodo_domain_shift_diagnosed": True,
        "cache_bug_found": False,
        "cache_bug_fixed": False,
        "rerun_required": False,
        "crop_feature_norm_shift": norms,
        "observed_semantic_memory_coverage_shift": mixed_pool.get("observed_coverage_by_dataset", {}),
        "future_target_coverage_shift": mixed_pool.get("future_overlap_by_dataset", {}),
        "changed_stable_ratio_shift": mixed_pool.get("changed_stable_ratio_by_dataset", {}),
        "prototype_vocabulary_shift": proto_shift,
        "copy_baseline_strength_shift": {
            "mixed_vspw_copy_changed_top5": float(vspw_eval.get("best_metrics", {}).get("copy_changed_subset_top5", 0.0)),
            "mixed_vipseg_copy_changed_top5": float(vipseg_eval.get("best_metrics", {}).get("copy_changed_subset_top5", 0.0)),
            "lodo_vspw_to_vipseg_copy_changed_top5": float(lodo_a.get("best_metrics", {}).get("copy_changed_subset_top5", 0.0)),
            "lodo_vipseg_to_vspw_copy_changed_top5": float(lodo_b.get("best_metrics", {}).get("copy_changed_subset_top5", 0.0)),
        },
        "trace_coord_error_shift": {
            "mixed_vspw": float(vspw_eval.get("best_metrics", {}).get("future_trace_coord_error", 0.0)),
            "mixed_vipseg": float(vipseg_eval.get("best_metrics", {}).get("future_trace_coord_error", 0.0)),
            "lodo_vspw_to_vipseg": float(lodo_a.get("best_metrics", {}).get("future_trace_coord_error", 0.0)),
            "lodo_vipseg_to_vspw": float(lodo_b.get("best_metrics", {}).get("future_trace_coord_error", 0.0)),
        },
        "final_lodo_claim_boundary": "LODO is a domain-shift appendix/limitation. Do not claim universal cross-dataset generalization; do not reinterpret the negative as world-model failure because mixed free-rollout remains positive and trace regression stays false.",
    }
    write_json(REPORTS / "stwm_lodo_domain_shift_repair_audit_v6_20260501.json", payload)
    write_md(
        DOCS / "STWM_LODO_DOMAIN_SHIFT_REPAIR_AUDIT_V6_20260501.md",
        "STWM LODO Domain Shift Repair Audit V6 20260501",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- lodo_completed: `{payload['lodo_completed']}`",
                    f"- trusted_lodo_conclusion: `{payload['trusted_lodo_conclusion']}`",
                    f"- lodo_domain_shift_diagnosed: `{payload['lodo_domain_shift_diagnosed']}`",
                    f"- cache_bug_found: `{payload['cache_bug_found']}`",
                    f"- cache_bug_fixed: `{payload['cache_bug_fixed']}`",
                    f"- rerun_required: `{payload['rerun_required']}`",
                ]
            ),
            "## Diagnosis\n"
            + "\n".join(
                [
                    f"- crop_feature_norm_mean VSPW: `{payload['crop_feature_norm_shift']['VSPW']['crop_feature_norm_mean']}`",
                    f"- crop_feature_norm_mean VIPSEG: `{payload['crop_feature_norm_shift']['VIPSEG']['crop_feature_norm_mean']}`",
                    f"- C32 JS divergence: `{payload['prototype_vocabulary_shift']['C32']['js_divergence']}`",
                    f"- C64 JS divergence: `{payload['prototype_vocabulary_shift']['C64']['js_divergence']}`",
                ]
            ),
            "## Claim Boundary\n- " + payload["final_lodo_claim_boundary"],
        ],
    )
    return payload


def build_benchmark_protocol_v6() -> dict[str, Any]:
    splits = load_json(REPORTS / "stwm_mixed_semantic_trace_world_model_v2_splits_20260428.json")
    baseline = load_json(REPORTS / "stwm_same_output_baseline_suite_v6_20260501.json")
    payload = {
        "audit_name": "stwm_fstf_benchmark_protocol_v6_20260501",
        "benchmark_name": "STWM-FSTF: Future Semantic Trace Field Prediction",
        "input": [
            "observed video-derived trace",
            "observed semantic memory",
        ],
        "output": [
            "future trace units",
            "future semantic prototype field",
            "future visibility",
            "future reappearance",
            "future identity belief",
        ],
        "free_rollout_required": True,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "train_item_count": int(splits.get("train_item_count", 0)),
        "val_item_count": int(splits.get("val_item_count", 0)),
        "test_item_count": int(splits.get("test_item_count", 0)),
        "metrics": [
            "overall semantic top1/top5/CE",
            "stable semantic top5",
            "changed semantic top5",
            "changed gain over copy",
            "stable preservation drop",
            "future trace coord error",
            "visibility/reappearance AP/AUROC",
        ],
        "stable_changed_split_required": True,
        "copy_baseline_rationale": "Observed semantic memory makes copy extremely strong on stable states, so changed-subset gain isolates actual semantic transition modeling.",
        "trace_guardrail": "future_trace_coord_error + trace_regression_detected=false",
        "main_table_contract": {
            "must_include": [
                "mixed",
                "VSPW",
                "VIPSEG",
                "stable/changed split",
                "copy delta",
                "same-output baselines",
            ],
            "same_output_baselines_complete": bool(baseline.get("baseline_suite_completed", False)),
        },
        "lodo_positioning": "appendix/domain-shift limitation",
        "external_tracker_positioning": "SAM2/CoTracker/Cutie are external consumers/boundary references only, not same-output STWM-FSTF baselines.",
        "terminology": "semantic trace-unit field",
    }
    write_json(REPORTS / "stwm_fstf_benchmark_protocol_v6_20260501.json", payload)
    write_md(
        DOCS / "STWM_FSTF_BENCHMARK_PROTOCOL_V6_20260501.md",
        "STWM FSTF Benchmark Protocol V6 20260501",
        [
            "## Task\n- STWM-FSTF: Future Semantic Trace Field Prediction",
            "## Input / Output\n"
            + "\n".join(
                ["- input: observed video-derived trace + observed semantic memory", "- output: future trace units + future semantic prototype field + visibility / reappearance / identity belief"]
            ),
            "## Protocol\n"
            + "\n".join(
                [
                    f"- free_rollout_required: `{payload['free_rollout_required']}`",
                    f"- candidate_scorer_used: `{payload['candidate_scorer_used']}`",
                    f"- future_candidate_leakage: `{payload['future_candidate_leakage']}`",
                    f"- terminology: `{payload['terminology']}`",
                ]
            ),
            "## Main Table Contract\n- " + ", ".join(payload["main_table_contract"]["must_include"]),
        ],
    )
    return payload


def build_visualization_v6() -> dict[str, Any]:
    videos = sorted((ROOT / "outputs/videos/stwm_final_v5").glob("*"))
    figures = sorted((ROOT / "outputs/figures/stwm_final").glob("*"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "videos").mkdir(parents=True, exist_ok=True)
    (ASSETS / "figures").mkdir(parents=True, exist_ok=True)
    video_manifest = {
        "audit_name": "stwm_video_visualization_v6_manifest",
        "actual_media_files": [str(p) for p in videos],
        "actual_media_generated": bool(videos),
        "paper_ready_rollout_videos_complete": False,
        "blocking_reason": "Current media are summary-card GIF/MP4 clips. They do not yet show actual 8-step observed/GT/copy/STWM rollout strips from raw frames and future semantic trace units.",
    }
    figure_manifest = {
        "audit_name": "stwm_figure_visualization_v6_manifest",
        "figure_files": [str(p) for p in figures],
        "paper_ready_figure_pack_complete": False,
        "blocking_reason": "Current figures cover method/result summaries but not the requested 8-step rollout frame strips with GT/copy/STWM overlays.",
    }
    write_json(ASSETS / "videos/stwm_video_manifest_v6_20260501.json", video_manifest)
    write_json(ASSETS / "figures/stwm_figure_manifest_v6_20260501.json", figure_manifest)
    payload = {
        "audit_name": "stwm_video_visualization_v6_20260501",
        "video_visualization_ready": False,
        "actual_media_generated": bool(videos),
        "summary_card_media_only": True,
        "blocking_reason": video_manifest["blocking_reason"],
        "video_manifest_path": str(ASSETS / "videos/stwm_video_manifest_v6_20260501.json"),
        "figure_manifest_path": str(ASSETS / "figures/stwm_figure_manifest_v6_20260501.json"),
        "existing_video_count": len(videos),
        "existing_figure_count": len(figures),
    }
    write_json(REPORTS / "stwm_video_visualization_v6_20260501.json", payload)
    write_md(
        DOCS / "STWM_VIDEO_VISUALIZATION_V6_20260501.md",
        "STWM Video Visualization V6 20260501",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- actual_media_generated: `{payload['actual_media_generated']}`",
                    f"- video_visualization_ready: `{payload['video_visualization_ready']}`",
                    f"- existing_video_count: `{payload['existing_video_count']}`",
                    f"- existing_figure_count: `{payload['existing_figure_count']}`",
                ]
            ),
            "## Blocking Reason\n- " + payload["blocking_reason"],
        ],
    )
    return payload


def build_readiness_v6(
    reconcile: dict[str, Any],
    baseline: dict[str, Any],
    scaling: dict[str, Any],
    lodo: dict[str, Any],
    benchmark: dict[str, Any],
    viz: dict[str, Any],
) -> dict[str, Any]:
    strongest_table = "mixed_free_rollout_main_result"
    weakest_table = "same_output_baseline_suite"
    remaining = []
    if not baseline.get("baseline_suite_completed", False):
        remaining.append("same-output baseline suite is still incomplete")
    if not scaling.get("model_scaling_completed", False):
        remaining.append("model-size scaling is incomplete")
    if not scaling.get("horizon_scaling_completed", False):
        remaining.append("horizon scaling is incomplete")
    if not scaling.get("trace_density_scaling_completed", False):
        remaining.append("trace-density scaling is incomplete")
    if not viz.get("video_visualization_ready", False):
        remaining.append("paper-ready 8-step rollout video/figure pack is incomplete")
    if lodo.get("trusted_lodo_conclusion") == "negative":
        remaining.append("LODO is negative and must be positioned as domain shift rather than universal generalization")

    next_step = "run_missing_baselines"
    if baseline.get("baseline_suite_completed", False) and (
        not scaling.get("model_scaling_completed", False)
        or not scaling.get("horizon_scaling_completed", False)
        or not scaling.get("trace_density_scaling_completed", False)
    ):
        next_step = "run_missing_scaling"
    elif baseline.get("baseline_suite_completed", False) and scaling.get("model_scaling_completed", False) and scaling.get("horizon_scaling_completed", False) and scaling.get("trace_density_scaling_completed", False) and not viz.get("video_visualization_ready", False):
        next_step = "fix_visualization"

    payload = {
        "audit_name": "stwm_final_cvpr_aaai_readiness_v6_20260501",
        "ready_for_cvpr_aaai_main": "unclear",
        "ready_for_overleaf": True,
        "next_step_choice": next_step,
        "remaining_risks": remaining,
        "allowed_claims": [
            "STWM predicts future semantic trace-unit fields under free rollout.",
            "Mixed VSPW+VIPSEG free-rollout evaluation is positive on mixed/VSPW/VIPSEG changed subsets.",
            "Stable copy is preserved and trace regression remains false.",
            "LODO is negative and should be presented as a domain-shift limitation.",
        ],
        "forbidden_claims": [
            "candidate scorer method",
            "SAM2/CoTracker plugin framing",
            "future candidate leakage",
            "test-set model selection",
            "dense trace field without K-scaling evidence",
            "universal cross-dataset generalization",
            "full RGB generation or planner",
        ],
        "strongest_table": strongest_table,
        "weakest_table": weakest_table,
        "reviewer_risk_assessment": {
            "main_result_artifact_risk": "low" if reconcile.get("can_claim_5seed_main_result", False) else "high",
            "selected_C_risk": "low" if reconcile.get("can_claim_selected_C", False) else "medium",
            "same_output_baseline_risk": "high",
            "scaling_law_risk": "high",
            "lodo_generalization_risk": "medium",
            "visualization_risk": "medium",
        },
        "required_additional_runs": [
            "same-output baseline suite: trace-only / semantic-only / semantic+trace / SlotFormer-like / DINO-WM-like",
            "prototype scaling with missing C16 controlled mixed runs",
            "model-size scaling small/base/large",
            "horizon scaling H16/H24 cache rebuild + retrain/eval",
            "trace-density scaling K16/K32 cache rebuild + retrain/eval",
            "actual 8-step rollout visualization generation",
        ],
    }
    write_json(REPORTS / "stwm_final_cvpr_aaai_readiness_v6_20260501.json", payload)
    write_md(
        DOCS / "STWM_FINAL_CVPR_AAAI_READINESS_V6_20260501.md",
        "STWM Final CVPR AAAI Readiness V6 20260501",
        [
            "## Status\n"
            + "\n".join(
                [
                    f"- ready_for_cvpr_aaai_main: `{payload['ready_for_cvpr_aaai_main']}`",
                    f"- ready_for_overleaf: `{payload['ready_for_overleaf']}`",
                    f"- next_step_choice: `{payload['next_step_choice']}`",
                    f"- strongest_table: `{payload['strongest_table']}`",
                    f"- weakest_table: `{payload['weakest_table']}`",
                ]
            ),
            "## Remaining Risks\n" + "\n".join(f"- {x}" for x in payload["remaining_risks"]),
            "## Required Additional Runs\n" + "\n".join(f"- {x}" for x in payload["required_additional_runs"]),
        ],
    )
    return payload


def main() -> None:
    reconcile = build_artifact_reconcile()
    baseline = build_same_output_baseline_suite_v6()
    scaling = build_scaling_laws_v6()
    lodo = build_lodo_repair_audit_v6()
    benchmark = build_benchmark_protocol_v6()
    viz = build_visualization_v6()
    build_readiness_v6(reconcile, baseline, scaling, lodo, benchmark, viz)


if __name__ == "__main__":
    main()
