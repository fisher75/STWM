#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.materialize_ostf_point_selection_v22_20260502 import derive_rows_with_point_selection, shuffle_point_identities
from stwm.tools.ostf_multimodal_metrics_v21 import paired_bootstrap_from_rows
from stwm.tools.ostf_multimodal_metrics_v22 import calibration_summary, expected_vs_oracle_bootstrap
from stwm.tools.ostf_v17_common_20260502 import ROOT, OSTFObjectSample, collapse_to_m1, dump_json, load_json, write_doc
from stwm.tools.ostf_v18_common_20260502 import analytic_affine_motion_predict, analytic_constant_velocity_predict
from stwm.tools.train_ostf_calibrated_multimodal_v22_20260502 import (
    build_model,
    evaluate_model,
    prepare_rows_for_model,
)
from stwm.tools.ostf_v20_common_20260502 import load_context_cache


def _load_runs() -> dict[str, dict[str, Any]]:
    out = {}
    run_dir = ROOT / "reports/stwm_ostf_v22_runs"
    for path in sorted(run_dir.glob("*.json")):
        obj = load_json(path)
        if obj:
            exp_name = str(obj.get("experiment_name", ""))
            if exp_name.startswith("debug_") or exp_name.startswith("v22_tmp_"):
                continue
            out[exp_name] = obj
    return out


def _load_model(run_report: dict[str, Any], device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(ROOT / run_report["best_checkpoint_path"], map_location=device, weights_only=False)
    model = build_model(run_report["model_kind"], int(run_report["horizon"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _hard_score(report: dict[str, Any]) -> float:
    hard = report["test_subset_metrics"]["cv_hard_top20"]
    allm = report["test_metrics"]
    calib = report["test_calibration"]
    return (
        -1.10 * float(hard["top1_endpoint_error_px"])
        -0.55 * float(hard["top1_point_l1_px"])
        -0.35 * float(hard["minFDE_K_px"])
        +16.0 * float(hard["top1_PCK_16px"])
        +8.0 * float(hard["BestOfK_PCK_16px"])
        -8.0 * float(hard["top1_MissRate_32px"])
        -4.0 * float(hard["MissRate_32px"])
        +60.0 * float(hard["top1_object_extent_iou"])
        -0.15 * float(allm["weighted_point_l1_px"])
        +5.0 * float(calib["top1_mode_accuracy"])
        -2.0 * float(calib["ece_top1_mode"])
    )


def _analytic_rows(samples: list[OSTFObjectSample], baseline_kind: str) -> list[dict[str, Any]]:
    from stwm.tools.ostf_multimodal_metrics_v22 import multimodal_item_scores_v22
    from stwm.tools.train_ostf_calibrated_multimodal_v22_20260502 import combo_for_model, subset_flags

    if not samples:
        return []
    combo = combo_for_model("v22_calibrated_m128" if samples[0].m <= 128 else "v22_calibrated_m512", samples[0].h)
    ctx_map = load_context_cache(ROOT / "outputs/cache/stwm_ostf_context_features_v20" / f"{combo}_context_features.npz")
    fn = analytic_constant_velocity_predict if baseline_kind == "constant_velocity_copy" else analytic_affine_motion_predict
    pred_points, pred_vis, pred_sem = fn(samples, proto_count=32, proto_centers=None, semantic_mode="observed_memory")
    point_modes = pred_points[:, :, :, None, :]
    logits = pred_points.new_zeros((pred_points.shape[0], 1)) if hasattr(pred_points, "new_zeros") else None
    import numpy as np

    if logits is None:
        logits = np.zeros((pred_points.shape[0], 1), dtype=np.float32)
    return multimodal_item_scores_v22(
        samples,
        point_modes=point_modes,
        mode_logits=logits,
        point_pred=pred_points,
        top1_pred=pred_points,
        pred_vis_logits=pred_vis,
        pred_proto_logits=pred_sem,
        logvar_modes=None,
        subset_flags=subset_flags(samples, ctx_map),
        cv_mode_index=0,
    )


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from stwm.tools.ostf_multimodal_metrics_v22 import aggregate_rows_v22

    return {
        "all": aggregate_rows_v22(rows),
        "cv_hard_top20": aggregate_rows_v22(rows, subset_key="top20_cv_hard"),
        "occlusion": aggregate_rows_v22(rows, subset_key="occlusion_hard"),
        "nonlinear": aggregate_rows_v22(rows, subset_key="nonlinear_hard"),
        "interaction": aggregate_rows_v22(rows, subset_key="interaction_hard"),
        "calibration": calibration_summary(rows),
    }


def _eval_custom(
    model: torch.nn.Module,
    run_report: dict[str, Any],
    samples: list[OSTFObjectSample],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ctx_map = load_context_cache(ROOT / run_report["context_cache_path"])
    allm, _, subm, rows, calib = evaluate_model(model, samples, ctx_map, batch_size, device)
    return {"all": allm, "subsets": subm, "calibration": calib}, rows


def _zero_semantic(samples: list[OSTFObjectSample]) -> list[OSTFObjectSample]:
    return [replace(s, semantic_feat=s.semantic_feat * 0.0, semantic_valid=False) for s in samples]


def _bootstrap_dict(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "minFDE_all": paired_bootstrap_from_rows(a_rows, b_rows, metric="minFDE_K_px", higher_better=False),
        "minFDE_hard": paired_bootstrap_from_rows(a_rows, b_rows, metric="minFDE_K_px", higher_better=False, subset_key="top20_cv_hard"),
        "miss32_all": paired_bootstrap_from_rows(a_rows, b_rows, metric="MissRate_32px", higher_better=False),
        "miss32_hard": paired_bootstrap_from_rows(a_rows, b_rows, metric="MissRate_32px", higher_better=False, subset_key="top20_cv_hard"),
        "top1_fde_all": paired_bootstrap_from_rows(a_rows, b_rows, metric="top1_endpoint_error_px", higher_better=False),
        "top1_fde_hard": paired_bootstrap_from_rows(a_rows, b_rows, metric="top1_endpoint_error_px", higher_better=False, subset_key="top20_cv_hard"),
        "top1_miss32_all": paired_bootstrap_from_rows(a_rows, b_rows, metric="top1_MissRate_32px", higher_better=False),
        "top1_miss32_hard": paired_bootstrap_from_rows(a_rows, b_rows, metric="top1_MissRate_32px", higher_better=False, subset_key="top20_cv_hard"),
    }


def _bool_positive(boot: dict[str, Any]) -> bool:
    return bool(boot.get("zero_excluded") and (boot.get("mean_delta") or 0.0) > 0.0)


def _top1_close_or_beats(main_rows: list[dict[str, Any]], cv_rows: list[dict[str, Any]]) -> bool:
    boot = paired_bootstrap_from_rows(main_rows, cv_rows, metric="top1_endpoint_error_px", higher_better=False)
    if _bool_positive(boot):
        return True
    agg_main = _aggregate(main_rows)["all"]
    agg_cv = _aggregate(cv_rows)["all"]
    main_fde = float(agg_main["top1_endpoint_error_px"])
    cv_fde = float(agg_cv["top1_endpoint_error_px"])
    return main_fde <= cv_fde * 1.05


def main() -> int:
    runs = _load_runs()
    if not runs:
        raise SystemExit("No V22 run reports found in reports/stwm_ostf_v22_runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audit_v21 = load_json(ROOT / "reports/stwm_ostf_v21_mode_selection_audit_v22_20260502.json")

    train_summary = {
        "audit_name": "stwm_ostf_v22_train_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(runs),
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "target_m": r["target_m"],
                "point_selection_strategy": r["point_selection_strategy"],
                "steps": r["steps"],
                "parameter_count": r["parameter_count"],
                "best_checkpoint_path": r["best_checkpoint_path"],
                "best_step": r.get("best_step"),
                "best_val_score": r.get("best_val_score"),
            }
            for name, r in runs.items()
        },
    }

    main_candidates = [r for r in runs.values() if r["model_kind"] in {"v22_calibrated_m128", "v22_calibrated_m256", "v22_calibrated_m512"}]
    best_report = max(main_candidates, key=_hard_score)
    best_name = str(best_report["experiment_name"])
    best_model = _load_model(best_report, device)

    base_rows, _, _, _, _ = prepare_rows_for_model(best_report["model_kind"], int(best_report["horizon"]), int(best_report["seed"]))
    test_rows = base_rows["test"]
    cv_rows = _analytic_rows(test_rows, "constant_velocity_copy")
    affine_rows = _analytic_rows(test_rows, "affine_motion_prior")

    ablation_prefix = None
    if "m512" in best_name:
        ablation_prefix = "v22_calibrated_m512"
    elif "m256" in best_name:
        ablation_prefix = "v22_calibrated_m256"
    else:
        ablation_prefix = "v22_calibrated_m128"

    dense_ablation_rows = None
    dense_name = f"{ablation_prefix}_wo_dense_points_seed42_h8"
    if dense_name in runs:
        dense_ablation_rows = runs[dense_name]["item_scores"]
    semantic_ablation_rows = None
    semantic_name = f"{ablation_prefix}_wo_semantic_memory_seed42_h8"
    if semantic_name in runs:
        semantic_ablation_rows = runs[semantic_name]["item_scores"]
    context_ablation_rows = None
    context_name = f"{ablation_prefix}_wo_context_seed42_h8"
    if context_name in runs:
        context_ablation_rows = runs[context_name]["item_scores"]
    single_mode_rows = None
    single_name = f"{ablation_prefix}_single_mode_seed42_h8"
    if single_name in runs:
        single_mode_rows = runs[single_name]["item_scores"]

    m1_rows_dict = collapse_to_m1(base_rows)
    best_on_m1, best_on_m1_rows = _eval_custom(best_model, best_report, m1_rows_dict["test"], device, batch_size=4)
    boundary_rows_dict = derive_rows_with_point_selection(base_rows, target_m=min(32, test_rows[0].m), strategy="boundary_only", seed=42)
    interior_rows_dict = derive_rows_with_point_selection(base_rows, target_m=min(32, test_rows[0].m), strategy="interior_only", seed=42)
    motion_rows_dict = derive_rows_with_point_selection(base_rows, target_m=min(32, test_rows[0].m), strategy="high_motion", seed=42)
    shuffle_rows_test = [shuffle_point_identities(s, seed=42) for s in test_rows]

    boundary_metrics, boundary_item_rows = _eval_custom(best_model, best_report, boundary_rows_dict["test"], device, batch_size=4)
    interior_metrics, interior_item_rows = _eval_custom(best_model, best_report, interior_rows_dict["test"], device, batch_size=4)
    motion_metrics, motion_item_rows = _eval_custom(best_model, best_report, motion_rows_dict["test"], device, batch_size=4)
    shuffle_metrics, shuffle_item_rows = _eval_custom(best_model, best_report, shuffle_rows_test, device, batch_size=4)
    zero_sem_metrics, zero_sem_rows = _eval_custom(best_model, best_report, _zero_semantic(test_rows), device, batch_size=4)

    bootstrap = {
        "audit_name": "stwm_ostf_v22_bootstrap",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
        "v22_best_vs_cv": _bootstrap_dict(best_report["item_scores"], cv_rows),
        "v22_best_vs_affine": _bootstrap_dict(best_report["item_scores"], affine_rows),
        "best_of_k_vs_expected_all": expected_vs_oracle_bootstrap(best_report["item_scores"]),
        "best_of_k_vs_expected_hard": expected_vs_oracle_bootstrap(best_report["item_scores"], subset_key="top20_cv_hard"),
        "v22_best_vs_m1_dense_eval": _bootstrap_dict(best_report["item_scores"], best_on_m1_rows),
        "v22_best_vs_shuffle_identity": _bootstrap_dict(best_report["item_scores"], shuffle_item_rows),
        "v22_best_vs_zero_semantic_memory": _bootstrap_dict(best_report["item_scores"], zero_sem_rows),
        "v22_boundary_vs_interior": _bootstrap_dict(boundary_item_rows, interior_item_rows),
        "v22_motion_vs_boundary": _bootstrap_dict(motion_item_rows, boundary_item_rows),
    }
    if dense_ablation_rows is not None:
        bootstrap["v22_best_vs_wo_dense_points"] = _bootstrap_dict(best_report["item_scores"], dense_ablation_rows)
    if semantic_ablation_rows is not None:
        bootstrap["v22_best_vs_wo_semantic_memory"] = _bootstrap_dict(best_report["item_scores"], semantic_ablation_rows)
    if context_ablation_rows is not None:
        bootstrap["v22_best_vs_wo_context"] = _bootstrap_dict(best_report["item_scores"], context_ablation_rows)
    if single_mode_rows is not None:
        bootstrap["v22_best_vs_single_mode"] = _bootstrap_dict(best_report["item_scores"], single_mode_rows)

    cv_agg = _aggregate(cv_rows)
    affine_agg = _aggregate(affine_rows)
    mode_cal = calibration_summary(best_report["item_scores"])

    top1_vs_cv = bootstrap["v22_best_vs_cv"]["top1_fde_all"]
    top1_hard_vs_cv = bootstrap["v22_best_vs_cv"]["top1_fde_hard"]
    minfde_vs_cv = bootstrap["v22_best_vs_cv"]["minFDE_all"]
    minfde_hard_vs_cv = bootstrap["v22_best_vs_cv"]["minFDE_hard"]
    miss_vs_cv = bootstrap["v22_best_vs_cv"]["miss32_all"]
    miss_hard_vs_cv = bootstrap["v22_best_vs_cv"]["miss32_hard"]
    dense_load = False
    if dense_ablation_rows is not None:
        dense_load = _bool_positive(bootstrap["v22_best_vs_wo_dense_points"]["minFDE_hard"])
    if not dense_load:
        dense_load = _bool_positive(bootstrap["v22_best_vs_shuffle_identity"]["minFDE_hard"])
    if not dense_load:
        dense_load = _bool_positive(bootstrap["v22_best_vs_m1_dense_eval"]["minFDE_hard"])

    semantic_load = False
    if semantic_ablation_rows is not None:
        sem_boot = bootstrap["v22_best_vs_wo_semantic_memory"]
        semantic_load = _bool_positive(sem_boot["minFDE_hard"]) or _bool_positive(sem_boot["top1_fde_hard"])
        if not semantic_load:
            best_sem = float(best_report["test_metrics"].get("semantic_top5") or 0.0)
            no_sem = float(_aggregate(semantic_ablation_rows)["all"].get("semantic_top5") or 0.0)
            semantic_load = best_sem > no_sem + 0.01
    if not semantic_load:
        sem_boot = bootstrap["v22_best_vs_zero_semantic_memory"]
        semantic_load = _bool_positive(sem_boot["minFDE_hard"]) or _bool_positive(sem_boot["top1_fde_hard"])
        if not semantic_load:
            best_sem = float(best_report["test_metrics"].get("semantic_top5") or 0.0)
            zero_sem = float(_aggregate(zero_sem_rows)["all"].get("semantic_top5") or 0.0)
            semantic_load = best_sem > zero_sem + 0.01

    top1_close = _top1_close_or_beats(best_report["item_scores"], cv_rows)
    mode_selection_usable = bool(mode_cal["top1_mode_accuracy"] >= 0.32 and top1_close)
    calibrated_claim = bool(_bool_positive(minfde_hard_vs_cv) and _bool_positive(miss_hard_vs_cv) and mode_selection_usable and mode_cal["hypothesis_diversity_valid"])
    object_dense_claim = bool(calibrated_claim and dense_load and semantic_load)

    if calibrated_claim and dense_load and semantic_load:
        next_step = "run_v22_multiseed_H16"
    elif not mode_selection_usable:
        next_step = "improve_mode_calibration"
    elif not dense_load:
        next_step = "improve_dense_point_selection"
    elif not semantic_load:
        next_step = "fallback_to_FSTF_paper"
    else:
        next_step = "fallback_to_FSTF_paper"

    eval_summary = {
        "audit_name": "stwm_ostf_v22_eval_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "v21_mode_selection_audit_path": "reports/stwm_ostf_v21_mode_selection_audit_v22_20260502.json",
        "best_variant_name": best_name,
        "best_variant_kind": best_report["model_kind"],
        "experiments": {
            name: {
                "model_kind": r["model_kind"],
                "source_combo": r["source_combo"],
                "target_m": r["target_m"],
                "point_selection_strategy": r["point_selection_strategy"],
                "test_metrics": r["test_metrics"],
                "test_subset_metrics": r["test_subset_metrics"],
                "test_metrics_by_dataset": r["test_metrics_by_dataset"],
                "test_calibration": r["test_calibration"],
            }
            for name, r in runs.items()
        },
        "baselines": {
            "constant_velocity_copy": cv_agg,
            "affine_motion_prior": affine_agg,
        },
        "selection_experiments": {
            "m1_anchor_eval": best_on_m1,
            "boundary_only_eval": boundary_metrics,
            "interior_only_eval": interior_metrics,
            "high_motion_eval": motion_metrics,
            "shuffled_point_identity_eval": shuffle_metrics,
            "zero_semantic_memory_eval": zero_sem_metrics,
        },
        "mode_calibration": mode_cal,
        "cv_comparison": {
            "minFDE_all": minfde_vs_cv,
            "minFDE_hard": minfde_hard_vs_cv,
            "miss32_all": miss_vs_cv,
            "miss32_hard": miss_hard_vs_cv,
            "top1_fde_all": top1_vs_cv,
            "top1_fde_hard": top1_hard_vs_cv,
        },
        "semantic_fairness_note": "Semantic metrics use observed semantic memory / corrected logits only. No proto_target one-hot analytic oracle is used in V22 comparisons.",
    }

    decision = {
        "audit_name": "stwm_ostf_v22_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "best_variant_name": best_name,
        "cv_saturation_detected": bool(audit_v21.get("mode_selection_usable_without_oracle") is False) if audit_v21 else True,
        "mode_selection_usable_without_oracle": mode_selection_usable,
        "top1_mode_beats_CV_or_close": top1_close,
        "calibrated_multimodal_claim_allowed": calibrated_claim,
        "dense_points_load_bearing": dense_load,
        "semantic_coupling_load_bearing": semantic_load,
        "object_dense_semantic_trace_field_claim_allowed": object_dense_claim,
        "top1_mode_accuracy": mode_cal["top1_mode_accuracy"],
        "ece_top1_mode": mode_cal["ece_top1_mode"],
        "mode_nll_mean": mode_cal["mode_nll_mean"],
        "hypothesis_diversity_valid": mode_cal["hypothesis_diversity_valid"],
        "next_step_choice": next_step,
    }

    train_path = ROOT / "reports/stwm_ostf_v22_train_summary_20260502.json"
    eval_path = ROOT / "reports/stwm_ostf_v22_eval_summary_20260502.json"
    boot_path = ROOT / "reports/stwm_ostf_v22_bootstrap_20260502.json"
    decision_path = ROOT / "reports/stwm_ostf_v22_decision_20260502.json"
    doc_path = ROOT / "docs/STWM_OSTF_V22_RESULTS_20260502.md"

    dump_json(train_path, train_summary)
    dump_json(eval_path, eval_summary)
    dump_json(boot_path, bootstrap)
    dump_json(decision_path, decision)
    write_doc(
        doc_path,
        "STWM OSTF V22 Results",
        {
            "best_variant_name": best_name,
            "mode_selection_usable_without_oracle": mode_selection_usable,
            "top1_mode_beats_CV_or_close": top1_close,
            "calibrated_multimodal_claim_allowed": calibrated_claim,
            "dense_points_load_bearing": dense_load,
            "semantic_coupling_load_bearing": semantic_load,
            "object_dense_semantic_trace_field_claim_allowed": object_dense_claim,
            "next_step_choice": next_step,
        },
        [
            "best_variant_name",
            "mode_selection_usable_without_oracle",
            "top1_mode_beats_CV_or_close",
            "calibrated_multimodal_claim_allowed",
            "dense_points_load_bearing",
            "semantic_coupling_load_bearing",
            "object_dense_semantic_trace_field_claim_allowed",
            "next_step_choice",
        ],
    )
    print(eval_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
