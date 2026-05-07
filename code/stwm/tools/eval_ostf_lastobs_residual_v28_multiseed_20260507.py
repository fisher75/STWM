#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_lastobs_v28_common_20260502 import (  # noqa: E402
    ROOT,
    add_v28_flags_to_item_rows,
    build_v28_rows,
    choose_visibility_aware_gamma_on_val,
    predict_damped_velocity,
    predict_last,
    predict_last_visible_copy,
    predict_median_object_anchor_copy,
    predict_visibility_aware_cv,
    predict_visibility_aware_damped_velocity,
    v28_subset_aggregate,
    visibility_logits_last_visible,
)
from stwm.tools.ostf_traceanything_metrics_v26_20260502 import (  # noqa: E402
    aggregate_item_rows_v26,
    multimodal_item_scores_v26,
    paired_bootstrap_from_rows_v26,
)
from stwm.tools.ostf_v17_common_20260502 import dump_json, write_doc  # noqa: E402
from stwm.tools.ostf_v27_prior_utils_20260502 import observed_memory_logits  # noqa: E402


RUN_DIR = ROOT / "reports/stwm_ostf_v28_runs"
SUMMARY_PATH = ROOT / "reports/stwm_ostf_v28_multiseed_summary_20260507.json"
BOOT_PATH = ROOT / "reports/stwm_ostf_v28_multiseed_bootstrap_20260507.json"
DECISION_PATH = ROOT / "reports/stwm_ostf_v28_multiseed_decision_20260507.json"
DOC_PATH = ROOT / "docs/STWM_OSTF_V28_MULTISEED_DECISION_20260507.md"
STRONG_PRIOR_PATH = ROOT / "reports/stwm_ostf_v28_stronger_prior_eval_20260507.json"
CONTRACT_AUDIT_PATH = ROOT / "reports/stwm_ostf_v28_contract_and_claims_audit_20260507.json"
SEEDS = [42, 123, 456, 789, 2026]
ABLATION_SEEDS = [42, 123, 456]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_runs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not RUN_DIR.exists():
        return out
    for path in sorted(RUN_DIR.glob("*.json")):
        try:
            x = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = str(x.get("experiment_name") or path.stem)
        out[name] = x
    return out


def _complete_run(run: dict[str, Any] | None) -> bool:
    if not run:
        return False
    if not run.get("item_scores") or not run.get("test_metrics"):
        return False
    for key in ("best_checkpoint_path", "final_checkpoint_path"):
        rel = run.get(key)
        if not rel:
            return False
        p = ROOT / str(rel)
        if not p.exists() or p.stat().st_size <= 0:
            return False
    return True


def _expected_names() -> list[str]:
    names: list[str] = []
    for seed in SEEDS:
        names.append(f"v28_lastobs_m128_h64_seed{seed}")
        names.append(f"v28_lastobs_m128_h32_seed{seed}")
    for seed in ABLATION_SEEDS:
        names.extend(
            [
                f"v28_lastobs_m128_h64_wo_dense_points_seed{seed}",
                f"v28_lastobs_m128_h64_wo_semantic_memory_seed{seed}",
                f"v28_lastobs_m128_h64_wo_residual_modes_seed{seed}",
                f"v28_lastobs_m128_h64_prior_only_seed{seed}",
            ]
        )
    return names


def _rows_for_pred(samples: list[Any], proto_centers: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    rows = multimodal_item_scores_v26(
        samples,
        point_modes=pred[:, :, :, None, :],
        mode_logits=np.zeros((len(samples), 1), dtype=np.float32),
        top1_point_pred=pred,
        weighted_point_pred=pred,
        pred_vis_logits=visibility_logits_last_visible(samples),
        pred_proto_logits=observed_memory_logits(samples, proto_centers, proto_count=32),
        pred_logvar=None,
        cv_mode_index=0,
    )
    return add_v28_flags_to_item_rows(rows, samples)


def _prior_rows(combo: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    split_rows, proto_centers, damped_gamma, _ = build_v28_rows(combo, seed=42)
    test_rows = split_rows["test"]
    val_gamma, val_scores = choose_visibility_aware_gamma_on_val(split_rows["val"], proto_centers)
    preds = {
        "last_observed_copy": predict_last(test_rows),
        "damped_velocity": predict_damped_velocity(test_rows, damped_gamma),
        "last_visible_copy": predict_last_visible_copy(test_rows),
        "visibility_aware_damped_velocity": predict_visibility_aware_damped_velocity(test_rows, val_gamma),
        "visibility_aware_cv": predict_visibility_aware_cv(test_rows),
        "median_object_anchor_copy": predict_median_object_anchor_copy(test_rows),
    }
    rows = {name: _rows_for_pred(test_rows, proto_centers, pred) for name, pred in preds.items()}
    summary = {
        "combo": combo,
        "damped_gamma": damped_gamma,
        "visibility_aware_gamma": val_gamma,
        "visibility_aware_val_scores": val_scores,
        "priors": {
            name: {
                "test_metrics": aggregate_item_rows_v26(rs),
                "test_subset_metrics": v28_subset_aggregate(rs),
                "test_metrics_by_dataset": {ds: aggregate_item_rows_v26(rs, dataset=ds) for ds in sorted({r["dataset"] for r in rs})},
            }
            for name, rs in rows.items()
        },
    }
    return rows, summary


def _seeded(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        x = copy.deepcopy(r)
        x["item_key"] = f"seed{seed}::{x['item_key']}"
        out.append(x)
    return out


def _collect(runs: dict[str, dict[str, Any]], prefix: str, seeds: list[int]) -> dict[int, dict[str, Any]]:
    return {seed: runs[f"{prefix}_seed{seed}"] for seed in seeds if _complete_run(runs.get(f"{prefix}_seed{seed}"))}


def _seed_stats(seed_runs: dict[int, dict[str, Any]], metric: str, subset_key: str | None = None) -> dict[str, Any]:
    vals = []
    for seed, run in sorted(seed_runs.items()):
        if subset_key:
            value = run.get("test_subset_metrics", {}).get(subset_key, {}).get(metric)
        else:
            value = run.get("test_metrics", {}).get(metric)
        if value is not None and np.isfinite(float(value)):
            vals.append(float(value))
    return {
        "completed_seed_count": len(vals),
        "mean": float(mean(vals)) if vals else None,
        "std": float(pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None,
        "values": vals,
    }


def _boot(
    out: dict[str, Any],
    key: str,
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    metric: str,
    higher_better: bool,
    subset_key: str | None = None,
) -> None:
    out[key] = paired_bootstrap_from_rows_v26(rows_a, rows_b, metric=metric, higher_better=higher_better, subset_key=subset_key)
    out[key]["metric"] = metric
    out[key]["subset_key"] = subset_key


def _positive(boot: dict[str, Any], *, allow_nonzero: bool = True) -> bool:
    if not boot or boot.get("mean_delta") is None:
        return False
    if allow_nonzero:
        return bool(boot.get("zero_excluded") and float(boot.get("mean_delta")) > 0.0)
    return bool(float(boot.get("mean_delta")) > 0.0)


def _negative(boot: dict[str, Any]) -> bool:
    return bool(boot and boot.get("zero_excluded") and boot.get("mean_delta") is not None and float(boot["mean_delta"]) < 0.0)


def _combine_seed_rows(seed_runs: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed, run in sorted(seed_runs.items()):
        rows.extend(_seeded(run["item_scores"], seed))
    return rows


def _combine_prior_rows(prior_rows: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        rows.extend(_seeded(prior_rows, seed))
    return rows


def _strongest_prior_name(prior_summary: dict[str, Any], subset_key: str = "last_observed_hard_top20") -> str:
    best = ("", float("inf"))
    for name, payload in prior_summary.get("priors", {}).items():
        metric = payload.get("test_subset_metrics", {}).get(subset_key, {}).get("minFDE_K_px")
        if metric is None:
            metric = payload.get("test_metrics", {}).get("minFDE_K_px")
        if metric is not None and float(metric) < best[1]:
            best = (name, float(metric))
    return best[0] or "last_visible_copy"


def _semantic_trace_status(boot: dict[str, Any]) -> str:
    if not boot or boot.get("mean_delta") is None:
        return "unknown"
    if _positive(boot):
        return "positive"
    if _negative(boot):
        return "negative"
    return "neutral"


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    runs = _load_runs()
    expected = _expected_names()
    completed = {name: _complete_run(runs.get(name)) for name in expected}
    missing = [name for name, ok in completed.items() if not ok]
    partial = bool(missing)

    prior_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    prior_summaries: dict[str, Any] = {}
    for combo in ["M128_H32", "M128_H64"]:
        rows, summary = _prior_rows(combo)
        prior_rows[combo] = rows
        prior_summaries[combo] = summary

    main_h32 = _collect(runs, "v28_lastobs_m128_h32", SEEDS)
    main_h64 = _collect(runs, "v28_lastobs_m128_h64", SEEDS)
    dense_h64 = _collect(runs, "v28_lastobs_m128_h64_wo_dense_points", ABLATION_SEEDS)
    sem_h64 = _collect(runs, "v28_lastobs_m128_h64_wo_semantic_memory", ABLATION_SEEDS)
    res_h64 = _collect(runs, "v28_lastobs_m128_h64_wo_residual_modes", ABLATION_SEEDS)
    prior_only_h64 = _collect(runs, "v28_lastobs_m128_h64_prior_only", ABLATION_SEEDS)

    summary: dict[str, Any] = {
        "summary_name": "stwm_ostf_v28_multiseed_summary",
        "generated_at_utc": generated_at,
        "partial": partial,
        "expected_run_count": len(expected),
        "completed_run_count": sum(1 for ok in completed.values() if ok),
        "missing_run_count": len(missing),
        "missing_run_names": missing,
        "stronger_prior_eval_path": str(STRONG_PRIOR_PATH.relative_to(ROOT)) if STRONG_PRIOR_PATH.exists() else None,
        "contract_audit_path": str(CONTRACT_AUDIT_PATH.relative_to(ROOT)) if CONTRACT_AUDIT_PATH.exists() else None,
        "prior_summaries": prior_summaries,
        "per_seed": {},
        "seed_level_mean_std": {},
    }

    for group_name, group_runs in {
        "H32_M128": main_h32,
        "H64_M128": main_h64,
        "H64_wo_dense_points": dense_h64,
        "H64_wo_semantic_memory": sem_h64,
        "H64_wo_residual_modes": res_h64,
        "H64_prior_only": prior_only_h64,
    }.items():
        summary["per_seed"][group_name] = {
            str(seed): {
                "experiment_name": run.get("experiment_name"),
                "minFDE_K_px_all": run.get("test_metrics", {}).get("minFDE_K_px"),
                "minFDE_K_px_last_observed_hard": run.get("test_subset_metrics", {}).get("last_observed_hard_top20", {}).get("minFDE_K_px"),
                "MissRate_32px_all": run.get("test_metrics", {}).get("MissRate_32px"),
                "MissRate_32px_last_observed_hard": run.get("test_subset_metrics", {}).get("last_observed_hard_top20", {}).get("MissRate_32px"),
                "semantic_top1": run.get("test_metrics", {}).get("semantic_top1"),
                "semantic_top5": run.get("test_metrics", {}).get("semantic_top5"),
                "visibility_F1": run.get("test_metrics", {}).get("top1_visibility_F1"),
                "per_dataset": run.get("test_metrics_by_dataset", {}),
            }
            for seed, run in sorted(group_runs.items())
        }
        summary["seed_level_mean_std"][group_name] = {
            "minFDE_K_px_all": _seed_stats(group_runs, "minFDE_K_px"),
            "minFDE_K_px_last_observed_hard": _seed_stats(group_runs, "minFDE_K_px", "last_observed_hard_top20"),
            "MissRate_32px_all": _seed_stats(group_runs, "MissRate_32px"),
            "semantic_top1": _seed_stats(group_runs, "semantic_top1"),
            "semantic_top5": _seed_stats(group_runs, "semantic_top5"),
            "visibility_F1": _seed_stats(group_runs, "top1_visibility_F1"),
        }

    bootstrap: dict[str, Any] = {
        "bootstrap_name": "stwm_ostf_v28_multiseed_bootstrap",
        "generated_at_utc": generated_at,
        "partial": partial,
        "comparisons": {},
    }
    comp = bootstrap["comparisons"]

    for combo, group_runs in [("M128_H32", main_h32), ("M128_H64", main_h64)]:
        if not group_runs:
            continue
        seeds = sorted(group_runs)
        model_rows = _combine_seed_rows(group_runs)
        for prior in ["last_visible_copy", "visibility_aware_damped_velocity", "last_observed_copy", "median_object_anchor_copy"]:
            base_rows = _combine_prior_rows(prior_rows[combo][prior], seeds)
            prefix = f"{combo}_vs_{prior}"
            _boot(comp, f"{prefix}_all_minfde", model_rows, base_rows, "minFDE_K_px", False)
            _boot(comp, f"{prefix}_hard_minfde", model_rows, base_rows, "minFDE_K_px", False, "last_observed_hard_top20")
            _boot(comp, f"{prefix}_all_miss32", model_rows, base_rows, "MissRate_32px", False)
            _boot(comp, f"{prefix}_hard_miss32", model_rows, base_rows, "MissRate_32px", False, "last_observed_hard_top20")

    paired_ablation_seeds = sorted(set(main_h64) & set(dense_h64))
    if paired_ablation_seeds:
        _boot(
            comp,
            "H64_dense_points_load_bearing_hard_minfde",
            _combine_seed_rows({s: main_h64[s] for s in paired_ablation_seeds}),
            _combine_seed_rows({s: dense_h64[s] for s in paired_ablation_seeds}),
            "minFDE_K_px",
            False,
            "last_observed_hard_top20",
        )
        _boot(
            comp,
            "H64_dense_shape_extent_load_bearing_hard_iou",
            _combine_seed_rows({s: main_h64[s] for s in paired_ablation_seeds}),
            _combine_seed_rows({s: dense_h64[s] for s in paired_ablation_seeds}),
            "top1_object_extent_iou",
            True,
            "last_observed_hard_top20",
        )

    paired_sem_seeds = sorted(set(main_h64) & set(sem_h64))
    if paired_sem_seeds:
        _boot(
            comp,
            "H64_semantic_field_load_bearing_top5",
            _combine_seed_rows({s: main_h64[s] for s in paired_sem_seeds}),
            _combine_seed_rows({s: sem_h64[s] for s in paired_sem_seeds}),
            "semantic_top5",
            True,
        )
        _boot(
            comp,
            "H64_semantic_trace_dynamics_hard_minfde",
            _combine_seed_rows({s: main_h64[s] for s in paired_sem_seeds}),
            _combine_seed_rows({s: sem_h64[s] for s in paired_sem_seeds}),
            "minFDE_K_px",
            False,
            "last_observed_hard_top20",
        )

    paired_res_seeds = sorted(set(main_h64) & set(res_h64))
    if paired_res_seeds:
        _boot(
            comp,
            "H64_residual_modes_load_bearing_hard_minfde",
            _combine_seed_rows({s: main_h64[s] for s in paired_res_seeds}),
            _combine_seed_rows({s: res_h64[s] for s in paired_res_seeds}),
            "minFDE_K_px",
            False,
            "last_observed_hard_top20",
        )
    paired_prior_seeds = sorted(set(main_h64) & set(prior_only_h64))
    if paired_prior_seeds:
        _boot(
            comp,
            "H64_vs_prior_only_hard_minfde",
            _combine_seed_rows({s: main_h64[s] for s in paired_prior_seeds}),
            _combine_seed_rows({s: prior_only_h64[s] for s in paired_prior_seeds}),
            "minFDE_K_px",
            False,
            "last_observed_hard_top20",
        )

    strongest_h64_prior = _strongest_prior_name(prior_summaries["M128_H64"])
    h64_vs_strong_hard = comp.get(f"M128_H64_vs_{strongest_h64_prior}_hard_minfde", {})
    h64_vs_last_visible_hard = comp.get("M128_H64_vs_last_visible_copy_hard_minfde", {})
    h64_vs_vis_damped_hard = comp.get("M128_H64_vs_visibility_aware_damped_velocity_hard_minfde", {})
    h64_miss_last_visible = comp.get("M128_H64_vs_last_visible_copy_hard_miss32", {})
    h64_miss_vis_damped = comp.get("M128_H64_vs_visibility_aware_damped_velocity_hard_miss32", {})
    dense_lb = _positive(comp.get("H64_dense_points_load_bearing_hard_minfde", {})) or _positive(comp.get("H64_dense_shape_extent_load_bearing_hard_iou", {}))
    semantic_field_lb = _positive(comp.get("H64_semantic_field_load_bearing_top5", {}))
    semantic_trace_status = _semantic_trace_status(comp.get("H64_semantic_trace_dynamics_hard_minfde", {}))
    h64_beats_strong_prior = _positive(h64_vs_strong_hard)
    h64_beats_last_visible = _positive(h64_vs_last_visible_hard)
    h64_beats_vis_damped = _positive(h64_vs_vis_damped_hard)
    h64_missrate_positive = _positive(h64_miss_last_visible) and _positive(h64_miss_vis_damped)

    if h64_beats_last_visible and h64_beats_vis_damped and h64_missrate_positive and semantic_field_lb and not partial:
        next_step = "run_H96_M512_scaling"
    elif semantic_field_lb and semantic_trace_status == "negative":
        next_step = "improve_semantic_identity_bridge"
    elif not h64_beats_last_visible and not h64_beats_vis_damped and main_h64:
        next_step = "fix_benchmark_if_last_visible_prior_dominates"
    elif (h64_beats_last_visible or h64_beats_vis_damped) and partial:
        next_step = "expand_traceanything_cache_to_1k_clips"
    elif not dense_lb and not semantic_field_lb and not h64_beats_strong_prior:
        next_step = "fallback_to_FSTF_only_for_backup"
    else:
        next_step = "expand_traceanything_cache_to_1k_clips"

    decision = {
        "decision_name": "stwm_ostf_v28_multiseed_decision",
        "generated_at_utc": generated_at,
        "partial": partial,
        "completed_run_count": summary["completed_run_count"],
        "expected_run_count": summary["expected_run_count"],
        "missing_run_names": missing,
        "strongest_causal_prior_H64": strongest_h64_prior,
        "H64_beats_strongest_causal_prior_hard_minFDE": h64_beats_strong_prior,
        "H64_beats_last_visible_copy_hard_minFDE": h64_beats_last_visible,
        "H64_beats_visibility_aware_damped_hard_minFDE": h64_beats_vis_damped,
        "H64_MissRate32_improves_vs_last_visible_and_visibility_aware_damped": h64_missrate_positive,
        "dense_points_load_bearing": dense_lb,
        "semantic_field_load_bearing": semantic_field_lb,
        "semantic_trace_dynamics_load_bearing": semantic_trace_status,
        "seed_level_stability_available": not partial and len(main_h64) == len(SEEDS),
        "paired_bootstrap_over_item_rows": True,
        "next_step_choice": next_step,
    }

    dump_json(SUMMARY_PATH, summary)
    dump_json(BOOT_PATH, bootstrap)
    dump_json(DECISION_PATH, decision)
    write_doc(
        DOC_PATH,
        "STWM OSTF V28 Multiseed Decision",
        {
            "generated_at_utc": generated_at,
            "partial": partial,
            "completed_run_count": summary["completed_run_count"],
            "expected_run_count": summary["expected_run_count"],
            "strongest_causal_prior_H64": strongest_h64_prior,
            "H64_beats_strongest_causal_prior_hard_minFDE": h64_beats_strong_prior,
            "dense_points_load_bearing": dense_lb,
            "semantic_field_load_bearing": semantic_field_lb,
            "semantic_trace_dynamics_load_bearing": semantic_trace_status,
            "next_step_choice": next_step,
        },
        [
            "partial",
            "completed_run_count",
            "expected_run_count",
            "strongest_causal_prior_H64",
            "H64_beats_strongest_causal_prior_hard_minFDE",
            "dense_points_load_bearing",
            "semantic_field_load_bearing",
            "semantic_trace_dynamics_load_bearing",
            "next_step_choice",
        ],
    )
    print(SUMMARY_PATH.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
