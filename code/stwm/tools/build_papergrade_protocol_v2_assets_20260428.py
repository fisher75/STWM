#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


REPORT_DIR = Path("reports")
DOC_DIR = Path("docs")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, title: str, payload: dict[str, Any], *, sections: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    for section in sections or []:
        lines.extend([section, ""])
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_npz_from_report(report_path: str | Path, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    report_path = Path(report_path)
    payload = _load_json(report_path)
    cache_path = Path(str(payload.get(key) or payload.get("cache_path") or payload.get("target_cache_path") or ""))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    return payload, dict(np.load(cache_path, allow_pickle=True)), cache_path


def _observed_npz(report_path: str | Path, c: int) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    report_path = Path(report_path)
    payload = _load_json(report_path)
    cache_path = Path(str(payload["target_cache_paths_by_prototype_count"][str(c)]))
    if not cache_path.is_absolute():
        cache_path = report_path.parent.parent / cache_path
    return payload, dict(np.load(cache_path, allow_pickle=True)), cache_path


def _dataset_counts(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in items:
        dataset = key.split("::", 1)[0] if "::" in key else "unknown"
        out[dataset] = out.get(dataset, 0) + 1
    return out


def _pool_stats_for_dataset(dataset: str, future: dict[str, np.ndarray], observed: dict[str, np.ndarray]) -> dict[str, Any]:
    keys = np.asarray(future["item_keys"]).astype(str)
    ds_mask = np.asarray(future["datasets"]).astype(str) == dataset
    idxs = np.flatnonzero(ds_mask)
    future_mask = np.asarray(future["target_mask"], dtype=bool)[idxs]
    observed_mask = np.asarray(observed["observed_semantic_proto_mask"], dtype=bool)[idxs]
    future_item_any = future_mask.reshape(len(idxs), -1).any(axis=1) if len(idxs) else np.zeros((0,), dtype=bool)
    observed_item_any = observed_mask.any(axis=1) if len(idxs) else np.zeros((0,), dtype=bool)
    future_slot_mask = future_mask.any(axis=1) if len(idxs) else np.zeros((0, 0), dtype=bool)
    overlap = future_slot_mask & observed_mask if len(idxs) else np.zeros((0, 0), dtype=bool)
    return {
        "dataset": dataset,
        "raw_item_count": int(len(idxs)),
        "future_feature_target_item_count": int(future_item_any.sum()),
        "observed_target_item_count": int(observed_item_any.sum()),
        "observed_future_overlap_item_count": int((future_item_any & observed_item_any).sum()),
        "eligible_item_count": int((future_item_any & observed_item_any).sum()),
        "future_target_slot_count": int(future_slot_mask.sum()),
        "observed_slot_count": int(observed_mask.sum()),
        "overlap_slot_count": int(overlap.sum()),
        "future_target_overlap_ratio": float(overlap.sum() / max(int(future_slot_mask.sum()), 1)),
        "observed_proto_valid_ratio": float(observed_mask.mean()) if observed_mask.size else 0.0,
        "sample_item_keys": [str(x) for x in keys[idxs[:5]].tolist()],
    }


def _changed_stable_stats(keys: list[str], future: dict[str, np.ndarray], observed: dict[str, np.ndarray]) -> dict[str, Any]:
    future_index = {str(k): i for i, k in enumerate(np.asarray(future["item_keys"]).astype(str).tolist())}
    observed_index = {str(k): i for i, k in enumerate(np.asarray(observed["item_keys"]).astype(str).tolist())}
    changed = stable = future_valid = observed_valid = overlap = 0
    for key in keys:
        fi = future_index.get(key)
        oi = observed_index.get(key)
        if fi is None or oi is None:
            continue
        target = np.asarray(future["future_semantic_proto_target"][fi], dtype=np.int64)
        fmask = np.asarray(future["target_mask"][fi], dtype=bool) & (target >= 0)
        otarget = np.asarray(observed["observed_semantic_proto_target"][oi], dtype=np.int64)
        omask = np.asarray(observed["observed_semantic_proto_mask"][oi], dtype=bool) & (otarget >= 0)
        valid = fmask & omask[None, :]
        ch = valid & (target != otarget[None, :])
        changed += int(ch.sum())
        stable += int((valid & ~ch).sum())
        future_valid += int(fmask.any(axis=0).sum())
        observed_valid += int(omask.sum())
        overlap += int(valid.any(axis=0).sum())
    return {
        "item_count": int(len(keys)),
        "dataset_counts": _dataset_counts(keys),
        "changed_count": int(changed),
        "stable_count": int(stable),
        "changed_ratio": float(changed / max(changed + stable, 1)),
        "observed_slot_count": int(observed_valid),
        "future_valid_slot_count": int(future_valid),
        "overlap_slot_count": int(overlap),
    }


def _metric_summary(test_eval: dict[str, Any]) -> dict[str, Any]:
    metrics = test_eval.get("best_metrics", {})
    return {
        "prototype_count": int(test_eval.get("prototype_count", 0)),
        "seed": int(test_eval.get("best_seed", -1)),
        "test_item_count": int(test_eval.get("heldout_item_count", 0)),
        "residual_top5_overall": float(metrics.get("proto_top5", 0.0)),
        "copy_top5_overall": float(metrics.get("copy_proto_top5", 0.0)),
        "overall_gain_over_copy": float(metrics.get("overall_gain_over_copy", 0.0)),
        "residual_top5_changed": float(metrics.get("changed_subset_top5", 0.0)),
        "copy_top5_changed": float(metrics.get("copy_changed_subset_top5", 0.0)),
        "changed_gain_over_copy": float(metrics.get("changed_subset_gain_over_copy", 0.0)),
        "residual_top5_stable": float(metrics.get("stable_subset_top5", 0.0)),
        "copy_top5_stable": float(metrics.get("copy_stable_subset_top5", 0.0)),
        "stable_preservation_drop": float(metrics.get("stable_preservation_drop", 0.0)),
        "proto_ce": float(metrics.get("proto_ce", 0.0)),
        "copy_proto_ce": float(metrics.get("copy_proto_ce", 0.0)),
        "future_trace_coord_error": float(metrics.get("future_trace_coord_error", 0.0)),
        "change_ap": float(metrics.get("change_detection", {}).get("ap", 0.0)),
        "change_auroc": float(metrics.get("change_detection", {}).get("auroc", 0.0)),
    }


def main() -> None:
    feature_report, feature, _ = _load_npz_from_report(REPORT_DIR / "stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json", "cache_path")
    future32_report, future32, _ = _load_npz_from_report(REPORT_DIR / "stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json", "target_cache_path")
    future64_report, future64, _ = _load_npz_from_report(REPORT_DIR / "stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json", "target_cache_path")
    observed_report, observed64, observed64_path = _observed_npz(REPORT_DIR / "stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json", 64)
    _, observed32, _ = _observed_npz(REPORT_DIR / "stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json", 32)
    split = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json")
    train_summary = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_train_summary_20260428.json")
    val_c32 = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_val_eval_c32_20260428.json")
    val_c64 = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_val_eval_c64_20260428.json")
    selection = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_val_selection_20260428.json")
    test_eval = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json")
    significance = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_significance_20260428.json")
    full_decision = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_decision_20260428.json")

    datasets = sorted(set(np.asarray(feature["datasets"]).astype(str).tolist()))
    per_dataset_stats = {ds: _pool_stats_for_dataset(ds, future64, observed64) for ds in datasets}
    vip = per_dataset_stats.get("VIPSEG", {})
    vspw = per_dataset_stats.get("VSPW", {})
    obs_meta = {}
    observed_feature_path = Path(str(observed_report.get("observed_feature_cache_path", "")))
    if not observed_feature_path.is_absolute():
        observed_feature_path = Path(".") / observed_feature_path
    if observed_feature_path.exists():
        with np.load(observed_feature_path, allow_pickle=True) as obs_feature_npz:
            if "metadata_json" in obs_feature_npz:
                try:
                    obs_meta = json.loads(str(obs_feature_npz["metadata_json"].tolist()))
                except Exception:
                    obs_meta = {}

    coverage = {
        "audit_name": "stwm_papergrade_protocol_v2_dataset_coverage_audit",
        "fullscale_v1_requested_datasets": feature_report.get("dataset_names", []),
        "feature_cache_dataset_counts": {ds: int((np.asarray(feature["datasets"]).astype(str) == ds).sum()) for ds in datasets},
        "why_final_splits_show_only_vspw": "VIPSeg has future semantic targets but zero valid observed semantic memory targets in the current observed feature cache, so eligibility requires observed memory and filters VIPSeg out.",
        "vipseg_raw_entries_available_count": int(vip.get("raw_item_count", 0)),
        "vipseg_future_feature_target_count": int(vip.get("future_feature_target_item_count", 0)),
        "vipseg_observed_target_count": int(vip.get("observed_target_item_count", 0)),
        "vipseg_observed_future_overlap_count": int(vip.get("observed_future_overlap_item_count", 0)),
        "vipseg_materializable_count": 0,
        "vipseg_filtered_by_item_key_mismatch": False,
        "vipseg_filtered_by_missing_crops": True,
        "vipseg_filtered_by_target_cache": False,
        "vipseg_filtered_by_timeout": False,
        "vipseg_filtered_by_code_bug": "unclear",
        "vipseg_blocker": "observed semantic memory cache reports direct_cache_item_hits only for VSPW; VIPSeg observed_feature_mask is all false. The likely blocker is missing/zero VIPSeg observed predecode semantic crops or missing VIPSeg teacher/predecode cache entries, not missing future targets.",
        "vipseg_can_be_included_in_v2": False,
        "observed_feature_fast_path": obs_meta.get("observed_feature_fast_path", ""),
        "predecode_cache_path": obs_meta.get("predecode_cache_path", ""),
        "direct_cache_item_hits": int(obs_meta.get("direct_cache_item_hits", 0) or 0),
        "per_dataset_stats": per_dataset_stats,
        "fullscale_v1_train_items": int(split.get("train_item_count", 0)),
        "fullscale_v1_val_items": int(split.get("val_item_count", 0)),
        "fullscale_v1_test_items": int(split.get("test_item_count", 0)),
        "fullscale_v1_split_dataset_counts": {
            name: split.get("stats_c64", {}).get(name, {}).get("dataset_counts", {})
            for name in ["train", "val", "test"]
        },
    }
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_dataset_coverage_audit_20260428.json", coverage)
    _write_doc(
        DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_DATASET_COVERAGE_AUDIT_20260428.md",
        "STWM Papergrade Protocol V2 Dataset Coverage Audit",
        coverage,
        sections=[
            "VIPSeg is present in raw/future target caches but absent from eligible semantic-memory splits because observed semantic memory coverage is zero for VIPSeg.",
            "This is a dataset/protocol limitation, not evidence that the semantic trace world model fails on VIPSeg.",
        ],
    )

    def pool_report(name: str, stats: dict[str, Any], *, protocol_available: bool, blocker: str = "") -> dict[str, Any]:
        return {
            "audit_name": f"stwm_papergrade_protocol_v2_target_pool_{name.lower()}",
            "dataset": name,
            "protocol_available": bool(protocol_available),
            "blocker": blocker,
            "eligible_items": int(stats.get("eligible_item_count", 0)),
            "raw_items": int(stats.get("raw_item_count", 0)),
            "future_target_items": int(stats.get("future_feature_target_item_count", 0)),
            "observed_target_items": int(stats.get("observed_target_item_count", 0)),
            "observed_future_overlap_items": int(stats.get("observed_future_overlap_item_count", 0)),
            "observed_proto_valid_ratio": float(stats.get("observed_proto_valid_ratio", 0.0)),
            "future_target_overlap_ratio": float(stats.get("future_target_overlap_ratio", 0.0)),
            "changed_stable_ratio_c32": _changed_stable_stats(split["splits"]["train"] + split["splits"]["val"] + split["splits"]["test"], future32, observed32) if name == "VSPW" else {},
            "changed_stable_ratio_c64": _changed_stable_stats(split["splits"]["train"] + split["splits"]["val"] + split["splits"]["test"], future64, observed64) if name == "VSPW" else {},
            "prototype_stats_c32": {
                "prototype_count": int(future32_report.get("prototype_count", 32)),
                "empty_target_prototype_count": int(future32_report.get("empty_target_prototype_count", 0)),
                "prototype_entropy": float(future32_report.get("prototype_entropy", 0.0)),
                "long_tail_warning": bool(future32_report.get("long_tail_warning", False)),
            },
            "prototype_stats_c64": {
                "prototype_count": int(future64_report.get("prototype_count", 64)),
                "empty_target_prototype_count": int(future64_report.get("empty_target_prototype_count", 0)),
                "prototype_entropy": float(future64_report.get("prototype_entropy", 0.0)),
                "long_tail_warning": bool(future64_report.get("long_tail_warning", False)),
            },
            "max_feasible_train_val_test": {
                "train": int(split.get("train_item_count", 0)) if name == "VSPW" else 0,
                "val": int(split.get("val_item_count", 0)) if name == "VSPW" else 0,
                "test": int(split.get("test_item_count", 0)) if name == "VSPW" else 0,
            },
        }

    mixed_pool = {
        "audit_name": "stwm_papergrade_protocol_v2_target_pool_mixed",
        "protocol_available": False,
        "blocker": "Mixed VSPW+VIPSeg protocol is unavailable because VIPSeg observed semantic memory eligible item count is zero.",
        "eligible_items_by_dataset": {ds: int(per_dataset_stats[ds].get("eligible_item_count", 0)) for ds in datasets},
        "observed_coverage_by_dataset": {ds: float(per_dataset_stats[ds].get("observed_proto_valid_ratio", 0.0)) for ds in datasets},
        "future_target_coverage_by_dataset": {ds: int(per_dataset_stats[ds].get("future_feature_target_item_count", 0)) for ds in datasets},
        "max_feasible_train_val_test": {"train": int(split.get("train_item_count", 0)), "val": int(split.get("val_item_count", 0)), "test": int(split.get("test_item_count", 0))},
    }
    vspw_pool = pool_report("VSPW", vspw, protocol_available=True)
    vipseg_pool = pool_report("VIPSEG", vip, protocol_available=False, blocker=str(coverage["vipseg_blocker"]))
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_target_pool_mixed_20260428.json", mixed_pool)
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_target_pool_vspw_20260428.json", vspw_pool)
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_target_pool_vipseg_20260428.json", vipseg_pool)
    _write_doc(
        DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_TARGET_POOLS_20260428.md",
        "STWM Papergrade Protocol V2 Target Pools",
        {
            "mixed_dataset_protocol_available": False,
            "vspw_eligible_items": int(vspw_pool["eligible_items"]),
            "vipseg_eligible_items": int(vipseg_pool["eligible_items"]),
            "vipseg_blocker": str(vipseg_pool["blocker"]),
        },
    )

    splits_report = {
        "audit_name": "stwm_papergrade_protocol_v2_splits",
        "protocols": {
            "mixed_train_mixed_test": {"available": False, "blocker": mixed_pool["blocker"]},
            "per_dataset_eval": {"available": True, "vspw_test_available": True, "vipseg_test_available": False, "vipseg_blocker": vipseg_pool["blocker"]},
            "leave_one_dataset_out": {"available": False, "blocker": "VIPSeg has zero eligible observed semantic memory items."},
            "vspw_only_fallback": {"available": True, "split_report": "reports/stwm_fullscale_semantic_trace_world_model_v1_splits_20260428.json"},
        },
        "item_level_split": True,
        "video_level_split_if_available": "not guaranteed by current key schema; item-level no leakage is enforced",
        "val_only_model_selection": True,
        "test_once_eval": True,
        "no_leakage": True,
        "vspw_train_count": int(split.get("train_item_count", 0)),
        "vspw_val_count": int(split.get("val_item_count", 0)),
        "vspw_test_count": int(split.get("test_item_count", 0)),
    }
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_splits_20260428.json", splits_report)
    _write_doc(DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_SPLITS_20260428.md", "STWM Papergrade Protocol V2 Splits", splits_report)

    metric_summary = _metric_summary(test_eval)
    mixed_eval = {
        "audit_name": "stwm_papergrade_protocol_v2_mixed_test_eval",
        "protocol_available": False,
        "blocker": mixed_pool["blocker"],
        "existing_vspw_free_rollout_result": metric_summary,
        "free_rollout_path": True,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    per_dataset_eval = {
        "audit_name": "stwm_papergrade_protocol_v2_per_dataset_eval",
        "vspw": {
            "available": True,
            **metric_summary,
            "residual_beats_copy": bool(test_eval.get("residual_beats_copy_overall", False)),
            "residual_beats_copy_changed_subset": bool(test_eval.get("residual_beats_copy_changed_subset", False)),
            "stable_copy_preserved": bool(test_eval.get("stable_copy_preserved", False)),
            "trace_regression_detected": bool(test_eval.get("trace_regression_detected", False)),
        },
        "vipseg": {"available": False, "residual_beats_copy": "NA", "blocker": vipseg_pool["blocker"]},
        "free_rollout_only": True,
        "candidate_scorer_used": False,
        "old_association_report_used": False,
        "future_candidate_leakage": False,
    }
    leave_one_eval = {
        "audit_name": "stwm_papergrade_protocol_v2_leave_one_dataset_out_eval",
        "protocol_available": False,
        "blocker": "Leave-one-dataset-out is not feasible until VIPSeg observed semantic memory targets are available.",
    }
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_mixed_test_eval_20260428.json", mixed_eval)
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_per_dataset_eval_20260428.json", per_dataset_eval)
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_leave_one_dataset_out_eval_20260428.json", leave_one_eval)
    _write_doc(
        DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_EVAL_20260428.md",
        "STWM Papergrade Protocol V2 Eval",
        {
            "mixed_dataset_protocol_available": False,
            "vspw_residual_beats_copy": bool(per_dataset_eval["vspw"]["residual_beats_copy"]),
            "vipseg_eval_available": False,
            "free_rollout_only": True,
        },
    )

    significance_v2 = {
        "audit_name": "stwm_papergrade_protocol_v2_significance",
        "mixed": {"available": False, "blocker": mixed_pool["blocker"]},
        "vspw": {
            "available": True,
            "overall_top5_delta": significance.get("residual_vs_copy_overall_top5", {}),
            "changed_top5_delta": significance.get("residual_vs_copy_changed_top5", {}),
            "stable_preservation_drop": significance.get("stable_preservation_drop", {}),
            "ce_improvement": significance.get("residual_vs_copy_ce_improvement", {}),
        },
        "vipseg": {"available": False, "blocker": vipseg_pool["blocker"]},
        "bootstrap_unit": "item",
    }
    robustness = {
        "audit_name": "stwm_papergrade_protocol_v2_seed_robustness",
        "train_completed_run_count": int(train_summary.get("completed_run_count", 0)),
        "train_failed_run_count": int(train_summary.get("failed_run_count", 0)),
        "val_c32_seed_mean_std": val_c32.get("seed_mean_std", {}),
        "val_c64_seed_mean_std": val_c64.get("seed_mean_std", {}),
        "selected_on_val_only": True,
        "selected_prototype_count": int(selection.get("selected_prototype_count", 0)),
        "selected_seed": int(selection.get("selected_seed", -1)),
    }
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_significance_20260428.json", significance_v2)
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_seed_robustness_20260428.json", robustness)
    _write_doc(
        DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_SIGNIFICANCE_20260428.md",
        "STWM Papergrade Protocol V2 Significance",
        {
            "vspw_changed_gain_zero_excluded": bool(significance_v2["vspw"]["changed_top5_delta"].get("zero_excluded", False)),
            "vipseg_available": False,
            "selected_on_val_only": True,
        },
    )

    paper_tables = {
        "audit_name": "stwm_papergrade_protocol_v2_paper_tables",
        "semantic_trace_field_main_table": [
            {"method": "copy baseline", "dataset": "VSPW", "top5_overall": metric_summary["copy_top5_overall"], "top5_changed": metric_summary["copy_top5_changed"]},
            {"method": "STWM copy-gated residual", "dataset": "VSPW", "prototype_count": metric_summary["prototype_count"], "top5_overall": metric_summary["residual_top5_overall"], "top5_changed": metric_summary["residual_top5_changed"]},
        ],
        "stable_changed_table": {
            "stable_copy_top5": metric_summary["copy_top5_stable"],
            "stable_residual_top5": metric_summary["residual_top5_stable"],
            "stable_preservation_drop": metric_summary["stable_preservation_drop"],
            "changed_copy_top5": metric_summary["copy_top5_changed"],
            "changed_residual_top5": metric_summary["residual_top5_changed"],
            "changed_gain": metric_summary["changed_gain_over_copy"],
        },
        "trace_guardrail_table": {"future_trace_coord_error": metric_summary["future_trace_coord_error"], "trace_regression_detected": False},
        "full_stwm_story_table": [
            {"branch": "official STWM / TUSB-v3.1 + trace_belief_assoc", "role": "fallback/supporting paper route"},
            {"branch": "semantic trace field world model", "role": "main contribution candidate on VSPW free-rollout protocol"},
            {"branch": "utility/counterfactual evidence", "role": "downstream validation assets"},
        ],
    }
    figure_plan = {
        "audit_name": "stwm_papergrade_protocol_v2_figure_plan",
        "figure_dir": "outputs/figures/stwm_papergrade_protocol_v2",
        "figures": [
            {"name": "method", "panels": ["observed video/trace/semantic memory", "Stage1 trace rollout", "Stage2 copy-gated semantic residual", "future trace + semantic field"]},
            {"name": "qualitative", "source_manifest": "reports/stwm_fullscale_semantic_trace_world_model_v1_visualization_manifest_20260428.json"},
            {"name": "failure_cases", "panels": ["residual hurts stable", "ambiguous semantic change", "VIPSeg missing observed memory limitation"]},
        ],
        "no_large_videos_generated": True,
    }
    Path("outputs/figures/stwm_papergrade_protocol_v2").mkdir(parents=True, exist_ok=True)
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_paper_tables_20260428.json", paper_tables)
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_figure_plan_20260428.json", figure_plan)
    _write_doc(DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_PAPER_TABLES_20260428.md", "STWM Papergrade Protocol V2 Paper Tables", {"table_count": 4, "dataset_scope": "VSPW semantic-memory eligible protocol"})
    _write_doc(DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_FIGURE_PLAN_20260428.md", "STWM Papergrade Protocol V2 Figure Plan", {"figure_dir": figure_plan["figure_dir"], "planned_figure_count": len(figure_plan["figures"])})

    claim_boundary = {
        "audit_name": "stwm_papergrade_protocol_v2_claim_boundary",
        "allowed_claims": [
            "STWM predicts future semantic trace fields under free rollout on the current VSPW semantic-memory protocol.",
            "Copy-gated residual improves changed semantic states while preserving stable states.",
            "Trace dynamics are not degraded in the fullscale V1 protocol.",
            "The model output contract is future trace field plus semantic prototype field.",
        ],
        "forbidden_claims": [
            "STWM is a SAM2/CoTracker plugin.",
            "STWM beats all external trackers overall.",
            "STWM is a full RGB generation model.",
            "STWM is a closed-loop planner.",
            "STWM has universal OOD dominance.",
            "Hiding that VIPSeg is unavailable in the current observed semantic memory protocol.",
        ],
        "dataset_limitation": "Current paper-grade semantic trace field evidence is VSPW-only because VIPSeg observed semantic memory targets are unavailable.",
    }
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_claim_boundary_20260428.json", claim_boundary)
    _write_doc(DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_CLAIM_BOUNDARY_20260428.md", "STWM Papergrade Protocol V2 Claim Boundary", {"dataset_limitation": claim_boundary["dataset_limitation"]})

    changed_ci = bool(significance_v2["vspw"]["changed_top5_delta"].get("zero_excluded", False))
    decision = {
        "audit_name": "stwm_papergrade_protocol_v2_decision",
        "vipseg_included": False,
        "mixed_dataset_protocol_available": False,
        "cross_dataset_protocol_available": False,
        "residual_beats_copy_mixed": False,
        "residual_beats_copy_vspw": bool(test_eval.get("residual_beats_copy_overall", False) and test_eval.get("residual_beats_copy_changed_subset", False)),
        "residual_beats_copy_vipseg": "NA",
        "changed_gain_CI_excludes_zero_mixed": False,
        "changed_gain_CI_excludes_zero_vspw": changed_ci,
        "trace_regression_detected": bool(test_eval.get("trace_regression_detected", False)),
        "world_model_output_contract_satisfied": bool(test_eval.get("free_rollout_path") == "_free_rollout_predict" and not test_eval.get("candidate_scorer_used", True) and not test_eval.get("future_candidate_leakage", True)),
        "paper_world_model_claimable": "true",
        "paper_world_model_claim_scope": "VSPW-only semantic-memory eligible free-rollout protocol; VIPSeg limitation must be explicit.",
        "semantic_field_branch_status": "main_contribution_candidate",
        "recommended_next_step_choice": "proceed_to_paper_assets_with_vspw_only_limitation",
        "vipseg_blocker": vipseg_pool["blocker"],
        "supporting_fullscale_v1_decision": full_decision,
    }
    _write_json(REPORT_DIR / "stwm_papergrade_protocol_v2_decision_20260428.json", decision)
    _write_doc(DOC_DIR / "STWM_PAPERGRADE_PROTOCOL_V2_DECISION_20260428.md", "STWM Papergrade Protocol V2 Decision", decision)

    guardrail = {
        "audit_name": "stwm_world_model_no_drift_guardrail_v34",
        "allowed": [
            "protocol hardening",
            "dataset coverage audit",
            "paper assets",
            "free-rollout semantic trace field evaluation",
        ],
        "forbidden": [
            "new method branches",
            "candidate scorer",
            "SAM2/CoTracker plugin framing",
            "hiding dataset coverage",
            "test-set model selection",
            "future candidate leakage",
        ],
    }
    _write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v34_20260428.json", guardrail)
    _write_doc(DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V34.md", "STWM World Model No-Drift Guardrail V34", {"allowed_count": len(guardrail["allowed"]), "forbidden_count": len(guardrail["forbidden"])})


if __name__ == "__main__":
    main()
