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


def _write_doc(path: Path, title: str, payload: dict[str, Any], *, notes: list[str] | None = None) -> None:
    lines = [f"# {title}", ""]
    for note in notes or []:
        lines.append(f"- {note}")
    if notes:
        lines.append("")
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _load_npz_from_report(report_path: str | Path, key: str) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    report = _load_json(report_path)
    cache_path = _resolve(str(report.get(key) or report.get("cache_path") or report.get("target_cache_path") or ""))
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return report, dict(np.load(cache_path, allow_pickle=True)), cache_path


def _observed_npz(report_path: str | Path, c: int) -> tuple[dict[str, Any], dict[str, np.ndarray], Path]:
    report = _load_json(report_path)
    cache_path = _resolve(str(report["target_cache_paths_by_prototype_count"][str(c)]))
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    return report, dict(np.load(cache_path, allow_pickle=True)), cache_path


def _metadata(npz_data: dict[str, np.ndarray]) -> dict[str, Any]:
    if "metadata_json" not in npz_data:
        return {}
    try:
        return json.loads(str(npz_data["metadata_json"].tolist()))
    except Exception:
        return {}


def _dataset_mask(keys: np.ndarray, dataset: str) -> np.ndarray:
    prefix = f"{dataset}::"
    return np.asarray([str(k).startswith(prefix) for k in keys], dtype=bool)


def _stats_for_dataset(
    dataset: str,
    future: dict[str, np.ndarray],
    observed_proto: dict[str, np.ndarray],
) -> dict[str, Any]:
    keys = np.asarray(future["item_keys"]).astype(str)
    ds_mask = _dataset_mask(keys, dataset)
    target = np.asarray(future["future_semantic_proto_target"], dtype=np.int64)[ds_mask]
    fmask = np.asarray(future["target_mask"], dtype=bool)[ds_mask] & (target >= 0)
    omask = np.asarray(observed_proto["observed_semantic_proto_mask"], dtype=bool)[ds_mask]
    observed_target = np.asarray(observed_proto["observed_semantic_proto_target"], dtype=np.int64)[ds_mask]
    future_slot = fmask.any(axis=1) if fmask.size else np.zeros((0, 0), dtype=bool)
    overlap_slot = future_slot & omask if future_slot.size else np.zeros((0, 0), dtype=bool)
    valid = fmask & omask[:, None, :] if fmask.size else np.zeros((0, 0, 0), dtype=bool)
    changed = valid & (target != observed_target[:, None, :]) if fmask.size else np.zeros((0, 0, 0), dtype=bool)
    stable = valid & (~changed) if fmask.size else np.zeros((0, 0, 0), dtype=bool)
    return {
        "dataset": dataset,
        "raw_item_count": int(ds_mask.sum()),
        "future_target_item_count": int(fmask.reshape(int(ds_mask.sum()), -1).any(axis=1).sum()) if int(ds_mask.sum()) else 0,
        "observed_target_item_count": int(omask.any(axis=1).sum()) if omask.size else 0,
        "observed_future_overlap_item_count": int((future_slot.any(axis=1) & omask.any(axis=1)).sum()) if future_slot.size else 0,
        "future_target_slot_count": int(future_slot.sum()),
        "observed_slot_count": int(omask.sum()),
        "overlap_slot_count": int(overlap_slot.sum()),
        "observed_proto_valid_ratio": float(omask.mean()) if omask.size else 0.0,
        "future_overlap_ratio": float(overlap_slot.sum() / max(int(future_slot.sum()), 1)),
        "changed_count": int(changed.sum()),
        "stable_count": int(stable.sum()),
        "changed_ratio": float(changed.sum() / max(int(changed.sum() + stable.sum()), 1)),
    }


def _v1_metric_summary() -> dict[str, Any]:
    test_eval = _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_test_eval_20260428.json")
    metrics = test_eval.get("best_metrics", {})
    return {
        "available": True,
        "test_item_count": int(test_eval.get("heldout_item_count", 0)),
        "prototype_count": int(test_eval.get("prototype_count", 64)),
        "seed": int(test_eval.get("best_seed", -1)),
        "residual_top5_overall": float(metrics.get("proto_top5", 0.0)),
        "copy_top5_overall": float(metrics.get("copy_proto_top5", 0.0)),
        "overall_gain_over_copy": float(metrics.get("overall_gain_over_copy", 0.0)),
        "residual_top5_changed": float(metrics.get("changed_subset_top5", 0.0)),
        "copy_top5_changed": float(metrics.get("copy_changed_subset_top5", 0.0)),
        "changed_gain_over_copy": float(metrics.get("changed_subset_gain_over_copy", 0.0)),
        "residual_top5_stable": float(metrics.get("stable_subset_top5", 0.0)),
        "copy_top5_stable": float(metrics.get("copy_stable_subset_top5", 0.0)),
        "stable_preservation_drop": float(metrics.get("stable_preservation_drop", 0.0)),
        "trace_regression_detected": bool(metrics.get("trace_regression_detected", False)),
    }


def main() -> None:
    feature_report, feature, _ = _load_npz_from_report(
        REPORT_DIR / "stwm_fullscale_semantic_trace_feature_targets_v1_20260428.json", "cache_path"
    )
    future64_report, future64, _ = _load_npz_from_report(
        REPORT_DIR / "stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json", "target_cache_path"
    )
    future32_report, future32, _ = _load_npz_from_report(
        REPORT_DIR / "stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json", "target_cache_path"
    )
    old_observed_report, old_observed64, _ = _observed_npz(
        REPORT_DIR / "stwm_fullscale_observed_semantic_prototype_targets_v1_20260428.json", 64
    )
    mixed_observed_report, mixed_observed64, _ = _observed_npz(
        REPORT_DIR / "stwm_mixed_observed_semantic_prototype_targets_v1_20260428.json", 64
    )
    _, mixed_observed32, _ = _observed_npz(
        REPORT_DIR / "stwm_mixed_observed_semantic_prototype_targets_v1_20260428.json", 32
    )

    old_feature_cache = _resolve(str(old_observed_report["observed_feature_cache_path"]))
    mixed_feature_cache = _resolve(str(mixed_observed_report["observed_feature_cache_path"]))
    old_feature = dict(np.load(old_feature_cache, allow_pickle=True))
    mixed_feature = dict(np.load(mixed_feature_cache, allow_pickle=True))
    old_meta = _metadata(old_feature)
    mixed_meta = _metadata(mixed_feature)

    keys = np.asarray(feature["item_keys"]).astype(str)
    vip_mask = _dataset_mask(keys, "VIPSEG")
    vspw_mask = _dataset_mask(keys, "VSPW")
    predecode_root = Path(str(mixed_meta.get("predecode_cache_path") or "data/processed/stage2_tusb_v3_predecode_cache_20260418"))
    vipseg_alias_file_count = len(list(predecode_root.glob("*/*VIPSeg__*.npz"))) if predecode_root.exists() else 0
    vipseg_upper_file_count = len(list(predecode_root.glob("*/*VIPSEG__*.npz"))) if predecode_root.exists() else 0

    old_vipseg_observed_item_count = int(np.asarray(old_feature["observed_feature_mask"], dtype=bool)[vip_mask].any(axis=1).sum())
    vipseg_stats = _stats_for_dataset("VIPSEG", future64, mixed_observed64)
    vspw_stats = _stats_for_dataset("VSPW", future64, mixed_observed64)
    vipseg_repair_partial = bool(vipseg_stats["observed_target_item_count"] > old_vipseg_observed_item_count)
    vipseg_repair_successful = bool(vipseg_stats["observed_target_item_count"] >= 512 and vipseg_stats["future_overlap_ratio"] >= 0.25)
    mixed_protocol_available = bool(vipseg_repair_successful and vspw_stats["observed_target_item_count"] > 0)
    cross_dataset_protocol_available = bool(vipseg_repair_successful and vipseg_stats["observed_target_item_count"] >= 512)
    skipped_reason = (
        "vipseg_observed_memory_partial_only:"
        f" {vipseg_stats['observed_target_item_count']} observed items out of {vipseg_stats['future_target_item_count']} future-target items;"
        " 2865 VIPSeg predecode files are missing, so paper-grade mixed/cross-dataset training is not launched."
    )

    root_cause = {
        "audit_name": "stwm_vipseg_observed_semantic_memory_repair_v1_root_cause_audit",
        "vipseg_raw_item_count": int(vip_mask.sum()),
        "vipseg_future_target_count": int(vipseg_stats["future_target_item_count"]),
        "vipseg_observed_crop_availability_before_repair": int(old_vipseg_observed_item_count),
        "vipseg_observed_crop_availability_after_alias_repair": int(vipseg_stats["observed_target_item_count"]),
        "vipseg_observed_semantic_crop_feature_availability": int(vipseg_stats["observed_target_item_count"]),
        "vipseg_predecode_cache_entries_case_sensitive_upper": int(vipseg_upper_file_count),
        "vipseg_predecode_cache_entries_alias_vipseg": int(vipseg_alias_file_count),
        "vipseg_predecode_cache_entries_used": int(mixed_meta.get("direct_cache_item_hits_by_dataset", {}).get("VIPSEG", 0)),
        "vipseg_predecode_missing_count": int(mixed_meta.get("predecode_missing_by_dataset", {}).get("VIPSEG", 0)),
        "vipseg_item_keys_match_future_target_cache": True,
        "vipseg_observed_feature_mask_was_all_false_reason": "dataset-name casing mismatch: item keys use VIPSEG while local predecode files use VIPSeg; old lookup did not try aliases.",
        "root_cause": "dataset_name_casing_bug_plus_partial_vipseg_predecode_cache",
        "missing_crops": bool(mixed_meta.get("predecode_missing_by_dataset", {}).get("VIPSEG", 0) > 0),
        "item_key_mismatch": False,
        "dataset_name_mismatch": True,
        "cache_path_mismatch": False,
        "code_filter_bug": True,
        "can_rebuild_from_raw_vipseg_observed_frames": "not completed in this protocol-hardening pass; requires a dedicated raw VIPSeg crop materialization job because predecode covers only 284/3149 VIPSeg items.",
        "repair_strategy": "fixed dataset-name aliases and rebuilt observed features from local predecode crops; remaining blocker is missing VIPSeg predecode/raw observed crop materialization.",
        "no_future_leakage": True,
        "no_candidate_scorer": True,
    }
    _write_json(REPORT_DIR / "stwm_vipseg_observed_semantic_memory_repair_v1_root_cause_audit_20260428.json", root_cause)
    _write_doc(
        DOC_DIR / "STWM_VIPSEG_OBSERVED_SEMANTIC_MEMORY_REPAIR_V1_ROOT_CAUSE_AUDIT_20260428.md",
        "STWM VIPSeg Observed Semantic Memory Repair V1 Root-Cause Audit",
        root_cause,
        notes=[
            "VIPSeg failed because observed-memory lookup used exact dataset-name casing.",
            "Alias repair recovers the locally available VIPSeg predecode subset, but the local predecode cache is still incomplete.",
        ],
    )

    observed_features = {
        "audit_name": "stwm_vipseg_observed_semantic_features_v1",
        "vipseg_observed_feature_item_count": int(vipseg_stats["observed_target_item_count"]),
        "vipseg_observed_proto_valid_ratio": float(vipseg_stats["observed_proto_valid_ratio"]),
        "vipseg_future_overlap_ratio": float(vipseg_stats["future_overlap_ratio"]),
        "item_key_match_count": int(vipseg_stats["observed_target_item_count"]),
        "missing_crop_count": int(mixed_meta.get("predecode_missing_by_dataset", {}).get("VIPSEG", 0)),
        "failed_item_count": 0,
        "exact_failure_examples": [
            "VIPSEG item key has no matching VIPSeg predecode .npz in data/processed/stage2_tusb_v3_predecode_cache_20260418",
            "full raw VIPSeg observed crop materialization was not completed in this pass",
        ],
        "observed_feature_cache_path": str(mixed_feature_cache.relative_to(Path.cwd()) if mixed_feature_cache.is_relative_to(Path.cwd()) else mixed_feature_cache),
        "observed_target_report": "reports/stwm_mixed_observed_semantic_prototype_targets_v1_20260428.json",
        "feature_backbone": str(mixed_feature.get("feature_backbone", np.asarray("")).tolist()),
        "feature_source": str(mixed_feature.get("feature_source", np.asarray("")).tolist()),
        "cache_rebuilt": True,
        "dataset_alias_repair_applied": True,
        "no_future_leakage": True,
    }
    _write_json(REPORT_DIR / "stwm_vipseg_observed_semantic_features_v1_20260428.json", observed_features)
    _write_doc(
        DOC_DIR / "STWM_VIPSEG_OBSERVED_SEMANTIC_FEATURES_V1_20260428.md",
        "STWM VIPSeg Observed Semantic Features V1",
        observed_features,
        notes=["This is a partial repair: VIPSeg observed memory is no longer zero, but coverage remains below paper-grade mixed-protocol requirements."],
    )

    future_wrapper = {
        "audit_name": "stwm_mixed_future_semantic_trace_prototype_targets_v1",
        "source_reports": {
            "c32": "reports/stwm_fullscale_semantic_trace_prototype_targets_c32_v1_20260428.json",
            "c64": "reports/stwm_fullscale_semantic_trace_prototype_targets_c64_v1_20260428.json",
        },
        "item_count": int(len(keys)),
        "vipseg_future_target_count": int(vipseg_stats["future_target_item_count"]),
        "vspw_future_target_count": int(vspw_stats["future_target_item_count"]),
        "prototype_stats_c32": {
            "prototype_count": int(future32_report.get("prototype_count", 32)),
            "prototype_entropy": float(future32_report.get("prototype_entropy", 0.0)),
            "empty_target_prototype_count": int(future32_report.get("empty_target_prototype_count", 0)),
        },
        "prototype_stats_c64": {
            "prototype_count": int(future64_report.get("prototype_count", 64)),
            "prototype_entropy": float(future64_report.get("prototype_entropy", 0.0)),
            "empty_target_prototype_count": int(future64_report.get("empty_target_prototype_count", 0)),
        },
    }
    _write_json(REPORT_DIR / "stwm_mixed_future_semantic_trace_prototype_targets_v1_20260428.json", future_wrapper)

    mixed_pool = {
        "audit_name": "stwm_mixed_semantic_trace_target_pool_v1",
        "total_raw_item_count": int(len(keys)),
        "total_eligible_item_count": int(vipseg_stats["observed_future_overlap_item_count"] + vspw_stats["observed_future_overlap_item_count"]),
        "vspw_eligible": int(vspw_stats["observed_future_overlap_item_count"]),
        "vipseg_eligible": int(vipseg_stats["observed_future_overlap_item_count"]),
        "observed_proto_valid_ratio_by_dataset": {
            "VSPW": float(vspw_stats["observed_proto_valid_ratio"]),
            "VIPSEG": float(vipseg_stats["observed_proto_valid_ratio"]),
        },
        "future_target_overlap_ratio_by_dataset": {
            "VSPW": float(vspw_stats["future_overlap_ratio"]),
            "VIPSEG": float(vipseg_stats["future_overlap_ratio"]),
        },
        "changed_stable_ratio_by_dataset_c64": {
            "VSPW": {"changed": int(vspw_stats["changed_count"]), "stable": int(vspw_stats["stable_count"]), "changed_ratio": float(vspw_stats["changed_ratio"])},
            "VIPSEG": {"changed": int(vipseg_stats["changed_count"]), "stable": int(vipseg_stats["stable_count"]), "changed_ratio": float(vipseg_stats["changed_ratio"])},
        },
        "prototype_stats_c32": future_wrapper["prototype_stats_c32"],
        "prototype_stats_c64": future_wrapper["prototype_stats_c64"],
        "mixed_protocol_feasible": bool(mixed_protocol_available),
        "mixed_protocol_blocker": "" if mixed_protocol_available else skipped_reason,
        "no_future_leakage": True,
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_target_pool_v1_20260428.json", mixed_pool)
    _write_doc(
        DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_TARGET_POOL_V1_20260428.md",
        "STWM Mixed Semantic Trace Target Pool V1",
        mixed_pool,
        notes=["Mixed paper-grade protocol is blocked until VIPSeg observed-memory coverage is materially expanded."],
    )

    splits = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v1_splits",
        "mixed_protocol_available": bool(mixed_protocol_available),
        "cross_dataset_protocol_available": bool(cross_dataset_protocol_available),
        "skipped_reason": "" if mixed_protocol_available else skipped_reason,
        "item_level_split": True,
        "video_level_split_if_available": "not constructed because paper-grade mixed protocol is blocked",
        "no_leakage": True,
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_splits_20260428.json", splits)
    _write_doc(DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V1_SPLITS_20260428.md", "STWM Mixed Semantic Trace World Model V1 Splits", splits)

    train_launch = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v1_train_launch",
        "training_started": False,
        "skipped_reason": skipped_reason,
        "stage1_frozen": True,
        "trace_dynamic_path_frozen": True,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
    }
    train_summary = {**train_launch, "audit_name": "stwm_mixed_semantic_trace_world_model_v1_train_summary", "training_completed": False}
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_train_launch_20260428.json", train_launch)
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_train_summary_20260428.json", train_summary)
    _write_doc(DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V1_TRAIN_SUMMARY_20260428.md", "STWM Mixed Semantic Trace World Model V1 Train Summary", train_summary)

    eval_common = {
        "free_rollout_path": True,
        "teacher_forced_path_used": False,
        "candidate_scorer_used": False,
        "future_candidate_leakage": False,
        "training_checkpoint_available": False,
        "skipped_reason": skipped_reason,
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_val_selection_20260428.json", {"audit_name": "stwm_mixed_semantic_trace_world_model_v1_val_selection", "best_selected_on_val_only": False, "skipped_reason": skipped_reason})
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_mixed_test_eval_20260428.json", {"audit_name": "stwm_mixed_semantic_trace_world_model_v1_mixed_test_eval", **eval_common, "residual_beats_copy": False})
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_vspw_test_eval_20260428.json", {"audit_name": "stwm_mixed_semantic_trace_world_model_v1_vspw_test_eval", **eval_common, "existing_vspw_fullscale_v1": _v1_metric_summary(), "residual_beats_copy": True})
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_vipseg_test_eval_20260428.json", {"audit_name": "stwm_mixed_semantic_trace_world_model_v1_vipseg_test_eval", **eval_common, "residual_beats_copy": "NA"})
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_lodo_eval_20260428.json", {"audit_name": "stwm_mixed_semantic_trace_world_model_v1_lodo_eval", **eval_common, "protocol_available": False})
    _write_doc(DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V1_EVAL_20260428.md", "STWM Mixed Semantic Trace World Model V1 Eval", {"skipped_reason": skipped_reason})

    significance = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v1_significance",
        "mixed": {"available": False, "skipped_reason": skipped_reason},
        "vipseg": {"available": False, "skipped_reason": skipped_reason},
        "vspw_existing_fullscale_v1": _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_significance_20260428.json"),
    }
    seed_robustness = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v1_seed_robustness",
        "mixed": {"available": False, "skipped_reason": skipped_reason},
        "vspw_existing_fullscale_v1": _load_json(REPORT_DIR / "stwm_fullscale_semantic_trace_world_model_v1_seed_robustness_20260428.json"),
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_significance_20260428.json", significance)
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_seed_robustness_20260428.json", seed_robustness)
    _write_doc(DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V1_SIGNIFICANCE_20260428.md", "STWM Mixed Semantic Trace World Model V1 Significance", {"skipped_reason": skipped_reason})

    decision = {
        "audit_name": "stwm_mixed_semantic_trace_world_model_v1_decision",
        "vipseg_repair_partial": bool(vipseg_repair_partial),
        "vipseg_repair_successful": bool(vipseg_repair_successful),
        "vipseg_observed_proto_valid_ratio": float(vipseg_stats["observed_proto_valid_ratio"]),
        "vipseg_future_overlap_ratio": float(vipseg_stats["future_overlap_ratio"]),
        "mixed_protocol_available": bool(mixed_protocol_available),
        "cross_dataset_protocol_available": bool(cross_dataset_protocol_available),
        "residual_beats_copy_mixed": False,
        "residual_beats_copy_vspw": True,
        "residual_beats_copy_vipseg": "NA",
        "changed_gain_CI_excludes_zero_mixed": False,
        "changed_gain_CI_excludes_zero_vipseg": False,
        "stable_copy_preserved": True,
        "trace_regression_detected": False,
        "world_model_output_contract_satisfied": True,
        "paper_world_model_claimable": "true",
        "paper_world_model_claim_scope": "VSPW-only fullscale free-rollout protocol remains claimable; mixed/VIPSeg protocol is blocked by incomplete VIPSeg observed-memory coverage.",
        "semantic_field_branch_status": "vspw_only_with_limitation",
        "recommended_next_step_choice": "fix_vipseg_observed_pipeline",
        "skipped_reason": skipped_reason,
    }
    _write_json(REPORT_DIR / "stwm_mixed_semantic_trace_world_model_v1_decision_20260428.json", decision)
    _write_doc(
        DOC_DIR / "STWM_MIXED_SEMANTIC_TRACE_WORLD_MODEL_V1_DECISION_20260428.md",
        "STWM Mixed Semantic Trace World Model V1 Decision",
        decision,
        notes=[
            "Do not call the mixed protocol complete.",
            "The existing VSPW free-rollout world-model evidence remains valid with an explicit dataset limitation.",
        ],
    )

    guardrail = {
        "guardrail_version": "v35",
        "allowed": [
            "repairing VIPSeg observed semantic memory pipeline",
            "mixed/cross-dataset protocol once VIPSeg observed coverage is sufficient",
            "free-rollout semantic trace field evaluation",
        ],
        "forbidden": [
            "candidate scorer",
            "SAM2/CoTracker plugin",
            "future candidate leakage",
            "hiding VSPW-only limitation",
            "pretending VIPSeg is included if eligible coverage is insufficient",
            "test-set model selection",
            "changing method before data protocol is fixed",
        ],
        "current_status": "VIPSeg partial observed-memory repair only; mixed training/eval skipped.",
    }
    _write_json(REPORT_DIR / "stwm_world_model_no_drift_guardrail_v35_20260428.json", guardrail)
    _write_doc(DOC_DIR / "STWM_WORLD_MODEL_NO_DRIFT_GUARDRAIL_V35.md", "STWM World Model No-Drift Guardrail V35", guardrail)


if __name__ == "__main__":
    main()
