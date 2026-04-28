#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import math
import os


OFFICIAL_METHOD = "TUSB-v3.1::best_semantic_hard.pt"
OFFICIAL_SCORING = "trace_belief_assoc"
SUBSETS = [
    "long_gap_persistence",
    "occlusion_reappearance",
    "crossing_ambiguity",
    "OOD_hard",
    "appearance_change",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_repo_root(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_root = os.environ.get("STWM_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path.cwd().resolve()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    if mx is None or my is None:
        return None
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def binary_auc(scores: list[float], labels: list[int]) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    positives = [s for s, y in zip(scores, labels) if int(y) == 1]
    negatives = [s for s, y in zip(scores, labels) if int(y) == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = float(len(positives) * len(negatives))
    for ps in positives:
        for ns in negatives:
            wins += 1.0 if ps > ns else 0.5 if ps == ns else 0.0
    return wins / total


def binary_ap(scores: list[float], labels: list[int]) -> float | None:
    if not scores or len(scores) != len(labels):
        return None
    total_pos = sum(1 for y in labels if int(y) == 1)
    if total_pos <= 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
    hits = 0
    precisions = []
    for rank, idx in enumerate(order, start=1):
        if int(labels[idx]) == 1:
            hits += 1
            precisions.append(hits / rank)
    return sum(precisions) / max(total_pos, 1)


def row_is_official(row: dict[str, Any]) -> bool:
    return (
        str(row.get("method_name")) == OFFICIAL_METHOD
        and str(row.get("scoring_mode")) in {OFFICIAL_SCORING, "trace_belief_assoc"}
    )


def aggregate_old_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "top1": None,
            "MRR": None,
            "false_confuser_rate": None,
            "visibility_AUROC": None,
            "visibility_accuracy": None,
            "uncertainty_ECE": None,
            "gap_length_decay": None,
            "visibility_metric_status": "unavailable_no_future_visibility_field",
            "uncertainty_metric_status": "unavailable_no_confidence_or_uncertainty_field",
        }
    top1 = [float(r.get("query_future_top1_acc") or 0.0) for r in rows]
    mrr = [float(r.get("mrr") or 0.0) for r in rows]
    return {
        "row_count": len(rows),
        "unique_item_count": len({str(r.get("protocol_item_id")) for r in rows}),
        "top1": mean(top1),
        "MRR": mean(mrr),
        "false_confuser_rate": 1.0 - float(mean(top1) or 0.0),
        "visibility_AUROC": None,
        "visibility_accuracy": None,
        "uncertainty_ECE": None,
        "gap_length_decay": None,
        "visibility_metric_status": "unavailable_no_future_visibility_field",
        "uncertainty_metric_status": "unavailable_no_confidence_or_uncertainty_field",
    }


def subset_rows(rows: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        tags = row.get("subset_tags")
        if isinstance(tags, list) and subset in {str(x) for x in tags}:
            out.append(row)
        elif isinstance(tags, dict) and bool(tags.get(subset)):
            out.append(row)
    return out


def hash_rows(rows: list[dict[str, Any]]) -> str:
    compact = [
        {
            "protocol_item_id": r.get("protocol_item_id"),
            "seed": r.get("seed"),
            "method_name": r.get("method_name"),
            "scoring_mode": r.get("scoring_mode"),
            "top1": r.get("query_future_top1_acc"),
            "rank": r.get("target_rank"),
        }
        for r in rows
    ]
    return hashlib.sha256(json.dumps(compact, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def hash_items(items: list[dict[str, Any]]) -> str:
    compact = [
        {
            "item_id": item.get("item_id"),
            "protocol_item_id": item.get("protocol_item_id"),
            "valid": item.get("future_semantic_trace_state_valid"),
            "visibility": item.get("target_visibility"),
            "coord_error": item.get("future_trace_coord_error"),
        }
        for item in items
    ]
    return hashlib.sha256(json.dumps(compact, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def run_old_report_mode(source_report: Path, out_report: Path, out_doc: Path, repo_root: Path) -> dict[str, Any]:
    source = load_json(source_report)
    panels_out: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for panel_name, panel in (source.get("panels") or {}).items():
        rows = [r for r in (panel.get("per_item_results") or []) if isinstance(r, dict) and row_is_official(r)]
        all_rows.extend(rows)
        panels_out[panel_name] = {
            "panel_name": panel_name,
            "task_definitions": {
                "future_target_grounding": "given observed query entity, rank future candidate entity",
                "future_visibility_reappearance_prediction": "blocked until explicit future visibility logits are emitted",
                "false_confuser_rejection": "measured as 1 - future target grounding top1",
                "semantic_hard_subset_breakdown": SUBSETS,
            },
            "overall": aggregate_old_rows(rows),
            "semantic_hard_subset_breakdown": {subset: aggregate_old_rows(subset_rows(rows, subset)) for subset in SUBSETS},
            "per_item_results_hash": hash_rows(rows),
        }
    payload = {
        "generated_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "mode": "read_old_association_report",
        "source_report": str(source_report),
        "official_method": OFFICIAL_METHOD,
        "official_scoring_mode": OFFICIAL_SCORING,
        "fresh_eval_executed": False,
        "read_only_existing_per_item_eval": True,
        "future_semantic_trace_field_available": False,
        "semantic_state_eval_consumed_future_semantic_trace_state": False,
        "old_association_report_only": True,
        "future_semantic_query_eval_added": True,
        "task_metrics": [
            "top1",
            "MRR",
            "false_confuser_rate",
            "visibility_AUROC_or_accuracy",
            "uncertainty_ECE_if_available",
            "gap_length_decay_if_available",
        ],
        "exact_blocking_reason_for_visibility_auroc": "old association reports do not contain FutureSemanticTraceState future_visibility_logit outputs",
        "exact_blocking_reason_for_uncertainty_ece": "old association reports do not contain FutureSemanticTraceState future_uncertainty outputs",
        "panels": panels_out,
        "overall_all_panels": aggregate_old_rows(all_rows),
        "all_panels_per_item_results_hash": hash_rows(all_rows),
    }
    write_json(out_report, payload)
    write_doc(out_doc, payload)
    return payload


def aggregate_export_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    valid_items = [it for it in items if bool(it.get("future_semantic_trace_state_valid", True))]
    valid_ratio = len(valid_items) / max(len(items), 1)
    visibility_probs = [float(it["future_visibility_prob"]) for it in valid_items if it.get("future_visibility_prob") is not None]
    visibility_labels = [int(it["target_visibility"]) for it in valid_items if it.get("target_visibility") is not None and it.get("future_visibility_prob") is not None]
    visibility_scores_for_labels = [float(it["future_visibility_prob"]) for it in valid_items if it.get("target_visibility") is not None and it.get("future_visibility_prob") is not None]
    visibility_acc = None
    if visibility_labels:
        visibility_acc = mean([1.0 if (s >= 0.5) == bool(y) else 0.0 for s, y in zip(visibility_scores_for_labels, visibility_labels)])
    coord_errors = [float(it["future_trace_coord_error"]) for it in valid_items if it.get("future_trace_coord_error") is not None]
    uncertainties = [float(it["future_uncertainty_mean"]) for it in valid_items if it.get("future_uncertainty_mean") is not None]
    paired_uncertainties = []
    paired_errors = []
    for it in valid_items:
        if it.get("future_uncertainty_mean") is not None and it.get("future_trace_coord_error") is not None:
            paired_uncertainties.append(float(it["future_uncertainty_mean"]))
            paired_errors.append(float(it["future_trace_coord_error"]))
    temporal_consistency = [float(it["semantic_embedding_temporal_consistency"]) for it in valid_items if it.get("semantic_embedding_temporal_consistency") is not None]
    return {
        "item_count": len(items),
        "valid_item_count": len(valid_items),
        "valid_output_ratio": valid_ratio,
        "visibility_accuracy": visibility_acc,
        "visibility_AUROC": binary_auc(visibility_scores_for_labels, visibility_labels),
        "visibility_metric_available": bool(visibility_labels),
        "future_trace_coord_error": mean(coord_errors),
        "future_trace_coord_error_available": bool(coord_errors),
        "uncertainty_error_correlation": pearson(paired_uncertainties, paired_errors),
        "uncertainty_metric_available": len(paired_uncertainties) >= 2,
        "semantic_embedding_temporal_consistency": mean(temporal_consistency),
        "semantic_embedding_temporal_consistency_available": bool(temporal_consistency),
        "future_visibility_prob_mean": mean(visibility_probs),
    }


def run_export_mode(export_report: Path, out_report: Path, out_doc: Path, repo_root: Path) -> dict[str, Any]:
    export = load_json(export_report)
    items = export.get("items") or []
    if not isinstance(items, list):
        items = []
    breakdown = {}
    for subset in SUBSETS:
        subset_items = subset_rows(items, subset)
        if subset_items:
            breakdown[subset] = aggregate_export_items(subset_items)
    payload = {
        "generated_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "mode": "consume_future_semantic_state_export",
        "source_export": str(export_report),
        "fresh_eval_executed": True,
        "read_only_existing_per_item_eval": False,
        "future_semantic_trace_field_available": bool(export.get("future_semantic_trace_field_available", bool(items))),
        "semantic_state_eval_consumed_future_semantic_trace_state": True,
        "old_association_report_only": False,
        "export_forward_scope": export.get("forward_scope"),
        "overall": aggregate_export_items(items),
        "per_subset_breakdown": breakdown,
        "per_item_results_hash": hash_items(items),
        "exact_blocking_reason_for_visibility_auroc": None if any(it.get("target_visibility") is not None for it in items) else "export lacks target_visibility labels",
        "exact_blocking_reason_for_uncertainty_ece": "ECE is not defined for continuous trace coord error; reported uncertainty-error correlation instead",
    }
    write_json(out_report, payload)
    write_doc(out_doc, payload)
    return payload


def _required_number(item: dict[str, Any], key: str) -> float:
    if key not in item:
        raise ValueError(f"raw export item missing required field: {key}")
    value = item.get(key)
    if value is None:
        raise ValueError(f"raw export item field is null: {key}")
    if not isinstance(value, (int, float)):
        raise ValueError(f"raw export item field is not numeric: {key}")
    return float(value)


def _required_shape(item: dict[str, Any], key: str) -> list[int]:
    if key not in item:
        raise ValueError(f"raw export item missing required shape field: {key}")
    value = item.get(key)
    if not isinstance(value, list) or not all(isinstance(x, int) for x in value):
        raise ValueError(f"raw export item shape field is invalid: {key}={value}")
    return [int(x) for x in value]


def _nonzero_ratio(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if abs(float(value)) > 1e-12) / len(values)


def _nonconstant_ratio(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if abs(float(value)) > 1e-9) / len(values)


def _mean_required(items: list[dict[str, Any]], key: str) -> float:
    values = [_required_number(item, key) for item in items]
    return float(sum(values) / max(len(values), 1))


def _raw_subset_items(items: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    return subset_rows(items, subset)


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _flatten_numeric_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    out: list[float] = []
    for raw in value:
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            out.append(float(raw))
    return out


def _both_classes(labels: list[int]) -> bool:
    return bool(labels) and len({int(x) for x in labels}) >= 2


def aggregate_raw_export_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {
            "item_count": 0,
            "valid_output_ratio": 0.0,
            "output_degenerate": True,
            "target_metrics_available": False,
            "reason": "no_items",
        }
    for item in items:
        _required_shape(item, "future_trace_coord_shape")
        _required_shape(item, "future_visibility_prob_shape")
        _required_shape(item, "future_semantic_embedding_shape")
        _required_shape(item, "future_identity_embedding_shape")
        _required_shape(item, "future_uncertainty_shape")
        for key in [
            "future_trace_coord_mean",
            "future_trace_coord_std",
            "future_visibility_prob_mean",
            "future_visibility_prob_std",
            "future_semantic_embedding_norm_mean",
            "future_semantic_embedding_norm_std",
            "future_semantic_embedding_var_unit",
            "future_semantic_embedding_var_horizon",
            "future_identity_embedding_norm_mean",
            "future_identity_embedding_norm_std",
            "future_identity_embedding_var_unit",
            "future_uncertainty_mean",
            "future_uncertainty_std",
        ]:
            _required_number(item, key)

    valid_items = [item for item in items if bool(item.get("valid_output"))]
    semantic_var_unit = [_required_number(item, "future_semantic_embedding_var_unit") for item in valid_items]
    semantic_var_horizon = [_required_number(item, "future_semantic_embedding_var_horizon") for item in valid_items]
    identity_var_unit = [_required_number(item, "future_identity_embedding_var_unit") for item in valid_items]
    visibility_std = [_required_number(item, "future_visibility_prob_std") for item in valid_items]
    uncertainty_std = [_required_number(item, "future_uncertainty_std") for item in valid_items]
    semantic_norm = [_required_number(item, "future_semantic_embedding_norm_mean") for item in valid_items]
    coord_error_pairs = [
        (float(item["future_uncertainty_mean"]), float(item["future_trace_coord_error"]))
        for item in valid_items
        if item.get("future_trace_coord_error") is not None and isinstance(item.get("future_trace_coord_error"), (int, float))
    ]
    visibility_pairs = [
        (float(item["future_visibility_prob_mean"]), int(item["target_visibility"]))
        for item in valid_items
        if item.get("target_visibility") is not None and isinstance(item.get("target_visibility"), (int, float))
    ]
    visibility_scores: list[float] = []
    visibility_labels: list[int] = []
    reappearance_scores: list[float] = []
    reappearance_labels: list[int] = []
    reappearance_event_scores: list[float] = []
    reappearance_event_labels: list[int] = []
    target_qualities = []
    target_sources = []
    vis_supervised = []
    rep_supervised = []
    rep_event_supervised = []
    rep_risk_slot = []
    rep_risk_entry = []
    rep_positive_all = []
    rep_positive_at_risk = []
    vis_positive = []
    rep_positive = []
    rep_event_positive = []
    reappearance_head_flags: list[bool] = []
    reappearance_event_head_flags: list[bool] = []
    reappearance_prob_sources: list[str] = []
    mask_policies: list[str] = []
    for item in valid_items:
        target_qualities.append(str(item.get("future_visibility_target_quality", "")))
        target_sources.append(str(item.get("future_visibility_target_source", "")))
        reappearance_head_flags.append(bool(item.get("future_reappearance_head_available")))
        reappearance_event_head_flags.append(bool(item.get("future_reappearance_event_head_available")))
        if item.get("reappearance_prob_source"):
            reappearance_prob_sources.append(str(item.get("reappearance_prob_source")))
        if item.get("future_reappearance_mask_policy"):
            mask_policies.append(str(item.get("future_reappearance_mask_policy")))
        if isinstance(item.get("future_visibility_supervised_ratio"), (int, float)):
            vis_supervised.append(float(item["future_visibility_supervised_ratio"]))
        if isinstance(item.get("future_reappearance_supervised_ratio"), (int, float)):
            rep_supervised.append(float(item["future_reappearance_supervised_ratio"]))
        if isinstance(item.get("future_reappearance_event_supervised_ratio"), (int, float)):
            rep_event_supervised.append(float(item["future_reappearance_event_supervised_ratio"]))
        if isinstance(item.get("future_reappearance_risk_slot_ratio"), (int, float)):
            rep_risk_slot.append(float(item["future_reappearance_risk_slot_ratio"]))
        if isinstance(item.get("future_reappearance_risk_entry_ratio"), (int, float)):
            rep_risk_entry.append(float(item["future_reappearance_risk_entry_ratio"]))
        if isinstance(item.get("future_visibility_target_positive_rate"), (int, float)):
            vis_positive.append(float(item["future_visibility_target_positive_rate"]))
        if isinstance(item.get("future_reappearance_target_positive_rate"), (int, float)):
            rep_positive.append(float(item["future_reappearance_target_positive_rate"]))
        if isinstance(item.get("future_reappearance_positive_rate_all_slots"), (int, float)):
            rep_positive_all.append(float(item["future_reappearance_positive_rate_all_slots"]))
        if isinstance(item.get("future_reappearance_positive_rate_at_risk"), (int, float)):
            rep_positive_at_risk.append(float(item["future_reappearance_positive_rate_at_risk"]))
        if isinstance(item.get("future_reappearance_event_target_positive_rate"), (int, float)):
            rep_event_positive.append(float(item["future_reappearance_event_target_positive_rate"]))
        vs = _flatten_numeric_list(item.get("future_visibility_prob_values"))
        vl = [int(round(x)) for x in _flatten_numeric_list(item.get("future_visibility_target_values"))]
        rs = _flatten_numeric_list(item.get("future_reappearance_prob_values"))
        rl = [int(round(x)) for x in _flatten_numeric_list(item.get("future_reappearance_target_values"))]
        es = _flatten_numeric_list(item.get("future_reappearance_event_prob_values"))
        el = [int(round(x)) for x in _flatten_numeric_list(item.get("future_reappearance_event_target_values"))]
        if len(vs) == len(vl):
            visibility_scores.extend(vs)
            visibility_labels.extend(vl)
        if len(rs) == len(rl):
            reappearance_scores.extend(rs)
            reappearance_labels.extend(rl)
        if len(es) == len(el):
            reappearance_event_scores.extend(es)
            reappearance_event_labels.extend(el)
    output_degenerate = bool(
        _safe_mean(semantic_var_unit) is None
        or float(_safe_mean(semantic_var_unit) or 0.0) <= 0.0
        or float(_safe_mean(semantic_var_horizon) or 0.0) <= 0.0
        or float(_safe_mean(identity_var_unit) or 0.0) <= 0.0
        or float(_safe_mean(visibility_std) or 0.0) <= 0.0
        or float(_safe_mean(uncertainty_std) or 0.0) <= 0.0
    )
    if not visibility_scores:
        visibility_scores = [x for x, _ in visibility_pairs]
        visibility_labels = [y for _, y in visibility_pairs]
    visibility_accuracy = None
    if visibility_scores and len(visibility_scores) == len(visibility_labels):
        visibility_accuracy = _safe_mean([1.0 if (score >= 0.5) == bool(label) else 0.0 for score, label in zip(visibility_scores, visibility_labels)])
    reappearance_accuracy = None
    if reappearance_scores and len(reappearance_scores) == len(reappearance_labels):
        reappearance_accuracy = _safe_mean([1.0 if (score >= 0.5) == bool(label) else 0.0 for score, label in zip(reappearance_scores, reappearance_labels)])
    target_quality = next((x for x in target_qualities if x), "weak_unavailable")
    target_source = next((x for x in target_sources if x), "unavailable")
    both_class_visibility = _both_classes(visibility_labels)
    both_class_reappearance = _both_classes(reappearance_labels)
    both_class_reappearance_event = _both_classes(reappearance_event_labels)
    reappearance_head_available = bool(reappearance_head_flags and all(reappearance_head_flags))
    reappearance_event_head_available = bool(reappearance_event_head_flags and all(reappearance_event_head_flags))
    reappearance_prob_source = (
        "future_reappearance_logit"
        if reappearance_prob_sources and all(x == "future_reappearance_logit" for x in reappearance_prob_sources)
        else next((x for x in reappearance_prob_sources if x), "missing_reappearance_head")
    )
    mask_policy = next((x for x in mask_policies if x), "unavailable")
    calibrated_visibility_available = bool(target_quality == "strong_slot_aligned" and both_class_visibility and binary_auc(visibility_scores, visibility_labels) is not None)
    calibrated_reappearance_available = bool(
        target_quality == "strong_slot_aligned"
        and both_class_reappearance
        and reappearance_head_available
        and reappearance_prob_source == "future_reappearance_logit"
        and binary_auc(reappearance_scores, reappearance_labels) is not None
    )
    calibrated_reappearance_event_available = bool(
        target_quality == "strong_slot_aligned"
        and both_class_reappearance_event
        and reappearance_event_head_available
        and binary_auc(reappearance_event_scores, reappearance_event_labels) is not None
    )
    visibility_metric_status = (
        "calibrated_visibility_available"
        if calibrated_visibility_available
        else "target_available_but_not_strong_slot_aligned"
        if target_quality != "strong_slot_aligned" and target_quality != "weak_unavailable"
        else "target_available_but_single_class"
        if target_quality == "strong_slot_aligned" and not both_class_visibility
        else "target_unavailable"
    )
    reappearance_metric_status = (
        "calibrated_reappearance_available"
        if calibrated_reappearance_available
        else "head_missing"
        if not reappearance_head_available
        else "prob_source_not_future_reappearance_logit"
        if reappearance_prob_source != "future_reappearance_logit"
        else "target_available_but_not_strong_slot_aligned"
        if target_quality != "strong_slot_aligned" and target_quality != "weak_unavailable"
        else "target_available_but_single_class"
        if target_quality == "strong_slot_aligned" and not both_class_reappearance
        else "target_unavailable"
    )
    return {
        "item_count": len(items),
        "valid_item_count": len(valid_items),
        "valid_output_ratio": len(valid_items) / max(len(items), 1),
        "semantic_embedding_nonzero_ratio": _nonzero_ratio(semantic_norm),
        "semantic_embedding_var_unit_mean": _safe_mean(semantic_var_unit),
        "semantic_embedding_var_horizon_mean": _safe_mean(semantic_var_horizon),
        "identity_embedding_var_unit_mean": _safe_mean(identity_var_unit),
        "visibility_prob_std_mean": _safe_mean(visibility_std),
        "uncertainty_std_mean": _safe_mean(uncertainty_std),
        "uncertainty_nonconstant_ratio": _nonconstant_ratio(uncertainty_std),
        "output_degenerate": output_degenerate,
        "target_metrics_available": bool(coord_error_pairs or visibility_pairs),
        "reason": None if bool(coord_error_pairs or visibility_pairs) else "raw export lacks target coord or target visibility labels",
        "future_trace_coord_error": _safe_mean([y for _, y in coord_error_pairs]),
        "visibility_accuracy": visibility_accuracy,
        "visibility_AUROC": binary_auc(visibility_scores, visibility_labels),
        "visibility_AP": binary_ap(visibility_scores, visibility_labels),
        "future_visibility_accuracy": visibility_accuracy,
        "future_visibility_AUROC": binary_auc(visibility_scores, visibility_labels),
        "future_visibility_AP": binary_ap(visibility_scores, visibility_labels),
        "future_reappearance_accuracy": reappearance_accuracy,
        "future_reappearance_AUROC": binary_auc(reappearance_scores, reappearance_labels),
        "future_reappearance_AP": binary_ap(reappearance_scores, reappearance_labels),
        "future_reappearance_event_AUROC": binary_auc(reappearance_event_scores, reappearance_event_labels),
        "future_reappearance_event_AP": binary_ap(reappearance_event_scores, reappearance_event_labels),
        "future_reappearance_event_accuracy": _safe_mean(
            [1.0 if (score >= 0.5) == bool(label) else 0.0 for score, label in zip(reappearance_event_scores, reappearance_event_labels)]
        )
        if reappearance_event_scores and len(reappearance_event_scores) == len(reappearance_event_labels)
        else None,
        "future_reappearance_head_available": reappearance_head_available,
        "future_reappearance_event_head_available": reappearance_event_head_available,
        "reappearance_head_available": reappearance_head_available,
        "reappearance_event_head_available": reappearance_event_head_available,
        "reappearance_prob_source": reappearance_prob_source,
        "future_visibility_target_source": target_source,
        "future_visibility_target_quality": target_quality,
        "future_visibility_supervised_ratio": _safe_mean(vis_supervised),
        "future_reappearance_supervised_ratio": _safe_mean(rep_supervised),
        "future_reappearance_event_supervised_ratio": _safe_mean(rep_event_supervised),
        "future_reappearance_risk_slot_ratio": _safe_mean(rep_risk_slot),
        "future_reappearance_risk_entry_ratio": _safe_mean(rep_risk_entry),
        "future_reappearance_mask_policy": mask_policy,
        "future_visibility_positive_rate": _safe_mean(vis_positive),
        "future_reappearance_positive_rate": _safe_mean(rep_positive),
        "future_reappearance_positive_rate_all_slots": _safe_mean(rep_positive_all),
        "future_reappearance_positive_rate_at_risk": _safe_mean(rep_positive_at_risk),
        "future_reappearance_event_positive_rate": _safe_mean(rep_event_positive),
        "both_class_visibility_available": both_class_visibility,
        "both_class_reappearance_available": both_class_reappearance,
        "both_class_reappearance_event_available": both_class_reappearance_event,
        "calibrated_visibility_available": calibrated_visibility_available,
        "calibrated_reappearance_available": calibrated_reappearance_available,
        "calibrated_reappearance_event_available": calibrated_reappearance_event_available,
        "visibility_metric_status": visibility_metric_status,
        "reappearance_metric_status": reappearance_metric_status,
        "uncertainty_error_correlation": pearson([x for x, _ in coord_error_pairs], [y for _, y in coord_error_pairs]),
    }


def run_raw_export_mode(export_report: Path, out_report: Path, out_doc: Path, repo_root: Path) -> dict[str, Any]:
    export = load_json(export_report)
    if export.get("raw_export_schema_version") != "future_semantic_trace_state_raw_export_v1":
        raise ValueError(f"not a repair-v1 raw export: {export_report}")
    if bool(export.get("old_association_report_used")):
        raise ValueError("raw export unexpectedly reports old_association_report_used=true")
    items = export.get("items")
    if not isinstance(items, list):
        raise ValueError("raw export missing item list")
    overall = aggregate_raw_export_items(items)
    breakdown = {}
    for subset in SUBSETS:
        subset_items = _raw_subset_items(items, subset)
        if subset_items:
            breakdown[subset] = aggregate_raw_export_items(subset_items)
    trace_regression = bool(export.get("trace_rollout_regression_detected", False))
    export_mode = str(export.get("export_mode", "unknown"))
    full_model_mode = export_mode in {"full_model_teacher_forced", "full_model_free_rollout"}
    full_model_forward = bool(export.get("full_model_forward_executed"))
    full_free_rollout = bool(export.get("full_free_rollout_executed"))
    random_hidden_used = bool(export.get("random_hidden_used"))
    semantic_state_from_model_hidden = bool(export.get("semantic_state_from_model_hidden"))
    engineering_output_claimable = bool(
        full_model_mode
        and full_model_forward
        and not random_hidden_used
        and semantic_state_from_model_hidden
        and overall["valid_output_ratio"] >= 0.95
        and overall["output_degenerate"] is False
        and float(overall["semantic_embedding_var_unit_mean"] or 0.0) > 0.0
        and float(overall["visibility_prob_std_mean"] or 0.0) > 0.0
        and float(overall["uncertainty_nonconstant_ratio"] or 0.0) > 0.5
        and not trace_regression
    )
    payload = {
        "generated_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "mode": "consume_future_semantic_state_raw_export",
        "source_export": str(export_report),
        "raw_export_schema_version": export.get("raw_export_schema_version"),
        "export_mode": export_mode,
        "semantic_state_eval_consumed_future_semantic_trace_state": True,
        "old_association_report_only": False,
        "old_association_report_used": False,
        "raw_export_consumed": True,
        "future_semantic_trace_field_available": bool(export.get("valid_ratio", 0.0) > 0.0),
        "full_model_forward_executed": full_model_forward,
        "full_free_rollout_executed": full_free_rollout,
        "random_hidden_used": random_hidden_used,
        "semantic_state_from_model_hidden": semantic_state_from_model_hidden,
        "teacher_forced_future_semantic_state_available": bool(export_mode == "full_model_teacher_forced" and engineering_output_claimable),
        "free_rollout_semantic_state_available": bool(export_mode == "full_model_free_rollout" and engineering_output_claimable),
        "engineering_output_claimable": bool(engineering_output_claimable),
        "paper_world_model_claimable": False,
        "paper_world_model_claimable_reason": "requires medium-scale semantic-state signal judgement without trace rollout regression",
        "visibility_metric_status": str(overall.get("visibility_metric_status") or export.get("visibility_metric_status") or "target_unavailable"),
        "calibrated_visibility_available": bool(overall.get("calibrated_visibility_available")),
        "reappearance_metric_status": str(overall.get("reappearance_metric_status") or "head_missing"),
        "calibrated_reappearance_available": bool(overall.get("calibrated_reappearance_available")),
        "calibrated_reappearance_event_available": bool(overall.get("calibrated_reappearance_event_available")),
        "future_reappearance_head_available": bool(overall.get("future_reappearance_head_available")),
        "future_reappearance_event_head_available": bool(overall.get("future_reappearance_event_head_available")),
        "reappearance_head_available": bool(overall.get("reappearance_head_available")),
        "reappearance_event_head_available": bool(overall.get("reappearance_event_head_available")),
        "reappearance_prob_source": str(overall.get("reappearance_prob_source") or "missing_reappearance_head"),
        "current_export_data_source": str(export.get("current_export_data_source") or "unknown"),
        "semantic_state_signal_positive": True if engineering_output_claimable else "unclear",
        "trace_rollout_regression_detected": trace_regression,
        "overall": overall,
        "per_subset_breakdown": breakdown,
        "per_item_results_hash": hash_items(items),
        "world_model_output_now_claimable": bool(engineering_output_claimable),
        "world_model_output_now_claimable_scope": "engineering_output_only_not_paper_level",
    }
    write_json(out_report, payload)
    write_doc(out_doc, payload)
    return payload


def run_external_hardcase_export_mode(export_report: Path, out_report: Path, out_doc: Path, repo_root: Path) -> dict[str, Any]:
    export = load_json(export_report)
    if export.get("raw_export_schema_version") != "future_semantic_trace_state_raw_export_v1":
        raise ValueError(f"not a FutureSemanticTraceState raw export: {export_report}")
    if export.get("manifest_mode") != "external_hardcase_semantic_state":
        raise ValueError("external hardcase eval requires manifest_mode=external_hardcase_semantic_state")
    if bool(export.get("old_association_report_used")):
        raise ValueError("external hardcase export unexpectedly used old association report")
    if bool(export.get("stage2_val_fallback_used")):
        raise ValueError("external hardcase export used Stage2 val fallback")
    items = export.get("items")
    if not isinstance(items, list):
        raise ValueError("external hardcase export missing item list")

    target_type_counts: dict[str, int] = defaultdict(int)
    subset_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    event_scores: list[float] = []
    event_labels: list[int] = []
    candidate_correct: list[float] = []
    candidate_rr: list[float] = []
    exported_valid = 0
    metric_eligible = 0
    for item in items:
        target_type = str(item.get("target_type") or "unavailable")
        target_type_counts[target_type] += 1
        if bool(item.get("valid_output")):
            exported_valid += 1
        tags = item.get("subset_tags") if isinstance(item.get("subset_tags"), dict) else {}
        for subset in SUBSETS:
            if bool(tags.get(subset)):
                subset_items[subset].append(item)
        if not bool(item.get("valid_output")):
            continue
        if target_type in {"external_candidate_aligned", "hardcase_event", "weak_event_only"}:
            # Future implementation can provide an event score; strict blocked
            # exports intentionally do not fabricate one.
            score = item.get("event_reappearance_score")
            label = item.get("event_reappearance_target")
            if isinstance(score, (int, float)) and label in {0, 1, True, False}:
                event_scores.append(float(score))
                event_labels.append(int(label))
                metric_eligible += 1
        if target_type == "external_candidate_aligned":
            pred = item.get("predicted_candidate_index")
            target = item.get("candidate_match_target")
            scores = item.get("candidate_scores")
            if isinstance(pred, int) and isinstance(target, int):
                candidate_correct.append(1.0 if pred == target else 0.0)
                metric_eligible += 1
            if isinstance(scores, list) and isinstance(target, int) and 0 <= target < len(scores):
                order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
                rank = order.index(target) + 1 if target in order else None
                if rank:
                    candidate_rr.append(1.0 / float(rank))

    def subset_summary(sub_items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "item_count": len(sub_items),
            "valid_output_count": sum(1 for x in sub_items if bool(x.get("valid_output"))),
            "target_type_counts": dict(Counter(str(x.get("target_type") or "unavailable") for x in sub_items)),
        }

    low_sample_warning = bool(metric_eligible < 50)
    event_ap = binary_ap(event_scores, event_labels) if _both_classes(event_labels) else None
    event_auc = binary_auc(event_scores, event_labels) if _both_classes(event_labels) else None
    candidate_top1 = mean(candidate_correct)
    candidate_mrr = mean(candidate_rr)
    signal_positive: bool | str = "unclear"
    if metric_eligible >= 50 and (event_ap is not None or candidate_top1 is not None):
        signal_positive = bool((event_ap is not None and event_ap > mean(event_labels or [0]) + 0.05) or (candidate_top1 is not None and candidate_top1 > 0.0))

    payload = {
        "generated_at_utc": now_iso(),
        "repo_root": str(repo_root),
        "mode": "consume_external_hardcase_semantic_state_export",
        "source_export": str(export_report),
        "old_association_report_used": False,
        "stage2_val_fallback_used": False,
        "external_manifest_consumed": bool(export.get("external_manifest_consumed")),
        "total_items": len(items),
        "exported_valid_items": exported_valid,
        "valid_output_ratio": exported_valid / max(len(items), 1),
        "metric_eligible_items": metric_eligible,
        "target_type_counts": dict(target_type_counts),
        "low_sample_warning": low_sample_warning,
        "event_AP": event_ap,
        "event_AUROC": event_auc,
        "candidate_top1": candidate_top1,
        "candidate_MRR": candidate_mrr,
        "per_subset_breakdown": {subset: subset_summary(xs) for subset, xs in subset_items.items()},
        "external_hardcase_eval_available": bool(metric_eligible > 0 and not low_sample_warning),
        "external_hardcase_signal_positive": signal_positive,
        "exact_block_reason": export.get("blocked_due_to_no_model_input_mapping_reason") if exported_valid == 0 else None,
        "paper_world_model_claimable": False,
    }
    write_json(out_report, payload)
    write_doc(out_doc, payload)
    return payload


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Future Semantic Query Eval 20260427",
        "",
        f"- mode: `{payload.get('mode')}`",
        f"- future_semantic_trace_field_available: `{payload.get('future_semantic_trace_field_available')}`",
        f"- semantic_state_eval_consumed_future_semantic_trace_state: `{payload.get('semantic_state_eval_consumed_future_semantic_trace_state')}`",
        f"- old_association_report_only: `{payload.get('old_association_report_only')}`",
        "",
    ]
    if payload.get("mode") == "consume_external_hardcase_semantic_state_export":
        lines += [
            "## External Hardcase Metrics",
            f"- total_items: `{payload.get('total_items')}`",
            f"- exported_valid_items: `{payload.get('exported_valid_items')}`",
            f"- metric_eligible_items: `{payload.get('metric_eligible_items')}`",
            f"- stage2_val_fallback_used: `{payload.get('stage2_val_fallback_used')}`",
            f"- old_association_report_used: `{payload.get('old_association_report_used')}`",
            f"- event_AP: `{fmt(payload.get('event_AP'))}`",
            f"- event_AUROC: `{fmt(payload.get('event_AUROC'))}`",
            f"- candidate_top1: `{fmt(payload.get('candidate_top1'))}`",
            f"- candidate_MRR: `{fmt(payload.get('candidate_MRR'))}`",
            f"- external_hardcase_eval_available: `{payload.get('external_hardcase_eval_available')}`",
            f"- exact_block_reason: `{payload.get('exact_block_reason')}`",
            "",
        ]
    elif payload.get("mode") in {"consume_future_semantic_state_export", "consume_future_semantic_state_raw_export"}:
        overall = payload.get("overall", {})
        lines += [
            "## Overall Semantic-State Metrics",
            f"- item_count: `{overall.get('item_count')}`",
            f"- valid_output_ratio: `{fmt(overall.get('valid_output_ratio'))}`",
            f"- output_degenerate: `{overall.get('output_degenerate')}`",
            f"- semantic_embedding_var_unit_mean: `{fmt(overall.get('semantic_embedding_var_unit_mean'))}`",
            f"- semantic_embedding_var_horizon_mean: `{fmt(overall.get('semantic_embedding_var_horizon_mean'))}`",
            f"- identity_embedding_var_unit_mean: `{fmt(overall.get('identity_embedding_var_unit_mean'))}`",
            f"- visibility_prob_std_mean: `{fmt(overall.get('visibility_prob_std_mean'))}`",
            f"- uncertainty_std_mean: `{fmt(overall.get('uncertainty_std_mean'))}`",
            f"- visibility_accuracy: `{fmt(overall.get('visibility_accuracy'))}`",
            f"- visibility_AUROC: `{fmt(overall.get('visibility_AUROC'))}`",
            f"- calibrated_visibility_available: `{payload.get('calibrated_visibility_available')}`",
            f"- reappearance_head_available: `{payload.get('reappearance_head_available')}`",
            f"- reappearance_prob_source: `{payload.get('reappearance_prob_source')}`",
            f"- future_reappearance_AUROC: `{fmt(overall.get('future_reappearance_AUROC'))}`",
            f"- future_reappearance_AP: `{fmt(overall.get('future_reappearance_AP'))}`",
            f"- future_reappearance_event_AUROC: `{fmt(overall.get('future_reappearance_event_AUROC'))}`",
            f"- future_reappearance_event_AP: `{fmt(overall.get('future_reappearance_event_AP'))}`",
            f"- calibrated_reappearance_available: `{payload.get('calibrated_reappearance_available')}`",
            f"- calibrated_reappearance_event_available: `{payload.get('calibrated_reappearance_event_available')}`",
            f"- uncertainty_error_correlation: `{fmt(overall.get('uncertainty_error_correlation'))}`",
            f"- semantic_embedding_temporal_consistency: `{fmt(overall.get('semantic_embedding_temporal_consistency'))}`",
            f"- future_trace_coord_error: `{fmt(overall.get('future_trace_coord_error'))}`",
            "",
        ]
    else:
        lines += [
            "## Old Association Report Panels",
            "| panel | rows | top1 | MRR | false confuser | visibility metric | uncertainty metric |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
        for name, panel in payload.get("panels", {}).items():
            overall = panel.get("overall", {})
            lines.append(
                f"| {name} | {overall.get('row_count', 0)} | {fmt(overall.get('top1'))} | "
                f"{fmt(overall.get('MRR'))} | {fmt(overall.get('false_confuser_rate'))} | "
                f"{overall.get('visibility_metric_status')} | {overall.get('uncertainty_metric_status')} |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> Any:
    p = ArgumentParser(description="Evaluate STWM future semantic trace query outputs.")
    p.add_argument("--repo-root", default=None)
    p.add_argument(
        "--mode",
        default="read_old_association_report",
        choices=[
            "read_old_association_report",
            "consume_future_semantic_state_export",
            "consume_future_semantic_state_raw_export",
            "consume_external_hardcase_semantic_state_export",
        ],
    )
    p.add_argument("--source-report", default=None)
    p.add_argument("--semantic-state-export", default=None)
    p.add_argument("--out-report", default=None)
    p.add_argument("--out-doc", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    reports = repo_root / "reports"
    docs = repo_root / "docs"
    out_report = Path(args.out_report) if args.out_report else reports / "stwm_future_semantic_query_eval_20260427.json"
    out_doc = Path(args.out_doc) if args.out_doc else docs / "STWM_FUTURE_SEMANTIC_QUERY_EVAL_20260427.md"
    if args.mode == "read_old_association_report":
        source = Path(args.source_report) if args.source_report else reports / "stwm_trace_belief_eval_20260424.json"
        run_old_report_mode(source, out_report, out_doc, repo_root)
    elif args.mode == "consume_future_semantic_state_export":
        if not args.semantic_state_export:
            raise SystemExit("--semantic-state-export is required for consume_future_semantic_state_export mode")
        run_export_mode(Path(args.semantic_state_export), out_report, out_doc, repo_root)
    else:
        if not args.semantic_state_export:
            raise SystemExit("--semantic-state-export is required for this mode")
        if args.mode == "consume_external_hardcase_semantic_state_export":
            run_external_hardcase_export_mode(Path(args.semantic_state_export), out_report, out_doc, repo_root)
        else:
            run_raw_export_mode(Path(args.semantic_state_export), out_report, out_doc, repo_root)


if __name__ == "__main__":
    main()
