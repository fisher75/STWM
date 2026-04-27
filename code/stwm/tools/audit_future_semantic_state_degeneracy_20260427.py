#!/usr/bin/env python3
from __future__ import annotations

from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import os


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


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Future Semantic State Degeneracy Audit Repair V1 20260427",
        "",
        f"- source_export: `{payload.get('source_export')}`",
        f"- raw_export_consumed: `{payload.get('raw_export_consumed')}`",
        f"- old_association_report_used: `{payload.get('old_association_report_used')}`",
        f"- raw_export_valid_ratio: `{payload.get('raw_export_valid_ratio')}`",
        f"- semantic_state_degenerate: `{payload.get('semantic_state_degenerate')}`",
        f"- safe_for_medium_training: `{payload.get('safe_for_medium_training')}`",
        f"- exact_failure_reason: `{payload.get('exact_failure_reason')}`",
        "",
        "## Means",
        f"- semantic_embedding_var_unit_mean: `{payload.get('semantic_embedding_var_unit_mean')}`",
        f"- semantic_embedding_var_horizon_mean: `{payload.get('semantic_embedding_var_horizon_mean')}`",
        f"- identity_embedding_var_unit_mean: `{payload.get('identity_embedding_var_unit_mean')}`",
        f"- visibility_prob_std_mean: `{payload.get('visibility_prob_std_mean')}`",
        f"- uncertainty_std_mean: `{payload.get('uncertainty_std_mean')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def fraction(values: list[bool]) -> float | None:
    return sum(1 for value in values if value) / len(values) if values else None


def item_number(item: dict[str, Any], key: str) -> float:
    value = item.get(key)
    if not finite(value):
        raise ValueError(f"raw export item missing finite numeric {key}: {value}")
    return float(value)


def audit(export_path: Path) -> dict[str, Any]:
    export = load_json(export_path)
    if export.get("raw_export_schema_version") != "future_semantic_trace_state_raw_export_v1":
        raise ValueError(f"degeneracy audit repair v1 requires raw export schema, got {export.get('raw_export_schema_version')}")
    if bool(export.get("old_association_report_used")):
        raise ValueError("raw export reports old_association_report_used=true")
    export_mode = str(export.get("export_mode", "unknown"))
    full_model_mode = export_mode in {"full_model_teacher_forced", "full_model_free_rollout"}
    items = export.get("items")
    if not isinstance(items, list):
        raise ValueError("raw export missing item list")
    valid_items = [item for item in items if isinstance(item, dict) and bool(item.get("valid_output"))]
    sem_var_unit = [item_number(item, "future_semantic_embedding_var_unit") for item in valid_items]
    sem_var_horizon = [item_number(item, "future_semantic_embedding_var_horizon") for item in valid_items]
    id_var_unit = [item_number(item, "future_identity_embedding_var_unit") for item in valid_items]
    visibility_std = [item_number(item, "future_visibility_prob_std") for item in valid_items]
    uncertainty_std = [item_number(item, "future_uncertainty_std") for item in valid_items]
    semantic_norm = [item_number(item, "future_semantic_embedding_norm_mean") for item in valid_items]
    identity_norm = [item_number(item, "future_identity_embedding_norm_mean") for item in valid_items]
    all_zero_flags = [
        abs(v) < 1e-12
        for values in [sem_var_unit, sem_var_horizon, id_var_unit, visibility_std, uncertainty_std, semantic_norm, identity_norm]
        for v in values
    ]
    constant_flags = [
        float(v) <= 1e-12
        for values in [sem_var_unit, sem_var_horizon, id_var_unit, visibility_std, uncertainty_std]
        for v in values
    ]
    nan_inf_flags = [
        not finite(v)
        for values in [sem_var_unit, sem_var_horizon, id_var_unit, visibility_std, uncertainty_std, semantic_norm, identity_norm]
        for v in values
    ]
    semantic_state_degenerate = bool(
        not valid_items
        or float(mean(sem_var_unit) or 0.0) <= 0.0
        or float(mean(sem_var_horizon) or 0.0) <= 0.0
        or float(mean(id_var_unit) or 0.0) <= 0.0
        or float(mean(visibility_std) or 0.0) <= 0.0
        or float(mean(uncertainty_std) or 0.0) <= 0.0
        or any(nan_inf_flags)
    )
    raw_export_valid_ratio = float(export.get("valid_ratio") or 0.0)
    old_assoc = bool(export.get("old_association_report_used"))
    raw_export_consumed = True
    target_metrics_available = any(item.get("future_trace_coord_error") is not None for item in valid_items) or any(
        item.get("target_visibility") is not None for item in valid_items
    )
    non_target_diagnostic_available = bool(valid_items)
    safe_for_medium_training = bool(
        full_model_mode
        and bool(export.get("full_model_forward_executed"))
        and not bool(export.get("random_hidden_used"))
        and bool(export.get("semantic_state_from_model_hidden"))
        and not semantic_state_degenerate
        and raw_export_valid_ratio >= 0.95
        and not old_assoc
        and raw_export_consumed
        and (target_metrics_available or non_target_diagnostic_available)
    )
    exact_failure_reason = None
    if not full_model_mode:
        exact_failure_reason = f"export_mode={export_mode} is not a full-model mode"
    elif bool(export.get("random_hidden_used")):
        exact_failure_reason = "random hidden was used"
    elif not bool(export.get("semantic_state_from_model_hidden")):
        exact_failure_reason = "semantic state did not come from model hidden"
    elif semantic_state_degenerate:
        exact_failure_reason = "raw FutureSemanticTraceState statistics are degenerate or non-finite"
    elif raw_export_valid_ratio < 0.95:
        exact_failure_reason = f"raw_export_valid_ratio below threshold: {raw_export_valid_ratio}"
    elif old_assoc:
        exact_failure_reason = "old association report was used"
    elif not (target_metrics_available or non_target_diagnostic_available):
        exact_failure_reason = "no target metrics or non-target diagnostics available"
    payload = {
        "generated_at_utc": now_iso(),
        "source_export": str(export_path),
        "export_mode": export_mode,
        "full_model_forward_executed": bool(export.get("full_model_forward_executed")),
        "full_free_rollout_executed": bool(export.get("full_free_rollout_executed")),
        "random_hidden_used": bool(export.get("random_hidden_used")),
        "semantic_state_from_model_hidden": bool(export.get("semantic_state_from_model_hidden")),
        "raw_export_consumed": raw_export_consumed,
        "old_association_report_used": old_assoc,
        "item_count": len(items),
        "valid_item_count": len(valid_items),
        "raw_export_valid_ratio": raw_export_valid_ratio,
        "all_zero_ratio": fraction(all_zero_flags),
        "nan_inf_ratio": fraction(nan_inf_flags),
        "constant_output_ratio": fraction(constant_flags),
        "semantic_embedding_var_unit_mean": mean(sem_var_unit),
        "semantic_embedding_var_horizon_mean": mean(sem_var_horizon),
        "identity_embedding_var_unit_mean": mean(id_var_unit),
        "visibility_prob_std_mean": mean(visibility_std),
        "uncertainty_std_mean": mean(uncertainty_std),
        "semantic_embedding_nonzero_ratio": fraction([abs(v) > 1e-12 for v in semantic_norm]),
        "identity_embedding_nonzero_ratio": fraction([abs(v) > 1e-12 for v in identity_norm]),
        "target_metrics_available": target_metrics_available,
        "non_target_diagnostic_available": non_target_diagnostic_available,
        "semantic_state_degenerate": semantic_state_degenerate,
        "engineering_output_claimable": bool(
            full_model_mode
            and bool(export.get("full_model_forward_executed"))
            and not bool(export.get("random_hidden_used"))
            and bool(export.get("semantic_state_from_model_hidden"))
            and not semantic_state_degenerate
            and raw_export_valid_ratio >= 0.95
        ),
        "paper_world_model_claimable": False,
        "paper_world_model_claimable_reason": "degeneracy audit is an engineering gate; medium-scale signal judgement is required for paper-level claim",
        "visibility_metric_status": str(export.get("visibility_metric_status") or "smoke_only_simplified_target"),
        "calibrated_visibility_available": False,
        "current_export_data_source": str(export.get("current_export_data_source") or "unknown"),
        "safe_for_medium_training": safe_for_medium_training,
        "exact_failure_reason": exact_failure_reason,
    }
    return payload


def parse_args() -> Any:
    p = ArgumentParser(description="Audit repair-v1 raw FutureSemanticTraceState export for degeneracy.")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--export", required=True)
    p.add_argument("--out-report", required=True)
    p.add_argument("--out-doc", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    resolve_repo_root(args.repo_root)
    payload = audit(Path(args.export))
    write_json(Path(args.out_report), payload)
    write_doc(Path(args.out_doc), payload)


if __name__ == "__main__":
    main()
