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


def flatten_numbers(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, bool):
        return out
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, list):
        for item in value:
            out.extend(flatten_numbers(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_numbers(item))
    return out


def stats(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    if not values:
        return {
            "count": 0,
            "finite_count": 0,
            "nan_inf_ratio": None,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "all_zero_ratio": None,
            "constant": None,
        }
    mean = sum(finite) / len(finite) if finite else None
    var = sum((x - mean) ** 2 for x in finite) / len(finite) if finite and mean is not None else None
    std = math.sqrt(var) if var is not None else None
    return {
        "count": len(values),
        "finite_count": len(finite),
        "nan_inf_ratio": 1.0 - (len(finite) / max(len(values), 1)),
        "mean": mean,
        "std": std,
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "all_zero_ratio": sum(1 for x in finite if abs(x) < 1e-12) / max(len(finite), 1),
        "constant": bool(std is not None and std < 1e-9),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 2:
        return None
    xs2, ys2 = zip(*pairs)
    mx = sum(xs2) / len(xs2)
    my = sum(ys2) / len(ys2)
    vx = sum((x - mx) ** 2 for x in xs2)
    vy = sum((y - my) ** 2 for y in ys2)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def item_has_raw_embeddings(item: dict[str, Any]) -> bool:
    return isinstance(item.get("future_semantic_embedding"), list) and isinstance(item.get("future_identity_embedding"), list)


def audit(export_path: Path) -> dict[str, Any]:
    export = load_json(export_path)
    items = [x for x in (export.get("items") or []) if isinstance(x, dict)]

    semantic_raw = []
    identity_raw = []
    semantic_norm = []
    semantic_norm_by_horizon = []
    identity_norm = []
    visibility = []
    uncertainty = []
    uncertainty_by_horizon = []
    coord_error = []
    trace_coord = []

    for item in items:
        semantic_raw.extend(flatten_numbers(item.get("future_semantic_embedding")))
        identity_raw.extend(flatten_numbers(item.get("future_identity_embedding")))
        semantic_norm.extend(flatten_numbers(item.get("future_semantic_embedding_norm")))
        semantic_norm_by_horizon.extend(flatten_numbers(item.get("future_semantic_embedding_norm_by_horizon")))
        identity_norm.extend(flatten_numbers(item.get("future_identity_embedding_norm")))
        visibility.extend(flatten_numbers(item.get("future_visibility_prob")))
        uncertainty.extend(flatten_numbers(item.get("future_uncertainty_mean")))
        uncertainty_by_horizon.extend(flatten_numbers(item.get("future_uncertainty_by_horizon")))
        coord_error.extend(flatten_numbers(item.get("future_trace_coord_error")))
        trace_coord.extend(flatten_numbers(item.get("future_trace_coord")))

    raw_embedding_available = any(item_has_raw_embeddings(item) for item in items)
    required_raw_fields_missing = not raw_embedding_available
    semantic_distribution = stats(semantic_raw if semantic_raw else semantic_norm)
    identity_distribution = stats(identity_raw if identity_raw else identity_norm)
    visibility_distribution = stats(visibility)
    uncertainty_distribution = stats(uncertainty)
    trace_coord_distribution = stats(trace_coord)
    semantic_horizon_distribution = stats(semantic_norm_by_horizon)
    uncertainty_horizon_distribution = stats(uncertainty_by_horizon)

    numeric_degenerate = any(
        bool(block.get("constant")) or (block.get("all_zero_ratio") is not None and float(block.get("all_zero_ratio")) > 0.99)
        for block in [
            semantic_distribution,
            identity_distribution,
            visibility_distribution,
            uncertainty_distribution,
            trace_coord_distribution,
        ]
        if block.get("count", 0) > 0
    )
    nan_inf_detected = any(
        block.get("nan_inf_ratio") is not None and float(block.get("nan_inf_ratio")) > 0.0
        for block in [
            semantic_distribution,
            identity_distribution,
            visibility_distribution,
            uncertainty_distribution,
            trace_coord_distribution,
        ]
    )

    exact_failure_reason = None
    safe_for_medium_training = True
    if required_raw_fields_missing:
        safe_for_medium_training = False
        exact_failure_reason = (
            "V2 export lacks raw future_semantic_embedding/future_identity_embedding tensors; "
            "only norm summaries are available, so unit/horizon embedding degeneracy cannot be ruled out."
        )
    elif numeric_degenerate:
        safe_for_medium_training = False
        exact_failure_reason = "available semantic-state numeric outputs are constant or all-zero."
    elif nan_inf_detected:
        safe_for_medium_training = False
        exact_failure_reason = "NaN/Inf detected in semantic-state numeric outputs."

    payload = {
        "generated_at_utc": now_iso(),
        "source_export": str(export_path),
        "item_count": len(items),
        "checkpoint": export.get("checkpoint"),
        "forward_scope": export.get("forward_scope"),
        "full_stage1_stage2_forward_executed": bool(export.get("full_stage1_stage2_forward_executed")),
        "raw_embedding_available": raw_embedding_available,
        "semantic_embedding_norm_distribution": semantic_distribution,
        "semantic_embedding_variance_across_units_available": raw_embedding_available,
        "semantic_embedding_variance_across_horizon_proxy": semantic_horizon_distribution,
        "identity_embedding_norm_distribution": identity_distribution,
        "identity_embedding_variance_across_units_available": raw_embedding_available,
        "visibility_probability_distribution": visibility_distribution,
        "uncertainty_distribution": uncertainty_distribution,
        "uncertainty_by_horizon_distribution": uncertainty_horizon_distribution,
        "trace_coord_distribution": trace_coord_distribution,
        "uncertainty_error_correlation": pearson(uncertainty, coord_error),
        "all_zero_ratio": {
            "semantic": semantic_distribution.get("all_zero_ratio"),
            "identity": identity_distribution.get("all_zero_ratio"),
            "visibility": visibility_distribution.get("all_zero_ratio"),
            "uncertainty": uncertainty_distribution.get("all_zero_ratio"),
            "trace_coord": trace_coord_distribution.get("all_zero_ratio"),
        },
        "constant_output_ratio": {
            "semantic_constant": semantic_distribution.get("constant"),
            "identity_constant": identity_distribution.get("constant"),
            "visibility_constant": visibility_distribution.get("constant"),
            "uncertainty_constant": uncertainty_distribution.get("constant"),
            "trace_coord_constant": trace_coord_distribution.get("constant"),
        },
        "nan_inf_ratio": {
            "semantic": semantic_distribution.get("nan_inf_ratio"),
            "identity": identity_distribution.get("nan_inf_ratio"),
            "visibility": visibility_distribution.get("nan_inf_ratio"),
            "uncertainty": uncertainty_distribution.get("nan_inf_ratio"),
            "trace_coord": trace_coord_distribution.get("nan_inf_ratio"),
        },
        "semantic_state_degenerate": bool(numeric_degenerate or nan_inf_detected),
        "degeneracy_audit_complete": raw_embedding_available,
        "exact_failure_reason": exact_failure_reason,
        "safe_for_medium_training": bool(safe_for_medium_training),
    }
    return payload


def write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# STWM Future Semantic State Degeneracy Audit 20260427",
        "",
        f"- source_export: `{payload.get('source_export')}`",
        f"- item_count: `{payload.get('item_count')}`",
        f"- raw_embedding_available: `{payload.get('raw_embedding_available')}`",
        f"- semantic_state_degenerate: `{payload.get('semantic_state_degenerate')}`",
        f"- safe_for_medium_training: `{payload.get('safe_for_medium_training')}`",
        f"- exact_failure_reason: `{payload.get('exact_failure_reason')}`",
        "",
        "## Key Distributions",
        f"- semantic_embedding_norm_distribution: `{payload.get('semantic_embedding_norm_distribution')}`",
        f"- identity_embedding_norm_distribution: `{payload.get('identity_embedding_norm_distribution')}`",
        f"- visibility_probability_distribution: `{payload.get('visibility_probability_distribution')}`",
        f"- uncertainty_distribution: `{payload.get('uncertainty_distribution')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n")


def parse_args() -> Any:
    p = ArgumentParser(description="Audit FutureSemanticTraceState export for degenerate outputs.")
    p.add_argument("--repo-root", default=None)
    p.add_argument("--export", default=None)
    p.add_argument("--out-report", default=None)
    p.add_argument("--out-doc", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = resolve_repo_root(args.repo_root)
    export_path = Path(args.export) if args.export else repo_root / "reports" / "stwm_future_semantic_state_export_20260427.json"
    out_report = Path(args.out_report) if args.out_report else repo_root / "reports" / "stwm_future_semantic_state_degeneracy_audit_20260427.json"
    out_doc = Path(args.out_doc) if args.out_doc else repo_root / "docs" / "STWM_FUTURE_SEMANTIC_STATE_DEGENERACY_AUDIT_20260427.md"
    payload = audit(export_path)
    write_json(out_report, payload)
    write_doc(out_doc, payload)


if __name__ == "__main__":
    main()
