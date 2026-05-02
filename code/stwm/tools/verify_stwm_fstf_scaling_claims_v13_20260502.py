#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_doc(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# STWM-FSTF Scaling Claim Verification V13", ""]
    for key in [
        "corrected_prototype_scaling_positive",
        "selected_C",
        "C128_overfit_or_fail",
        "corrected_horizon_scaling_positive",
        "per_step_long_horizon_analysis_missing",
        "corrected_trace_density_scaling_positive",
        "dense_trace_field_claim_allowed",
        "corrected_model_size_scaling_positive",
        "model_size_scaling_claim_allowed",
        "future_hidden_load_bearing_retained_at_H16_H24",
        "long_horizon_claim_allowed_before_horizon_trace_audit",
    ]:
        lines.append(f"- {key}: `{payload.get(key)}`")
    lines.append("")
    lines.append("## Key Corrections")
    for item in payload.get("claim_corrections", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Group Summaries")
    for axis in ["prototype_by_C", "horizon_by_H", "trace_density_by_K", "model_size_by_scale"]:
        lines.append(f"### {axis}")
        for key, value in payload.get(axis, {}).items():
            lines.append(f"- `{key}`: {value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _metric(row: dict[str, Any], key: str) -> float:
    metrics = row.get("metrics", row)
    value = metrics.get(key)
    return float(value) if value is not None else 0.0


def _collect(pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path("reports").glob(pattern)):
        data = _load(path)
        if data:
            data["_path"] = str(path)
            rows.append(data)
    return rows


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _group(rows: list[dict[str, Any]], value_key: str = "scaling_value") -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(value_key, "")), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for name, items in sorted(grouped.items()):
        changed = [_metric(x, "changed_subset_gain_over_copy") for x in items]
        stable = [_metric(x, "stable_preservation_drop") for x in items]
        top5 = [_metric(x, "proto_top5") for x in items]
        ce = [_metric(x, "proto_ce") for x in items]
        out[name] = {
            "run_count": len(items),
            "seeds": sorted(int(x.get("seed", -1)) for x in items),
            "changed_gain_mean": _mean(changed),
            "changed_gain_std": _std(changed),
            "stable_drop_mean": _mean(stable),
            "stable_drop_std": _std(stable),
            "overall_top5_mean": _mean(top5),
            "overall_top5_std": _std(top5),
            "proto_ce_mean": _mean(ce),
            "proto_ce_std": _std(ce),
            "reports": [x["_path"] for x in items],
        }
    return out


def _safe_get(group: dict[str, dict[str, Any]], name: str, metric: str) -> float:
    return float(group.get(name, {}).get(metric, 0.0))


def main() -> int:
    prototype_rows = _collect("stwm_fstf_scaling_v11_prototype_c*_seed*_eval_20260502.json")
    model_rows = _collect("stwm_fstf_scaling_v11_model_*_seed*_eval_20260502.json")
    horizon_rows = _collect("stwm_fstf_scaling_v12_horizon_h*_seed*_eval_20260502.json")
    density_rows = _collect("stwm_fstf_scaling_v12_density_k*_seed*_eval_20260502.json")
    prototype = _group(prototype_rows)
    model = _group(model_rows)
    horizon = _group(horizon_rows)
    density = _group(density_rows)

    # C is a granularity/stability tradeoff, so do not select by top5 alone.
    c_scores: dict[str, float] = {}
    for c, row in prototype.items():
        c_scores[c] = float(row["changed_gain_mean"]) - 0.5 * float(row["stable_drop_mean"]) + 0.1 * float(row["overall_top5_mean"])
    selected_c = max(c_scores, key=c_scores.get) if c_scores else "unknown"
    c128_fails = bool(
        "C128" in prototype
        and (
            prototype["C128"]["changed_gain_mean"] <= 0.0
            or prototype["C128"]["stable_drop_mean"] > max(prototype.get("C32", {}).get("stable_drop_mean", 0.0), 0.0) + 0.05
        )
    )
    prototype_positive = bool(selected_c == "C32" and prototype.get("C32", {}).get("changed_gain_mean", 0.0) > 0.0 and c128_fails)

    h8_changed = _safe_get(horizon, "H8", "changed_gain_mean")
    h16_changed = _safe_get(horizon, "H16", "changed_gain_mean")
    h24_changed = _safe_get(horizon, "H24", "changed_gain_mean")
    h8_top5 = _safe_get(horizon, "H8", "overall_top5_mean")
    horizon_positive = bool(
        all(h in horizon and horizon[h]["run_count"] >= 3 for h in ["H8", "H16", "H24"])
        and h16_changed > 0.0
        and h24_changed > 0.0
        and _safe_get(horizon, "H16", "overall_top5_mean") >= 0.75 * max(h8_top5, 1e-6)
        and _safe_get(horizon, "H24", "overall_top5_mean") >= 0.75 * max(h8_top5, 1e-6)
    )

    k_cache = {
        "K16": _load(Path("reports/stwm_fstf_trace_density_cache_k16_v12_20260502.json")),
        "K32": _load(Path("reports/stwm_fstf_trace_density_cache_k32_v12_20260502.json")),
    }
    k8_target_coverage = 0.0
    k8_observed_coverage = 0.0
    k8_manifest = _load(Path("reports/stwm_fstf_scaling_cache_manifest_v11_20260502.json"))
    if k8_manifest:
        for entry in k8_manifest.get("trace_density", []):
            if int(entry.get("K", -1)) == 8:
                k8_target_coverage = float(entry.get("target_coverage", 0.0) or 0.0)
                k8_observed_coverage = float(entry.get("observed_semantic_memory_coverage", 0.0) or 0.0)
    if k8_target_coverage <= 0:
        # Fallback from the K8 future cache report is handled by the density-validity audit.
        k8_target_coverage = max(_safe_get(density, "K8", "overall_top5_mean"), 0.0)
    k16_cov = float(k_cache["K16"].get("target_coverage", 0.0) or 0.0)
    k32_cov = float(k_cache["K32"].get("target_coverage", 0.0) or 0.0)
    coverage_collapse = bool(k16_cov > 0 and k32_cov < 0.75 * k16_cov)
    density_positive_numeric = bool(
        all(k in density and density[k]["run_count"] >= 3 for k in ["K8", "K16", "K32"])
        and density.get("K16", {}).get("changed_gain_mean", 0.0) > density.get("K8", {}).get("changed_gain_mean", 0.0)
    )
    density_positive: str | bool = "weak_or_inconclusive" if density_positive_numeric and coverage_collapse else density_positive_numeric
    dense_claim_allowed = bool(density_positive is True and not coverage_collapse)

    small = model.get("small", {})
    base = model.get("base", {})
    large = model.get("large", {})
    base_beats_small = bool(
        base
        and small
        and base["changed_gain_mean"] > small["changed_gain_mean"]
        and base["overall_top5_mean"] > small["overall_top5_mean"]
        and base["stable_drop_mean"] <= small["stable_drop_mean"] + 1e-6
    )
    large_beats_base = bool(
        large
        and base
        and large["changed_gain_mean"] > base["changed_gain_mean"]
        and large["overall_top5_mean"] > base["overall_top5_mean"]
        and large["stable_drop_mean"] <= base["stable_drop_mean"] + 1e-6
    )
    model_positive = bool(base_beats_small or large_beats_base)

    v12 = _load(Path("reports/stwm_fstf_full_scaling_laws_v12_20260502.json"))
    horizon_trace_audit = _load(Path("reports/stwm_fstf_trace_conditioning_horizon_v13_20260502.json"))
    h16_hidden = horizon_trace_audit.get("future_hidden_load_bearing_at_H16")
    h24_hidden = horizon_trace_audit.get("future_hidden_load_bearing_at_H24")
    retained_hidden = bool(h16_hidden is True and h24_hidden is True)
    long_horizon_claim_allowed = bool(horizon_positive and retained_hidden)

    corrections = []
    if v12.get("model_size_scaling_positive") and not model_positive:
        corrections.append("model_size_scaling_positive downgraded: small/current outperforms base and large under the strict grouped rule.")
    if v12.get("dense_trace_field_claim_allowed") and not dense_claim_allowed:
        corrections.append("dense_trace_field_claim_allowed downgraded: K16/K32 numeric gains occur under reduced valid-slot coverage, so wording must remain semantic trace-unit field.")
    if v12.get("future_hidden_load_bearing_retained_at_H16_H24") and not retained_hidden:
        corrections.append("future_hidden_load_bearing_retained_at_H16_H24 downgraded until V13 H16/H24 intervention audit verifies it.")
    if c128_fails:
        corrections.append("C128 marked failed/overfit: negative changed gain and high stable drop relative to C32.")

    payload = {
        "audit_name": "stwm_fstf_scaling_claim_verification_v13",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_v12_summary": "reports/stwm_fstf_full_scaling_laws_v12_20260502.json",
        "prototype_by_C": prototype,
        "horizon_by_H": horizon,
        "trace_density_by_K": density,
        "model_size_by_scale": model,
        "prototype_selection_scores": c_scores,
        "corrected_prototype_scaling_positive": prototype_positive,
        "selected_C": int(selected_c.removeprefix("C")) if selected_c.startswith("C") else selected_c,
        "selected_C_rule": "changed_gain minus half stable_drop plus small overall_top5 tie-break; C128 rejected if changed gain is non-positive or stable drop worsens.",
        "C32_vs_C64": {
            "C32_changed_gain": prototype.get("C32", {}).get("changed_gain_mean"),
            "C64_changed_gain": prototype.get("C64", {}).get("changed_gain_mean"),
            "C32_stable_drop": prototype.get("C32", {}).get("stable_drop_mean"),
            "C64_stable_drop": prototype.get("C64", {}).get("stable_drop_mean"),
        },
        "C128_overfit_or_fail": c128_fails,
        "corrected_horizon_scaling_positive": horizon_positive,
        "H16_vs_H8": {
            "changed_gain_delta": h16_changed - h8_changed,
            "overall_top5_delta": _safe_get(horizon, "H16", "overall_top5_mean") - h8_top5,
            "stable_drop_delta": _safe_get(horizon, "H16", "stable_drop_mean") - _safe_get(horizon, "H8", "stable_drop_mean"),
        },
        "H24_vs_H8": {
            "changed_gain_delta": h24_changed - h8_changed,
            "overall_top5_delta": _safe_get(horizon, "H24", "overall_top5_mean") - h8_top5,
            "stable_drop_delta": _safe_get(horizon, "H24", "stable_drop_mean") - _safe_get(horizon, "H8", "stable_drop_mean"),
        },
        "per_step_long_horizon_analysis_missing": True,
        "corrected_trace_density_scaling_positive": density_positive,
        "trace_density_coverage_checks": {
            "K8_target_coverage_reference": k8_target_coverage,
            "K8_observed_coverage_reference": k8_observed_coverage,
            "K16_target_coverage": k16_cov,
            "K32_target_coverage": k32_cov,
            "K16_changed_count": k_cache["K16"].get("changed_count"),
            "K32_changed_count": k_cache["K32"].get("changed_count"),
            "K16_stable_count": k_cache["K16"].get("stable_count"),
            "K32_stable_count": k_cache["K32"].get("stable_count"),
            "coverage_collapse_detected": coverage_collapse,
        },
        "dense_trace_field_claim_allowed": dense_claim_allowed,
        "corrected_model_size_scaling_positive": model_positive,
        "model_size_scaling_claim_allowed": model_positive,
        "model_size_rule": "positive only if base beats small or large beats base on changed gain and overall top5 without worse stable drop.",
        "base_beats_small": base_beats_small,
        "large_beats_base": large_beats_base,
        "future_hidden_load_bearing_retained_at_H16_H24": retained_hidden,
        "long_horizon_claim_allowed_before_horizon_trace_audit": False,
        "long_horizon_claim_allowed": long_horizon_claim_allowed,
        "claim_corrections": corrections,
    }
    _dump(Path("reports/stwm_fstf_scaling_claim_verification_v13_20260502.json"), payload)
    _write_doc(Path("docs/STWM_FSTF_SCALING_CLAIM_VERIFICATION_V13_20260502.md"), payload)
    print("reports/stwm_fstf_scaling_claim_verification_v13_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
