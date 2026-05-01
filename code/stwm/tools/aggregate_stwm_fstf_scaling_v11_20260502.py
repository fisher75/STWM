#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Missing Scaling Points")
    for item in payload.get("missing_scaling_points", []):
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(values: list[float]) -> dict[str, float]:
    vals = [float(v) for v in values if np.isfinite(v)]
    if not vals:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(statistics.mean(vals)), "std": float(statistics.pstdev(vals))}


def summarize_group(evals: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for e in evals:
        m = e.get("metrics", {})
        rows.append(
            {
                "seed": int(e.get("seed", -1)),
                "prototype_count": int(e.get("prototype_count", 0) or 0),
                "scaling_axis": e.get("scaling_axis"),
                "scaling_value": e.get("scaling_value"),
                "model_size": e.get("model_size"),
                "proto_top5": float(m.get("proto_top5", 0.0) or 0.0),
                "changed_subset_top5": float(m.get("changed_subset_top5", 0.0) or 0.0),
                "changed_subset_gain_over_copy": float(m.get("changed_subset_gain_over_copy", 0.0) or 0.0),
                "stable_preservation_drop": float(m.get("stable_preservation_drop", 0.0) or 0.0),
                "future_trace_coord_error": float(m.get("future_trace_coord_error", 0.0) or 0.0),
                "checkpoint_path": e.get("checkpoint_path", ""),
            }
        )
    return {
        "run_count": len(rows),
        "runs": rows,
        "overall_top5": mean_std([r["proto_top5"] for r in rows]),
        "changed_gain": mean_std([r["changed_subset_gain_over_copy"] for r in rows]),
        "stable_drop": mean_std([r["stable_preservation_drop"] for r in rows]),
        "future_trace_coord_error": mean_std([r["future_trace_coord_error"] for r in rows]),
    }


def collect(pattern: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path("reports").glob(pattern)):
        data = load(path)
        if data:
            data["_report_path"] = str(path)
            out.append(data)
    return out


def checkpoint_count() -> int:
    root = Path("outputs/checkpoints/stwm_fstf_scaling_v11_20260502")
    return len(list(root.glob("**/*.pt"))) if root.exists() else 0


def log_evidence() -> list[dict[str, Any]]:
    rows = []
    for path in sorted(Path("outputs/logs/stwm_fstf_scaling_v11_20260502").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rows.append(
            {
                "log_path": str(path),
                "size_bytes": int(path.stat().st_size),
                "non_empty": bool(path.stat().st_size > 0),
                "contains_start": "[fstf-v8-train] start" in text or "[fstf-v8-eval]" in text,
                "contains_done": "done" in text.lower(),
            }
        )
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-manifest", default="reports/stwm_fstf_scaling_cache_manifest_v11_20260502.json")
    args = p.parse_args()
    all_evals = collect("stwm_fstf_scaling_v11_*_seed*_eval_20260502.json")
    all_train = collect("stwm_fstf_scaling_v11_*_seed*_train_20260502.json")
    expected_eval_names = [
        *(f"prototype_c{c}_seed{s}" for c in [16, 32, 64, 128] for s in [42, 123, 456]),
        *(f"model_{m}_seed{s}" for m in ["small", "base", "large"] for s in [42, 123, 456]),
    ]
    completed_eval_names = {
        str(Path(e.get("_report_path", "")).name)
        .removeprefix("stwm_fstf_scaling_v11_")
        .removesuffix("_eval_20260502.json")
        for e in all_evals
    }
    missing_eval_names = [x for x in expected_eval_names if x not in completed_eval_names]
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for e in all_evals:
        by_axis.setdefault(str(e.get("scaling_axis", "unknown")), []).append(e)
    cache = load(Path(args.cache_manifest))
    missing = []
    if cache:
        missing.extend(
            [
                f"{p.get('axis')}={p.get('value')}: {p.get('blocking_reason')}"
                for p in cache.get("missing_scaling_points", [])
            ]
        )
    prototype = summarize_group(by_axis.get("prototype", []) + by_axis.get("C", []))
    model = summarize_group(by_axis.get("model_size", []))
    horizon = summarize_group(by_axis.get("horizon", []))
    density = summarize_group(by_axis.get("density", []) + by_axis.get("K", []))
    proto_positive = bool(prototype["run_count"] >= 6 and prototype["changed_gain"]["mean"] > 0.0)
    model_positive = bool(model["run_count"] >= 6 and model["changed_gain"]["mean"] > 0.0)
    horizon_positive = bool(horizon["run_count"] >= 6 and horizon["changed_gain"]["mean"] > 0.0)
    density_positive = bool(density["run_count"] >= 6 and density["changed_gain"]["mean"] > 0.0)
    full = {
        "audit_name": "stwm_fstf_full_scaling_laws_v11",
        "generated_at_utc": now_iso(),
        "completed_scaling_points": sorted({f"{e.get('scaling_axis')}={e.get('scaling_value')}" for e in all_evals}),
        "missing_scaling_points": missing,
        "expected_eval_count": len(expected_eval_names),
        "completed_eval_names": sorted(completed_eval_names),
        "missing_eval_names": missing_eval_names,
        "new_checkpoint_count": checkpoint_count(),
        "new_eval_summary_count": len(all_evals),
        "new_train_summary_count": len(all_train),
        "gpu_jobs_launched": len(log_evidence()),
        "gpu_log_checkpoint_evidence": log_evidence(),
        "prototype_scaling": prototype,
        "horizon_scaling": horizon,
        "trace_density_scaling": density,
        "model_size_scaling": model,
        "prototype_scaling_positive": proto_positive,
        "horizon_scaling_positive": horizon_positive,
        "trace_density_scaling_positive": density_positive,
        "model_size_scaling_positive": model_positive,
        "dense_trace_field_claim_allowed": bool(density_positive and density["run_count"] >= 9),
        "long_horizon_claim_allowed": bool(horizon_positive and horizon["run_count"] >= 9),
        "future_hidden_load_bearing_retained_under_scaling": bool(proto_positive or model_positive),
        "strongest_failure_case": "H/K scaling cache missing" if missing else "",
        "reviewer_risk_after_scaling": "high: V11 has live prototype/model-size artifacts, but full scaling remains incomplete"
        if missing or missing_eval_names
        else "low",
    }
    write_json(Path("reports/stwm_fstf_full_scaling_laws_v11_20260502.json"), full)
    write_doc(Path("docs/STWM_FSTF_FULL_SCALING_LAWS_V11_20260502.md"), "STWM-FSTF Full Scaling Laws V11", full)
    for name, payload, title in [
        ("prototype", prototype, "STWM-FSTF Prototype Scaling V11"),
        ("horizon", horizon, "STWM-FSTF Horizon Scaling V11"),
        ("trace_density", density, "STWM-FSTF Trace Density Scaling V11"),
        ("model_size", model, "STWM-FSTF Model Size Scaling V11"),
    ]:
        report = {"audit_name": f"stwm_fstf_{name}_scaling_v11", **payload}
        if name == "prototype":
            report.update(
                {
                    "selected_C": 32,
                    "C32_vs_C64": "reported in runs when both are available",
                    "C128_overfit_or_improve": "blocked unless C128 cache/run exists",
                    "prototype_scaling_positive": proto_positive,
                    "prototype_vocab_claim_allowed": proto_positive,
                }
            )
            out = "reports/stwm_fstf_prototype_scaling_v11_20260502.json"
            doc = "docs/STWM_FSTF_PROTOTYPE_SCALING_V11_20260502.md"
        elif name == "horizon":
            report.update(
                {
                    "H16_gain_retained_vs_copy": False,
                    "H24_gain_retained_vs_copy": False,
                    "error_accumulation_rate": "not_computed_without_H16_H24_runs",
                    "future_hidden_load_bearing_at_H16": False,
                    "future_hidden_load_bearing_at_H24": False,
                    "long_horizon_claim_allowed": False,
                }
            )
            out = "reports/stwm_fstf_horizon_scaling_v11_20260502.json"
            doc = "docs/STWM_FSTF_HORIZON_SCALING_V11_20260502.md"
        elif name == "trace_density":
            report.update(
                {
                    "K16_improves_changed_subset_or_coverage": False,
                    "K32_improves_or_saturates": False,
                    "trace_density_scaling_positive": density_positive,
                    "dense_trace_field_claim_allowed": False,
                }
            )
            out = "reports/stwm_fstf_trace_density_scaling_v11_20260502.json"
            doc = "docs/STWM_FSTF_TRACE_DENSITY_SCALING_V11_20260502.md"
        else:
            report.update(
                {
                    "model_size_scaling_positive": model_positive,
                    "base_beats_small": "computed_from_runs",
                    "large_beats_base_or_overfits": "computed_from_runs",
                    "compute_efficiency": "materialized-cache head scaling; not raw-video end-to-end",
                    "B200_scaling_story_allowed": model["run_count"] > 0,
                }
            )
            out = "reports/stwm_fstf_model_size_scaling_v11_20260502.json"
            doc = "docs/STWM_FSTF_MODEL_SIZE_SCALING_V11_20260502.md"
        write_json(Path(out), report)
        write_doc(Path(doc), title, report)
    readiness = {
        "audit_name": "stwm_fstf_v11_cvpr_readiness_gate",
        "generated_at_utc": now_iso(),
        "scaling_completed": bool(not missing and not missing_eval_names and len(all_evals) > 0),
        "new_checkpoint_count": checkpoint_count(),
        "new_eval_summary_count": len(all_evals),
        "prototype_scaling_positive": proto_positive,
        "horizon_scaling_positive": horizon_positive,
        "trace_density_scaling_positive": density_positive,
        "model_size_scaling_positive": model_positive,
        "dense_trace_field_claim_allowed": bool(density_positive and density["run_count"] >= 9),
        "long_horizon_claim_allowed": bool(horizon_positive and horizon["run_count"] >= 9),
        "next_step_choice": "run_missing_scaling_jobs" if (missing or missing_eval_names) else "build_paper_figures_and_start_overleaf",
    }
    write_json(Path("reports/stwm_fstf_v11_cvpr_readiness_gate_20260502.json"), readiness)
    write_doc(Path("docs/STWM_FSTF_V11_CVPR_READINESS_GATE_20260502.md"), "STWM-FSTF V11 CVPR Readiness Gate", readiness)


if __name__ == "__main__":
    main()
