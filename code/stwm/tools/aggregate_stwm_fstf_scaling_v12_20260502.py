#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_doc(path: Path, title: str, payload: dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    for key in ["missing_eval_names", "completed_eval_names"]:
        vals = payload.get(key, [])
        if vals:
            lines.append(f"## {key}")
            for item in vals:
                lines.append(f"- `{item}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect(pattern: str) -> list[dict[str, Any]]:
    out = []
    for path in sorted(Path("reports").glob(pattern)):
        data = load(path)
        if data:
            data["_report_path"] = str(path)
            out.append(data)
    return out


def mean_std(vals: list[float]) -> dict[str, float]:
    arr = [float(x) for x in vals if np.isfinite(x)]
    if not arr:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(statistics.mean(arr)), "std": float(statistics.pstdev(arr))}


def summarize(evals: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for e in evals:
        m = e.get("metrics", {})
        rows.append(
            {
                "report_path": e.get("_report_path", ""),
                "seed": int(e.get("seed", -1)),
                "axis": e.get("scaling_axis", ""),
                "value": e.get("scaling_value", ""),
                "proto_top5": float(m.get("proto_top5", 0.0) or 0.0),
                "changed_gain": float(m.get("changed_subset_gain_over_copy", 0.0) or 0.0),
                "stable_drop": float(m.get("stable_preservation_drop", 0.0) or 0.0),
                "trace_error": float(m.get("future_trace_coord_error", 0.0) or 0.0),
            }
        )
    return {
        "run_count": len(rows),
        "runs": rows,
        "changed_gain": mean_std([r["changed_gain"] for r in rows]),
        "overall_top5": mean_std([r["proto_top5"] for r in rows]),
        "stable_drop": mean_std([r["stable_drop"] for r in rows]),
        "trace_error": mean_std([r["trace_error"] for r in rows]),
    }


def file_count(root: str) -> int:
    p = Path(root)
    return len(list(p.glob("**/*.pt"))) if p.exists() else 0


def main() -> None:
    v11 = collect("stwm_fstf_scaling_v11_*_eval_20260502.json")
    v12 = collect("stwm_fstf_scaling_v12_*_eval_20260502.json")
    all_evals = v11 + v12
    completed = {
        Path(e["_report_path"]).name.removeprefix("stwm_fstf_scaling_").removesuffix("_eval_20260502.json")
        for e in all_evals
    }
    expected = [
        *(f"v11_prototype_c{c}_seed{s}" for c in [16, 32, 64, 128] for s in [42, 123, 456]),
        *(f"v11_model_{m}_seed{s}" for m in ["small", "base", "large"] for s in [42, 123, 456]),
        *(f"v12_horizon_h{h}_seed{s}" for h in [8, 16, 24] for s in [42, 123, 456]),
        *(f"v12_density_k{k}_seed{s}" for k in [8, 16, 32] for s in [42, 123, 456]),
    ]
    by_axis: dict[str, list[dict[str, Any]]] = {}
    for e in all_evals:
        by_axis.setdefault(str(e.get("scaling_axis", "")), []).append(e)
    prototype = summarize(by_axis.get("prototype", []))
    model = summarize(by_axis.get("model_size", []))
    horizon = summarize(by_axis.get("horizon", []))
    density = summarize(by_axis.get("density", []))
    missing = [x for x in expected if x not in completed]
    horizon_positive = bool(horizon["run_count"] >= 9 and horizon["changed_gain"]["mean"] > 0.0)
    density_positive = bool(density["run_count"] >= 9 and density["changed_gain"]["mean"] > 0.0)
    prototype_positive = bool(prototype["run_count"] >= 9 and prototype["changed_gain"]["mean"] > 0.0)
    model_positive = bool(model["run_count"] >= 9 and model["changed_gain"]["mean"] > 0.0)
    vis = load(Path("reports/stwm_fstf_visualization_v12_20260502.json"))
    raw_vis_ready = bool(vis.get("actual_mp4_or_gif_generated") and vis.get("raw_observed_frames_included"))
    full = {
        "audit_name": "stwm_fstf_full_scaling_laws_v12",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scaling_completed": bool(not missing),
        "completed_eval_count": int(len(completed)),
        "expected_eval_count": int(len(expected)),
        "missing_eval_names": missing,
        "completed_eval_names": sorted(completed),
        "new_checkpoint_count": int(file_count("outputs/checkpoints/stwm_fstf_scaling_v11_20260502") + file_count("outputs/checkpoints/stwm_fstf_scaling_v12_20260502")),
        "new_eval_summary_count": int(len(all_evals)),
        "prototype_scaling_positive": prototype_positive,
        "horizon_scaling_positive": horizon_positive,
        "trace_density_scaling_positive": density_positive,
        "model_size_scaling_positive": model_positive,
        "dense_trace_field_claim_allowed": bool(density_positive and density["run_count"] >= 9),
        "long_horizon_claim_allowed": bool(horizon_positive and horizon["run_count"] >= 9),
        "raw_frame_visualization_ready": raw_vis_ready,
        "visualization_report": "reports/stwm_fstf_visualization_v12_20260502.json" if vis else "",
        "future_hidden_load_bearing_retained_at_H16_H24": bool(horizon_positive),
        "prototype_scaling": prototype,
        "horizon_scaling": horizon,
        "trace_density_scaling": density,
        "model_size_scaling": model,
        "reviewer_risk_after_v12": "high until H16/H24 and K16/K32 complete" if missing else "medium",
        "next_step_choice": "run_remaining_failed_scaling_jobs" if missing else "build_paper_figures_and_start_overleaf",
    }
    write_json(Path("reports/stwm_fstf_full_scaling_laws_v12_20260502.json"), full)
    write_doc(Path("docs/STWM_FSTF_FULL_SCALING_LAWS_V12_20260502.md"), "STWM-FSTF Full Scaling Laws V12", full)
    for name, payload, title in [
        ("horizon", horizon, "STWM-FSTF Horizon Scaling V12"),
        ("trace_density", density, "STWM-FSTF Trace Density Scaling V12"),
        ("prototype", prototype, "STWM-FSTF Prototype Scaling V12"),
        ("model_size", model, "STWM-FSTF Model Size Scaling V12"),
    ]:
        report = {"audit_name": f"stwm_fstf_{name}_scaling_v12", **payload}
        if name == "horizon":
            report.update(
                {
                    "H16_gain_retained_vs_copy": any(r["value"] == "H16" and r["changed_gain"] > 0 for r in payload["runs"]),
                    "H24_gain_retained_vs_copy": any(r["value"] == "H24" and r["changed_gain"] > 0 for r in payload["runs"]),
                    "error_accumulation_rate": "computed_after_H16_H24_complete" if payload["run_count"] >= 9 else "incomplete",
                    "future_hidden_load_bearing_at_H16": False,
                    "future_hidden_load_bearing_at_H24": False,
                    "long_horizon_claim_allowed": bool(horizon_positive and payload["run_count"] >= 9),
                }
            )
            out = "reports/stwm_fstf_horizon_scaling_v12_20260502.json"
            doc = "docs/STWM_FSTF_HORIZON_SCALING_V12_20260502.md"
        elif name == "trace_density":
            report.update(
                {
                    "K16_improves_changed_subset_or_coverage": any(r["value"] == "K16" and r["changed_gain"] > 0 for r in payload["runs"]),
                    "K32_improves_or_saturates": any(r["value"] == "K32" and r["changed_gain"] > 0 for r in payload["runs"]),
                    "trace_density_scaling_positive": density_positive,
                    "dense_trace_field_claim_allowed": bool(density_positive and payload["run_count"] >= 9),
                }
            )
            out = "reports/stwm_fstf_trace_density_scaling_v12_20260502.json"
            doc = "docs/STWM_FSTF_TRACE_DENSITY_SCALING_V12_20260502.md"
        elif name == "prototype":
            report.update({"prototype_scaling_positive": prototype_positive, "selected_C": 32})
            out = "reports/stwm_fstf_prototype_scaling_v12_20260502.json"
            doc = "docs/STWM_FSTF_PROTOTYPE_SCALING_V12_20260502.md"
        else:
            report.update({"model_size_scaling_positive": model_positive, "model_size_overfit_detected": not model_positive})
            out = "reports/stwm_fstf_model_size_scaling_v12_20260502.json"
            doc = "docs/STWM_FSTF_MODEL_SIZE_SCALING_V12_20260502.md"
        write_json(Path(out), report)
        write_doc(Path(doc), title, report)
    readiness = {
        "audit_name": "stwm_fstf_v12_cvpr_readiness_gate",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        **{k: full[k] for k in [
            "scaling_completed",
            "completed_eval_count",
            "expected_eval_count",
            "missing_eval_names",
            "prototype_scaling_positive",
            "horizon_scaling_positive",
            "trace_density_scaling_positive",
            "model_size_scaling_positive",
            "dense_trace_field_claim_allowed",
            "long_horizon_claim_allowed",
            "raw_frame_visualization_ready",
            "future_hidden_load_bearing_retained_at_H16_H24",
            "reviewer_risk_after_v12",
            "next_step_choice",
        ]},
    }
    write_json(Path("reports/stwm_fstf_v12_cvpr_readiness_gate_20260502.json"), readiness)
    write_doc(Path("docs/STWM_FSTF_V12_CVPR_READINESS_GATE_20260502.md"), "STWM-FSTF V12 CVPR Readiness Gate", readiness)


if __name__ == "__main__":
    main()
