#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, bootstrap_delta, dump_json, load_json, write_doc


def _aggregate_train_summary(run_reports: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audit_name": "stwm_ostf_v17_train_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(run_reports),
        "experiments": [
            {
                "experiment_name": r["experiment_name"],
                "model_kind": r["model_kind"],
                "effective_M": r["effective_M"],
                "horizon": r["horizon"],
                "seed": r["seed"],
                "checkpoint_path": r.get("checkpoint_path"),
                "loss_curve": r.get("loss_curve", []),
                "split_object_counts": r.get("split_object_counts", {}),
            }
            for r in run_reports
        ],
    }


def _to_item_map(report: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    out = {}
    for row in report.get("item_scores", []):
        out[(str(row["item_key"]), int(row.get("object_id", row["object_index"])))] = row
    return out


def main() -> int:
    cache_verification = load_json(ROOT / "reports/stwm_ostf_v17_cache_verification_20260502.json")
    run_paths = sorted((ROOT / "reports/stwm_ostf_v17_runs").glob("*.json"))
    runs = [load_json(p) for p in run_paths if "smoke_" not in p.name]
    train_summary = _aggregate_train_summary(runs)
    dump_json(ROOT / "reports/stwm_ostf_v17_train_summary_20260502.json", train_summary)
    by_name = {r["experiment_name"]: r for r in runs}
    eval_summary = {
        "audit_name": "stwm_ostf_v17_eval_summary",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_verification_passed": bool(cache_verification.get("cache_verification_passed", False)),
        "experiments": {
            r["experiment_name"]: {
                "model_kind": r["model_kind"],
                "effective_M": r["effective_M"],
                "horizon": r["horizon"],
                "seed": r["seed"],
                "test_metrics": r["test_metrics"],
                "val_metrics": r["val_metrics"],
                "metric_note": r["metric_note"],
            }
            for r in runs
        },
    }
    dump_json(ROOT / "reports/stwm_ostf_v17_eval_summary_20260502.json", eval_summary)
    m1 = by_name.get("m1_anchor_stwm_seed42_h8")
    m1_m128 = by_name.get("m1_anchor_stwm_m128_seed42_h8")
    m128 = by_name.get("ostf_multitrace_m128_seed42_h8")
    m512 = by_name.get("ostf_multitrace_m512_seed42_h8")
    pt = by_name.get("point_transformer_dense_seed42_h8")
    bootstrap = {"audit_name": "stwm_ostf_v17_bootstrap", "generated_at_utc": datetime.now(timezone.utc).isoformat()}
    if m512 and m1:
        a = _to_item_map(m512)
        b = _to_item_map(m1)
        keys = sorted(set(a) & set(b))
        bootstrap["M512_vs_M1_point_L1"] = bootstrap_delta(
            np.asarray([-a[k]["point_l1_px"] for k in keys], dtype=float),
            np.asarray([-b[k]["point_l1_px"] for k in keys], dtype=float),
        )
        bootstrap["M512_vs_M1_anchor_L1"] = bootstrap_delta(
            np.asarray([-a[k]["anchor_l1_px"] for k in keys], dtype=float),
            np.asarray([-b[k]["anchor_l1_px"] for k in keys], dtype=float),
        )
    if m128 and m1_m128:
        a = _to_item_map(m128)
        b = _to_item_map(m1_m128)
        keys = sorted(set(a) & set(b))
        bootstrap["M128_vs_M1_point_L1"] = bootstrap_delta(
            np.asarray([-a[k]["point_l1_px"] for k in keys], dtype=float),
            np.asarray([-b[k]["point_l1_px"] for k in keys], dtype=float),
        )
        bootstrap["M128_vs_M1_anchor_L1"] = bootstrap_delta(
            np.asarray([-a[k]["anchor_l1_px"] for k in keys], dtype=float),
            np.asarray([-b[k]["anchor_l1_px"] for k in keys], dtype=float),
        )
    if m512 and pt:
        a = _to_item_map(m512)
        b = _to_item_map(pt)
        keys = sorted(set(a) & set(b))
        bootstrap["M512_vs_point_transformer_point_L1"] = bootstrap_delta(
            np.asarray([-a[k]["point_l1_px"] for k in keys], dtype=float),
            np.asarray([-b[k]["point_l1_px"] for k in keys], dtype=float),
        )
    dump_json(ROOT / "reports/stwm_ostf_v17_bootstrap_20260502.json", bootstrap)
    decision = {
        "audit_name": "stwm_ostf_v17_decision",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "M128_beats_M1": bool(m128 and m1_m128 and m128["test_metrics"]["object_extent_iou"] > m1_m128["test_metrics"]["object_extent_iou"]),
        "M512_beats_M1": bool(m512 and m1 and m512["test_metrics"]["object_extent_iou"] > m1["test_metrics"]["object_extent_iou"]),
        "dense_points_load_bearing": bool(m512 and m1 and m512["test_metrics"]["object_extent_iou"] > m1["test_metrics"]["object_extent_iou"]),
        "semantic_unit_compression_helpful": bool(m512 and pt and m512["test_metrics"]["point_L1_px"] < pt["test_metrics"]["point_L1_px"]),
        "point_residual_decoder_load_bearing": bool(
            by_name.get("ostf_multitrace_m512_seed42_h8")
            and by_name.get("ostf_m512_wo_point_residual_decoder_seed42_h8")
            and by_name["ostf_multitrace_m512_seed42_h8"]["test_metrics"]["point_L1_px"]
            < by_name["ostf_m512_wo_point_residual_decoder_seed42_h8"]["test_metrics"]["point_L1_px"]
        ),
        "object_dense_semantic_trace_field_claim_allowed": False,
        "proceed_to": (
            "run_full_ostf_scaling"
            if m512 and pt and m512["test_metrics"]["point_L1_px"] < pt["test_metrics"]["point_L1_px"]
            else "improve_model_capacity_or_loss"
        ),
    }
    dump_json(ROOT / "reports/stwm_ostf_v17_decision_20260502.json", decision)
    write_doc(
        ROOT / "docs/STWM_OSTF_V17_RESULTS_20260502.md",
        "STWM OSTF V17 Results",
        {
            "cache_verification_passed": bool(cache_verification.get("cache_verification_passed", False)),
            "run_count": len(runs),
            **decision,
        },
        ["cache_verification_passed", "run_count", "M128_beats_M1", "M512_beats_M1", "dense_points_load_bearing", "semantic_unit_compression_helpful", "point_residual_decoder_load_bearing", "proceed_to"],
    )
    print("reports/stwm_ostf_v17_eval_summary_20260502.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
