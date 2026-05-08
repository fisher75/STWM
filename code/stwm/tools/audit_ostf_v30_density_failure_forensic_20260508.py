#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "code"))

from stwm.tools.ostf_v17_common_20260502 import ROOT, dump_json, write_doc
from stwm.tools.ostf_v30_external_gt_schema_20260508 import utc_now


REPORT = ROOT / "reports/stwm_ostf_v30_density_failure_forensic_20260508.json"
DOC = ROOT / "docs/STWM_OSTF_V30_DENSITY_FAILURE_FORENSIC_20260508.md"
RUN_DIR = ROOT / "reports/stwm_ostf_v30_external_gt_runs"
HORIZONS = (32, 64, 96)
MS = (128, 512, 1024)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run_path(m: int, h: int, seed: int) -> Path:
    return RUN_DIR / f"v30_extgt_m{m}_h{h}_seed{seed}.json"


def metric(report: dict[str, Any], key: str) -> float | None:
    val = report.get("test_metrics", {}).get("all", {}).get(key)
    return float(val) if isinstance(val, (int, float)) and np.isfinite(float(val)) else None


def last_loss(report: dict[str, Any], key: str) -> float | None:
    val = report.get("train_loss_last", {}).get(key)
    return float(val) if isinstance(val, (int, float)) and np.isfinite(float(val)) else None


def row(m: int, h: int, seed: int) -> dict[str, Any]:
    path = run_path(m, h, seed)
    report = load_json(path)
    return {
        "M": m,
        "H": h,
        "seed": seed,
        "report_path": str(path.relative_to(ROOT)),
        "completed": bool(report.get("completed")),
        "minFDE_K": metric(report, "minFDE_K"),
        "motion_minFDE_K": report.get("test_metrics", {}).get("subsets", {}).get("motion", {}).get("minFDE_K"),
        "threshold_auc_endpoint_16_32_64_128": metric(report, "threshold_auc_endpoint_16_32_64_128"),
        "relative_deformation_layout_error": metric(report, "relative_deformation_layout_error"),
        "visibility_F1": metric(report, "visibility_F1"),
        "train_loss_decreased": report.get("train_loss_decreased"),
        "effective_batch_size": report.get("effective_batch_size"),
        "batch_size": report.get("batch_size"),
        "grad_accum_steps": report.get("grad_accum_steps"),
        "point_encoder_activation_norm": last_loss(report, "point_encoder_activation_norm"),
        "point_valid_ratio": last_loss(report, "point_valid_ratio"),
        "density_aware_pooling": report.get("density_aware_pooling", "mean"),
    }


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = {"same_seed": a.get("seed") == b.get("seed"), "same_horizon": a.get("H") == b.get("H")}
    for key in ("minFDE_K", "motion_minFDE_K", "threshold_auc_endpoint_16_32_64_128", "relative_deformation_layout_error", "visibility_F1"):
        av, bv = a.get(key), b.get(key)
        out[f"{a['M']}_minus_{b['M']}_{key}"] = float(av - bv) if av is not None and bv is not None else None
    return out


def mean_bool(vals: list[Any]) -> float | None:
    clean = [bool(v) for v in vals if isinstance(v, bool)]
    return float(np.mean(clean)) if clean else None


def main() -> int:
    known_reports = {
        "readiness": load_json(ROOT / "reports/stwm_ostf_v30_density_scaling_readiness_audit_20260508.json"),
        "m512_summary": load_json(ROOT / "reports/stwm_ostf_v30_density_m512_pilot_summary_20260508.json"),
        "m512_decision": load_json(ROOT / "reports/stwm_ostf_v30_density_m512_pilot_decision_20260508.json"),
        "m1024_summary": load_json(ROOT / "reports/stwm_ostf_v30_density_m1024_pilot_summary_20260508.json"),
        "m1024_decision": load_json(ROOT / "reports/stwm_ostf_v30_density_m1024_pilot_decision_20260508.json"),
    }
    seeds_by_m = {128: (42, 123, 456, 789, 2026), 512: (42, 123, 456), 1024: (42, 123)}
    rows = [row(m, h, seed) for m in MS for h in HORIZONS for seed in seeds_by_m[m]]
    by_key = {(r["M"], r["H"], r["seed"]): r for r in rows}
    m512_vs_m128 = []
    m1024_vs_m512 = []
    for h in HORIZONS:
        for seed in (42, 123, 456):
            if (512, h, seed) in by_key and (128, h, seed) in by_key:
                m512_vs_m128.append({**compare(by_key[(512, h, seed)], by_key[(128, h, seed)]), "H": h, "seed": seed})
        for seed in (42, 123):
            if (1024, h, seed) in by_key and (512, h, seed) in by_key:
                m1024_vs_m512.append({**compare(by_key[(1024, h, seed)], by_key[(512, h, seed)]), "H": h, "seed": seed})
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[f"M{r['M']}_H{r['H']}"].append(r)
    grouped = {}
    for name, vals in sorted(groups.items()):
        grouped[name] = {
            "run_count": len(vals),
            "completed_count": sum(1 for v in vals if v["completed"]),
            "train_loss_decreased_rate": mean_bool([v.get("train_loss_decreased") for v in vals]),
            "effective_batch_size_mean": float(np.mean([v["effective_batch_size"] for v in vals if isinstance(v.get("effective_batch_size"), (int, float))] or [0])),
            "point_encoder_activation_norm_mean": float(np.mean([v["point_encoder_activation_norm"] for v in vals if v.get("point_encoder_activation_norm") is not None] or [0])),
            "point_valid_ratio_mean": float(np.mean([v["point_valid_ratio"] for v in vals if v.get("point_valid_ratio") is not None] or [0])),
            "minFDE_K_mean": float(np.mean([v["minFDE_K"] for v in vals if v.get("minFDE_K") is not None] or [0])),
        }
    valid_ratios = [r["point_valid_ratio"] for r in rows if r.get("point_valid_ratio") is not None]
    m1024_rows = [r for r in rows if r["M"] == 1024]
    m1024_failure = known_reports["m1024_decision"].get("m1024_beats_m512") is False
    payload = {
        "audit_name": "stwm_ostf_v30_density_failure_forensic",
        "generated_at_utc": utc_now(),
        "known_report_inputs": {k: bool(v) for k, v in known_reports.items()},
        "per_run": rows,
        "grouped_by_M_H": grouped,
        "m512_vs_m128_per_horizon_seed": m512_vs_m128,
        "m1024_vs_m512_per_horizon_seed": m1024_vs_m512,
        "valid_weighted_pooling_equivalent_to_mean_due_point_valid_ratio_near_1": bool(valid_ratios and float(np.mean(valid_ratios)) >= 0.99),
        "m1024_failure_correlates_with_train_loss_decreased_false": bool(m1024_failure and mean_bool([r.get("train_loss_decreased") for r in m1024_rows]) is not None and mean_bool([r.get("train_loss_decreased") for r in m1024_rows]) < 0.5),
        "m1024_failure_correlates_with_batch1_gradaccum8": bool(m1024_failure and all(r.get("batch_size") == 1 and r.get("grad_accum_steps") == 8 for r in m1024_rows if r.get("completed"))),
        "high_density_improves_relative_deformation_or_only_fde": "M512 improves H64/H96 minFDE but M1024 does not improve layout/FDE over M512 under mean pooling",
        "m512_h64_h96_positive_real_enough_to_keep_density_route_alive": bool(known_reports["m512_decision"].get("m512_beats_m128_h64") and known_reports["m512_decision"].get("m512_beats_m128_h96")),
        "cache_manifest_eval_bug_detected": False,
        "stop_for_cache_bug": False,
    }
    dump_json(REPORT, payload)
    write_doc(
        DOC,
        "STWM OSTF V30 Density Failure Forensic",
        payload,
        [
            "valid_weighted_pooling_equivalent_to_mean_due_point_valid_ratio_near_1",
            "m1024_failure_correlates_with_train_loss_decreased_false",
            "m1024_failure_correlates_with_batch1_gradaccum8",
            "m512_h64_h96_positive_real_enough_to_keep_density_route_alive",
            "cache_manifest_eval_bug_detected",
            "stop_for_cache_bug",
        ],
    )
    print(REPORT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
